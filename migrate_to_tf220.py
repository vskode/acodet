import tensorflow as tf
import numpy as np
import collections
from acodet.models import FBetaScore # Ensure this import works for loading

# ==============================================================================
# 1. ROBUST CLASS DEFINITIONS (Keras 3 Compatible)
# ==============================================================================

Config = collections.namedtuple(
    "Config",
    ["stft_frame_length", "stft_frame_step", "freq_bins", "sample_rate", "lower_f", "upper_f"],
)
Config.__new__.__defaults__ = (1024, 300, 64, 10000.0, 0.0, 5000.0)

@tf.keras.utils.register_keras_serializable(package="Whale")
class MelSpectrogram(tf.keras.layers.Layer):
    def __init__(self, config=None, name="mel_spectrogram", **kwargs):
        super(MelSpectrogram, self).__init__(name=name, **kwargs)
        if config is None:
            config = Config()
        elif isinstance(config, dict):
            config = Config(**config)
        self.config = config

    def build(self, input_shape):
        self._stft = tf.keras.layers.Lambda(
            lambda t: tf.signal.stft(
                tf.squeeze(t, 2),
                frame_length=self.config.stft_frame_length,
                frame_step=self.config.stft_frame_step,
            ),
            name="stft",
        )
        num_spectrogram_bins = self.config.stft_frame_length // 2 + 1
        self._bin = tf.keras.layers.Lambda(
            lambda t: tf.square(
                tf.tensordot(
                    tf.abs(t),
                    tf.signal.linear_to_mel_weight_matrix(
                        num_mel_bins=self.config.freq_bins,
                        num_spectrogram_bins=num_spectrogram_bins,
                        sample_rate=self.config.sample_rate,
                        lower_edge_hertz=self.config.lower_f,
                        upper_edge_hertz=self.config.upper_f,
                        name="matrix",
                    ),
                    1,
                )
            ),
            name="mel_bins",
        )

    def call(self, inputs):
        return self._bin(self._stft(inputs))

    def get_config(self):
        config = super().get_config()
        config.update({"config": self.config._asdict()})
        return config

@tf.keras.utils.register_keras_serializable(package="Whale")
class PCEN(tf.keras.layers.Layer):
    # Fixed: Removed conflicting _trainable logic. 
    # We use a distinct name 'weights_trainable' to avoid collision with Keras 'trainable' property.
    def __init__(self, alpha=0.98, smooth_coef=0.04, delta=2.0, root=2.0, floor=1e-6, weights_trainable=True, name="PCEN", **kwargs):
        super().__init__(name=name, **kwargs)
        self._alpha_init = alpha
        self._delta_init = delta
        self._root_init = root
        self._smooth_coef = smooth_coef
        self._floor = floor
        self.weights_trainable = weights_trainable

    def build(self, input_shape):
        num_channels = input_shape[-1]
        self.alpha = self.add_weight(
            name="alpha", 
            shape=[num_channels], 
            initializer=tf.keras.initializers.Constant(self._alpha_init), 
            trainable=self.weights_trainable
        )
        self.delta = self.add_weight(
            name="delta", 
            shape=[num_channels], 
            initializer=tf.keras.initializers.Constant(self._delta_init), 
            trainable=self.weights_trainable
        )
        self.root = self.add_weight(
            name="root", 
            shape=[num_channels], 
            initializer=tf.keras.initializers.Constant(self._root_init), 
            trainable=self.weights_trainable
        )
        self.ema = tf.keras.layers.SimpleRNN(
            units=num_channels, activation=None, use_bias=False,
            kernel_initializer=tf.keras.initializers.Identity(gain=self._smooth_coef),
            recurrent_initializer=tf.keras.initializers.Identity(gain=1.0 - self._smooth_coef),
            return_sequences=True, trainable=False,
        )

    def call(self, inputs):
        # PCEN Logic
        alpha = tf.math.minimum(self.alpha, 1.0)
        root = tf.math.maximum(self.root, 1.0)
        ema_smoother = self.ema(inputs, initial_state=tf.gather(inputs, 0, axis=1))
        one_over_root = 1.0 / root
        output = (inputs / (self._floor + ema_smoother) ** alpha + self.delta) ** one_over_root - self.delta**one_over_root
        return output

    def get_config(self):
        config = super().get_config()
        config.update({
            "alpha": self._alpha_init, 
            "smooth_coef": self._smooth_coef, 
            "delta": self._delta_init, 
            "root": self._root_init, 
            "floor": self._floor, 
            "weights_trainable": self.weights_trainable
        })
        return config

def Conv2D(filters, kernel_size, strides=(1, 1), padding="VALID", name=None):
    return tf.keras.layers.Conv2D(filters, kernel_size, strides, padding=padding, activation=None, use_bias=False, name=name)

@tf.keras.utils.register_keras_serializable(package="Whale")
class ResidualPath(tf.keras.layers.Layer):
    def __init__(self, num_output_channels, input_stride, **kwargs):
        super().__init__(**kwargs)
        self.num_output_channels = num_output_channels
        self.input_stride = input_stride

    def build(self, input_shape):
        num_input_channels = input_shape[-1]
        self._layers = []
        if num_input_channels != self.num_output_channels:
            self._layers = [
                Conv2D(self.num_output_channels, 1, self.input_stride, name="conv_residual"),
                tf.keras.layers.BatchNormalization(epsilon=1e-4, momentum=0.9997, scale=False, center=True, name="batch_normalization_residual"),
            ]

    def call(self, inputs):
        t = inputs
        for layer in self._layers: t = layer(t)
        return t

    def get_config(self):
        config = super().get_config()
        config.update({"num_output_channels": self.num_output_channels, "input_stride": self.input_stride})
        return config

@tf.keras.utils.register_keras_serializable(package="Whale")
class MainPath(tf.keras.layers.Layer):
    def __init__(self, num_inner_channels, num_output_channels, input_stride, **kwargs):
        super().__init__(**kwargs)
        self.num_inner_channels = num_inner_channels
        self.num_output_channels = num_output_channels
        self.input_stride = input_stride

    def build(self, input_shape):
        num_input_channels = input_shape[-1]
        pad = "VALID" if num_input_channels != self.num_output_channels else "SAME"
        self._layers = [
            Conv2D(self.num_inner_channels, 1, self.input_stride, padding=pad, name="conv_bottleneck"),
            tf.keras.layers.BatchNormalization(epsilon=1e-4, momentum=0.9997, scale=False, center=True, name="batch_normalization_bottleneck"),
            tf.keras.layers.ReLU(name="relu_bottleneck"),
            Conv2D(self.num_inner_channels, 3, 1, padding="SAME", name="conv"),
            tf.keras.layers.BatchNormalization(epsilon=1e-4, momentum=0.9997, scale=False, center=True, name="batch_normalization"),
            tf.keras.layers.ReLU(name="relu"),
            Conv2D(self.num_output_channels, 1, 1, padding="SAME", name="conv_output"),
            tf.keras.layers.BatchNormalization(epsilon=1e-4, momentum=0.9997, scale=False, center=True, name="batch_normalization_output"),
        ]

    def call(self, inputs):
        t = inputs
        for layer in self._layers: t = layer(t)
        return t

    def get_config(self):
        config = super().get_config()
        config.update({"num_inner_channels": self.num_inner_channels, "num_output_channels": self.num_output_channels, "input_stride": self.input_stride})
        return config

@tf.keras.utils.register_keras_serializable(package="Whale")
class Block(tf.keras.layers.Layer):
    def __init__(self, num_inner_channels, num_output_channels, input_stride=1, name="block", **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_inner_channels = num_inner_channels
        self.num_output_channels = num_output_channels
        self.input_stride = input_stride

    def build(self, input_shape):
        self._residual_path = ResidualPath(self.num_output_channels, self.input_stride)
        self._main_path = MainPath(self.num_inner_channels, self.num_output_channels, self.input_stride)
        self._activation = tf.keras.layers.ReLU(name="relu_output")

    def call(self, features):
        return self._activation(self._residual_path(features) + self._main_path(features))

    def get_config(self):
        config = super().get_config()
        config.update({"num_inner_channels": self.num_inner_channels, "num_output_channels": self.num_output_channels, "input_stride": self.input_stride})
        return config

# ==============================================================================
# 2. ROBUST BUILD AND MIGRATE
# ==============================================================================

def build_and_migrate(legacy_path, output_path="flat_whales_tf220.h5"):
    print("1. Loading Legacy Model (this may take a moment)...")
    legacy_model = tf.keras.models.load_model(
        legacy_path,
        custom_objects={"Addons>FBetaScore": FBetaScore}
    )

    print("2. Extracting weights from Legacy layers...")
    # STRATEGY: Instead of relying on layer indices (which are messy in your legacy model),
    # we collect ONLY the layers that actually have weights.
    # This aligns perfectly because stateless layers (Input, Lambda, MaxPool) are skipped automatically.
    
    source_weights_queue = []
    
    for i, layer in enumerate(legacy_model.layers):
        weights = layer.get_weights()
        if len(weights) > 0:
            print(f"   Found weights in Legacy Layer {i}: {layer.__class__.__name__} (Shapes: {[w.shape for w in weights]})")
            source_weights_queue.append(weights)
    
    print(f"   Total stateful layers found: {len(source_weights_queue)}")

    print("3. Building New Flat Architecture...")
    layers_list = []
    
    # --- A. Frontend (Audio Array -> MelSpec -> PCEN) ---
    # Input: Raw Audio Array
    # layers_list.append(tf.keras.Input(shape=(7755,), name="audio_input")) 
    
    # # Expand to 3D for Stft: (Batch, Time, 1)
    # layers_list.append(tf.keras.layers.Lambda(lambda t: tf.expand_dims(t, -1), name="expand_dims_audio"))
    
    # # MelSpectrogram (Stateless in terms of trainable variables)
    # layers_list.append(MelSpectrogram(name="mel_spectrogram"))
    
    # PCEN (Has weights: Alpha, Delta, Root) -> MATCH 1
    layers_list.append(PCEN(alpha=0.98, delta=2.0, root=2.0, smooth_coef=0.025, floor=1e-6, weights_trainable=True, name="pcen"))
    
    # --- B. Stem (Pre-ResNet) ---
    # Legacy PCEN output was likely (Batch, T, F). 
    # Conv2D needs (Batch, T, F, 1).
    layers_list.append(tf.keras.layers.Lambda(lambda t: tf.expand_dims(t, 3), name="expand_dims_cnn"))
    
    # Stem Components -> MATCH 2, 3
    layers_list.append(Conv2D(64, 7, padding="SAME", name="stem_conv"))
    layers_list.append(tf.keras.layers.BatchNormalization(epsilon=1e-4, momentum=0.9997, scale=False, center=True, name="stem_bn"))
    layers_list.append(tf.keras.layers.ReLU(name="stem_relu"))
    layers_list.append(tf.keras.layers.MaxPool2D(3, 2, padding="SAME", name="stem_pool"))
    
    # --- C. ResNet Blocks ---
    # Group 1
    layers_list.append(Block(64, 256, input_stride=1, name="g0_b0"))
    for i in range(1, 3): layers_list.append(Block(64, 256, 1, name=f"g0_b{i}"))
    
    # Group 2
    layers_list.append(Block(128, 512, input_stride=2, name="g1_b0"))
    for i in range(1, 4): layers_list.append(Block(128, 512, 1, name=f"g1_b{i}"))
    
    # Group 3
    layers_list.append(Block(256, 1024, input_stride=2, name="g2_b0"))
    for i in range(1, 6): layers_list.append(Block(256, 1024, 1, name=f"g2_b{i}"))
    
    # Group 4
    layers_list.append(Block(512, 2048, input_stride=2, name="g3_b0"))
    for i in range(1, 3): layers_list.append(Block(512, 2048, 1, name=f"g3_b{i}"))
    
    # --- D. Head ---
    layers_list.append(tf.keras.layers.GlobalAveragePooling2D(name="global_pool"))
    layers_list.append(tf.keras.layers.Dense(1, name="logits"))
    layers_list.append(tf.keras.layers.Activation("sigmoid", name="predictions"))
    
    # Assemble
    flat_model = tf.keras.Sequential(layers_list)
    flat_model.build(input_shape=(None, 128, 64))

    print("4. Injecting Weights...")
    
    weight_idx = 0
    for layer in flat_model.layers:
        if not layer.weights:
            continue
            
        if weight_idx >= len(source_weights_queue):
            print(f"CRITICAL WARNING: New model has more layers than old model. Stopped at {layer.name}.")
            break
            
        source_w = source_weights_queue[weight_idx]
        dest_w_shapes = [w.shape for w in layer.get_weights()]
        source_w_shapes = [w.shape for w in source_w]
        
        # Shape check
        if str(dest_w_shapes) == str(source_w_shapes):
            layer.set_weights(source_w)
            print(f"  [OK] Transferred to {layer.name}")
            weight_idx += 1
        else:
            print(f"  [MISMATCH] {layer.name} expects {dest_w_shapes} but got {source_w_shapes}")
            print("  Aborting to prevent corrupted model.")
            return

    if weight_idx == len(source_weights_queue):
        print("Success! All weights matched and transferred.")
    else:
        print(f"Warning: Leftover weights in source? Transferred {weight_idx}/{len(source_weights_queue)}")

    print(f"5. Saving to {output_path}...")
    flat_model.save(output_path, save_format="h5")
    print("DONE. You can now use this .h5 file in your TF 2.20 environment.")

# Execution
legacy_saved_model_path = "acodet/src/models/Humpback_20221130" 
# build_and_migrate(legacy_saved_model_path, "flat_whale_model.h5")



# array([[0.996311  ],
#        [0.99999076],
#        [0.5851846 ]], dtype=float32)

# array([[0.996311  ],
#        [0.99999076],
#        [0.5851846 ]], dtype=float32)


# .keras model with model.model(audio, training=False)
# <tf.Tensor: shape=(3, 1), dtype=float32, numpy=
# array([[0.6614784 ],
#        [0.66664135],
#        [0.669127  ]], dtype=float32)>


# .h5 with model.model(audio, training=False)
# <tf.Tensor: shape=(3, 1), dtype=float32, numpy=
# array([[0.996311  ],
#        [0.99999076],
#        [0.58518505]], dtype=float32)>
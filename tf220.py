import os
import tensorflow as tf
import keras  # Explicitly using Keras 3
import collections
from acodet import global_config as conf

# ==============================================================================
# 1. DEFINE LAYERS (Fixed: Lowercase padding)
# ==============================================================================

# We keep Config for compatibility, though we pass args manually to PCEN
Config = collections.namedtuple(
    "Config",
    ["stft_frame_length", "stft_frame_step", "freq_bins", "sample_rate", "lower_f", "upper_f"],
)
Config.__new__.__defaults__ = (
    conf.STFT_FRAME_LEN,
    conf.FFT_HOP,
    64,
    conf.SR,
    0.0,
    conf.SR / 2,
    )
# Add this import at the top of your script if not present
from keras import ops

# ... imports remain the same ...

@keras.saving.register_keras_serializable(package="Whale")
class PCEN(keras.layers.Layer):
    def __init__(self, alpha=0.98, smooth_coef=0.025, delta=2.0, root=2.0, floor=1e-6, weights_trainable=True, name="PCEN", **kwargs):
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
            initializer=keras.initializers.Constant(self._alpha_init), 
            trainable=self.weights_trainable
        )
        self.delta = self.add_weight(
            name="delta", 
            shape=[num_channels], 
            initializer=keras.initializers.Constant(self._delta_init), 
            trainable=self.weights_trainable
        )
        self.root = self.add_weight(
            name="root", 
            shape=[num_channels], 
            initializer=keras.initializers.Constant(self._root_init), 
            trainable=self.weights_trainable
        )
        # We also force the RNN to be built, though we might run it on CPU
        self.ema = keras.layers.SimpleRNN(
            units=num_channels, activation=None, use_bias=False,
            kernel_initializer=keras.initializers.Identity(gain=self._smooth_coef),
            recurrent_initializer=keras.initializers.Identity(gain=1.0 - self._smooth_coef),
            return_sequences=True, trainable=False,
        )

    def call(self, inputs):
        # FORCE CPU EXECUTION to bypass "libdevice not found" error
        with tf.device("/cpu:0"):
            alpha = ops.minimum(self.alpha, 1.0)
            root = ops.maximum(self.root, 1.0)
            
            initial_state = ops.take(inputs, 0, axis=1)
            ema_smoother = self.ema(inputs, initial_state=initial_state)
            
            one_over_root = 1.0 / root
            
            # These operations will now run safely on the CPU
            term1 = ops.power(self._floor + ema_smoother, alpha)
            term2 = ops.power(inputs / term1 + self.delta, one_over_root)
            term3 = ops.power(self.delta, one_over_root)
            
            return term2 - term3

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

def Conv2D(filters, kernel_size, strides=(1, 1), padding="valid", name=None):
    # Keras 3 REQUIRES lowercase 'valid'/'same'
    return keras.layers.Conv2D(filters, kernel_size, strides=strides, padding=padding, activation=None, use_bias=False, name=name)

@keras.saving.register_keras_serializable(package="Whale")
class ResidualPath(keras.layers.Layer):
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
                keras.layers.BatchNormalization(epsilon=1e-4, momentum=0.9997, scale=False, center=True, name="batch_normalization_residual"),
            ]

    def call(self, inputs):
        t = inputs
        for layer in self._layers: t = layer(t)
        return t

    def get_config(self):
        config = super().get_config()
        config.update({"num_output_channels": self.num_output_channels, "input_stride": self.input_stride})
        return config

@keras.saving.register_keras_serializable(package="Whale")
class MainPath(keras.layers.Layer):
    def __init__(self, num_inner_channels, num_output_channels, input_stride, **kwargs):
        super().__init__(**kwargs)
        self.num_inner_channels = num_inner_channels
        self.num_output_channels = num_output_channels
        self.input_stride = input_stride

    def build(self, input_shape):
        num_input_channels = input_shape[-1]
        # FIXED: Lowercase "valid" and "same"
        pad = "valid" if num_input_channels != self.num_output_channels else "same"
        
        self._layers = [
            Conv2D(self.num_inner_channels, 1, self.input_stride, padding=pad, name="conv_bottleneck"),
            keras.layers.BatchNormalization(epsilon=1e-4, momentum=0.9997, scale=False, center=True, name="batch_normalization_bottleneck"),
            keras.layers.ReLU(name="relu_bottleneck"),
            # FIXED: Lowercase "same"
            Conv2D(self.num_inner_channels, 3, 1, padding="same", name="conv"),
            keras.layers.BatchNormalization(epsilon=1e-4, momentum=0.9997, scale=False, center=True, name="batch_normalization"),
            keras.layers.ReLU(name="relu"),
            # FIXED: Lowercase "same"
            Conv2D(self.num_output_channels, 1, 1, padding="same", name="conv_output"),
            keras.layers.BatchNormalization(epsilon=1e-4, momentum=0.9997, scale=False, center=True, name="batch_normalization_output"),
        ]

    def call(self, inputs):
        t = inputs
        for layer in self._layers: t = layer(t)
        return t

    def get_config(self):
        config = super().get_config()
        config.update({"num_inner_channels": self.num_inner_channels, "num_output_channels": self.num_output_channels, "input_stride": self.input_stride})
        return config

@keras.saving.register_keras_serializable(package="Whale")
class Block(keras.layers.Layer):
    def __init__(self, num_inner_channels, num_output_channels, input_stride=1, name="block", **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_inner_channels = num_inner_channels
        self.num_output_channels = num_output_channels
        self.input_stride = input_stride

    def build(self, input_shape):
        self._residual_path = ResidualPath(self.num_output_channels, self.input_stride)
        self._main_path = MainPath(self.num_inner_channels, self.num_output_channels, self.input_stride)
        self._activation = keras.layers.ReLU(name="relu_output")

    def call(self, features):
        return self._activation(self._residual_path(features) + self._main_path(features))

    def get_config(self):
        config = super().get_config()
        config.update({"num_inner_channels": self.num_inner_channels, "num_output_channels": self.num_output_channels, "input_stride": self.input_stride})
        return config

# ==============================================================================
# 2. LOAD SCRIPT
# ==============================================================================

def load_spectrogram_model(h5_path):
    print("1. Constructing Model Architecture (Input -> PCEN)...")
    
    layers_list = []
    
    # --- 1. Input: Expects Spectrogram (128, 64) ---
    # This matches your 'change_input_to_array' intent later
    layers_list.append(keras.Input(shape=(128, 64), name="spectrogram_input"))
    
    # --- 2. PCEN ---
    layers_list.append(PCEN(alpha=0.98, delta=2.0, root=2.0, smooth_coef=0.025, floor=1e-6, weights_trainable=True, name="pcen"))
    
    # --- 3. Lambda (Expand Dims) ---
    # PCEN output is (Batch, 128, 64). Conv2D needs (Batch, 128, 64, 1).
    # layers_list.append(keras.layers.Lambda(lambda t: tf.expand_dims(t, -1), name="expand_dims"))
    layers_list.append(keras.layers.Reshape((128, 64, 1), name="expand_dims"))
    
    # --- 4. Stem ---
    # FIXED: padding="same" (lowercase)
    layers_list.append(Conv2D(64, 7, padding="same", name="stem_conv"))
    layers_list.append(keras.layers.BatchNormalization(epsilon=1e-4, momentum=0.9997, scale=False, center=True, name="stem_bn"))
    layers_list.append(keras.layers.ReLU(name="stem_relu"))
    layers_list.append(keras.layers.MaxPool2D(3, 2, padding="same", name="stem_pool"))
    
    # --- 5. Blocks ---
    # Group 0 (3 blocks)
    layers_list.append(Block(64, 256, input_stride=1, name="g0_b0"))
    for i in range(1, 3): layers_list.append(Block(64, 256, 1, name=f"g0_b{i}"))
    
    # Group 1 (4 blocks)
    layers_list.append(Block(128, 512, input_stride=2, name="g1_b0"))
    for i in range(1, 4): layers_list.append(Block(128, 512, 1, name=f"g1_b{i}"))
    
    # Group 2 (6 blocks)
    layers_list.append(Block(256, 1024, input_stride=2, name="g2_b0"))
    for i in range(1, 6): layers_list.append(Block(256, 1024, 1, name=f"g2_b{i}"))
    
    # Group 3 (3 blocks)
    layers_list.append(Block(512, 2048, input_stride=2, name="g3_b0"))
    for i in range(1, 3): layers_list.append(Block(512, 2048, 1, name=f"g3_b{i}"))
    
    # --- 6. Head ---
    layers_list.append(keras.layers.GlobalAveragePooling2D(name="global_pool"))
    layers_list.append(keras.layers.Dense(1, name="logits"))
    layers_list.append(keras.layers.Activation("sigmoid", name="predictions"))
    
    model = keras.Sequential(layers_list)
    
    # Build to initialize weights (so load_weights has somewhere to put them)
    model.build(input_shape=(None, 128, 64))
    
    print(f"2. Loading weights from {h5_path}...")
    try:
        # We use load_weights, NOT load_model. 
        # This bypasses the 'Layer count mismatch' on architecture parsing.
        # It blindly pours weights into the layers we just built.
        model.load_weights(h5_path)
        print("   SUCCESS: Weights loaded.")
    except Exception as e:
        print(f"   ERROR loading weights: {e}")
        return None
    
    return model

# Usage
path_to_h5 = "flat_whale_model.h5" 
# model = load_spectrogram_model(path_to_h5)
# import keras
# keras.models.save_model(model, 'test.keras')

if False:#model:
    # Test with dummy spectrogram input (Batch, Time, Freq)
    dummy_input = tf.random.normal((1, 128, 64))
    pred = model(dummy_input)
    print(f"Test prediction shape: {pred.shape}")
    print(f"First layer is: {model.layers[0].name}") # Should be 'pcen'
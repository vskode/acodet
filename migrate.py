import numpy as np
import tensorflow as tf
import collections
from acodet import global_config as conf

# ==============================================================================
# 1. ARCHITECTURE DEFINITION (Aligned to your NPZ names)
# ==============================================================================

Config = collections.namedtuple(
    "Config",
    ["stft_frame_length", 
     "stft_frame_step", 
     "freq_bins", 
     "sample_rate", 
     "lower_f", 
     "upper_f"],
)
Config.__new__.__defaults__ = (
    conf.STFT_FRAME_LEN,
    conf.FFT_HOP,
    64,
    conf.SR,
    0.0,
    conf.SR / 2,
    )

@tf.keras.utils.register_keras_serializable(package="Whale")
class PCEN(tf.keras.layers.Layer):
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
        self.alpha = self.add_weight(name="alpha", shape=[num_channels], initializer=tf.keras.initializers.Constant(self._alpha_init), trainable=self.weights_trainable)
        self.delta = self.add_weight(name="delta", shape=[num_channels], initializer=tf.keras.initializers.Constant(self._delta_init), trainable=self.weights_trainable)
        self.root = self.add_weight(name="root", shape=[num_channels], initializer=tf.keras.initializers.Constant(self._root_init), trainable=self.weights_trainable)
        
        # Explicitly named 'simple_rnn' to match NPZ key: pcen/simple_rnn/...
        self.ema = tf.keras.layers.SimpleRNN(units=num_channels, activation=None, use_bias=False, return_sequences=True, trainable=False, name="simple_rnn")
        self.ema.build((None, None, num_channels)) 

    def call(self, inputs):
        # 1. Smooth
        ema = self.ema(inputs)
        # 2. Normalize
        return (inputs / (self._floor + ema)**self.alpha + self.delta)**self.root - self.delta**self.root

    def get_config(self):
        config = super().get_config()
        config.update({
            "alpha": self._alpha_init,
            "delta": self._delta_init,
            "root": self._root_init,
            "smooth_coef": self._smooth_coef,
            "floor": self._floor,
            "weights_trainable": self.weights_trainable,
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
        if input_shape[-1] != self.num_output_channels:
            self._layers = [
                Conv2D(self.num_output_channels, 1, self.input_stride, name="conv_residual"),
                tf.keras.layers.BatchNormalization(epsilon=1e-4, momentum=0.9997, scale=False, center=True, name="batch_normalization_residual")
            ]
        else:
            self._layers = []

    def call(self, inputs):
        t = inputs
        for layer in self._layers:
            t = layer(t)
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
        pad = "VALID" if num_input_channels != self.num_output_channels else "same"
        self._layers = [
            Conv2D(self.num_inner_channels, 1, self.input_stride, padding=pad, name="conv_bottleneck"),
            tf.keras.layers.BatchNormalization(epsilon=1e-4, momentum=0.9997, scale=False, center=True, name="batch_normalization_bottleneck"),
            tf.keras.layers.ReLU(name="relu_bottleneck"),
            Conv2D(self.num_inner_channels, 3, 1, padding="same", name="conv"),
            tf.keras.layers.BatchNormalization(epsilon=1e-4, momentum=0.9997, scale=False, center=True, name="batch_normalization"),
            tf.keras.layers.ReLU(name="relu"),
            Conv2D(self.num_output_channels, 1, 1, padding="same", name="conv_output"),
            tf.keras.layers.BatchNormalization(epsilon=1e-4, momentum=0.9997, scale=False, center=True, name="batch_normalization_output"),
        ]

    def call(self, inputs):
        t = inputs
        for layer in self._layers:
            t = layer(t)
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
        # Explicit names are CRITICAL for matching the NPZ keys like 'g0_b0/residual_path/...'
        self._residual_path = ResidualPath(self.num_output_channels, self.input_stride, name="residual_path")
        self._main_path = MainPath(self.num_inner_channels, self.num_output_channels, self.input_stride, name="main_path")
        self._activation = tf.keras.layers.ReLU(name="relu_output")

    def call(self, features):
        return self._activation(self._residual_path(features) + self._main_path(features))

    def get_config(self):
        config = super().get_config()
        config.update({"num_inner_channels": self.num_inner_channels, "num_output_channels": self.num_output_channels, "input_stride": self.input_stride})
        return config

# ==============================================================================
# 2. INJECTION LOGIC
# ==============================================================================

def inject_weights():
    # 1. Create the clean model (TF 2.18 compatible)
    print("Building model...")
    layers_list = [
        PCEN(name="pcen", weights_trainable=True),
        tf.keras.layers.Reshape((128, 64, 1), name="expand_dims_cnn"), # Safe replacement for Lambda
        Conv2D(64, 7, padding="same", name="stem_conv"),
        tf.keras.layers.BatchNormalization(epsilon=1e-4, momentum=0.9997, scale=False, center=True, name="stem_bn"),
        tf.keras.layers.ReLU(name="stem_relu"),
        tf.keras.layers.MaxPool2D(3, 2, padding="same", name="stem_pool"),
        Block(64, 256, 1, name="g0_b0"),
        Block(64, 256, 1, name="g0_b1"), Block(64, 256, 1, name="g0_b2"),
        Block(128, 512, 2, name="g1_b0"),
        Block(128, 512, 1, name="g1_b1"), Block(128, 512, 1, name="g1_b2"), Block(128, 512, 1, name="g1_b3"),
        Block(256, 1024, 2, name="g2_b0"),
        *[Block(256, 1024, 1, name=f"g2_b{i}") for i in range(1, 6)],
        Block(512, 2048, 2, name="g3_b0"),
        Block(512, 2048, 1, name="g3_b1"), Block(512, 2048, 1, name="g3_b2"),
        tf.keras.layers.GlobalAveragePooling2D(name="global_pool"),
        tf.keras.layers.Dense(1, name="logits"),
        tf.keras.layers.Activation("sigmoid", name="predictions")
    ]
    model = tf.keras.Sequential(layers_list)
    model.build(input_shape=(None, 128, 64))
    
    # 4. Save
    model.save("migrated_model.keras")
    print("Saved migrated_model.keras")

if __name__ == "__main__":
    inject_weights()
    
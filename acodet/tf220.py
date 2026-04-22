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

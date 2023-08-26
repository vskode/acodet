# Copyright 2022 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PCEN implementation, forked from google-research/leaf-audio."""

import tensorflow as tf


class PCEN(tf.keras.layers.Layer):
    """Per-Channel Energy Normalization.

    This applies a fixed or learnable normalization by an exponential moving
    average smoother, and a compression.
    This implementation replicates the computation of fixed_pcen and
    trainable_pcen defined in
    google3/audio/hearing/tts/tensorflow/python/ops/pcen_ops.py.
    See https://arxiv.org/abs/1607.05666 for more details.
    """

    def __init__(
        self,
        alpha: float,
        smooth_coef: float,
        delta: float = 2.0,
        root: float = 2.0,
        floor: float = 1e-6,
        trainable: bool = False,
        name="PCEN",
    ):
        """PCEN constructor.

        Args:
          alpha: float, exponent of EMA smoother
          smooth_coef: float, smoothing coefficient of EMA
          delta: float, bias added before compression
          root: float, one over exponent applied for compression (r in the paper)
          floor: float, offset added to EMA smoother
          trainable: bool, False means fixed_pcen, True is trainable_pcen
          name: str, name of the layer
        """
        super().__init__(name=name)
        self._alpha_init = alpha
        self._delta_init = delta
        self._root_init = root
        self._smooth_coef = smooth_coef
        self._floor = floor
        self._trainable = trainable

    def build(self, input_shape):
        num_channels = input_shape[-1]
        self.alpha = self.add_weight(
            name="alpha",
            shape=[num_channels],
            initializer=tf.keras.initializers.Constant(self._alpha_init),
            trainable=self._trainable,
        )
        self.delta = self.add_weight(
            name="delta",
            shape=[num_channels],
            initializer=tf.keras.initializers.Constant(self._delta_init),
            trainable=self._trainable,
        )
        self.root = self.add_weight(
            name="root",
            shape=[num_channels],
            initializer=tf.keras.initializers.Constant(self._root_init),
            trainable=self._trainable,
        )
        self.ema = tf.keras.layers.SimpleRNN(
            units=num_channels,
            activation=None,
            use_bias=False,
            kernel_initializer=tf.keras.initializers.Identity(
                gain=self._smooth_coef
            ),
            recurrent_initializer=tf.keras.initializers.Identity(
                gain=1.0 - self._smooth_coef
            ),
            return_sequences=True,
            trainable=False,
        )

    def call(self, inputs):
        alpha = tf.math.minimum(self.alpha, 1.0)
        root = tf.math.maximum(self.root, 1.0)
        ema_smoother = self.ema(
            inputs, initial_state=tf.gather(inputs, 0, axis=1)
        )
        one_over_root = 1.0 / root
        output = (
            inputs / (self._floor + ema_smoother) ** alpha + self.delta
        ) ** one_over_root - self.delta**one_over_root
        return output

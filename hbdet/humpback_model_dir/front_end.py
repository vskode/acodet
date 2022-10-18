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
"""Functions for representing raw audio as images.

Having these in the graph allows hyperparameters involved in this representation
to be fixed at training time instead of needing to materialize extracted
features on disk.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import tensorflow as tf

Config = collections.namedtuple("Config", [
    "stft_frame_length",
    "stft_frame_step",
    "freq_bins",
    "sample_rate",
    "lower_f",
    "upper_f",
])
"""Configuration for the front end.

Attributes:
  stft_frame_length: The window length for the STFT in samples.
  stft_frame_step: The number of samples from the start of one STFT snapshot to
    the next.
  freq_bins: The number of mel bins in the spectrogram.
  lower_f: Lower boundary of mel bins in Hz.
  upper_f: Upper boundary of mel bins in Hz.
"""
# Config.__new__.__defaults__ = (1024, 300, 64, 10000.0, 0.0, 5000.0)
Config.__new__.__defaults__ = (1024, 53, 64, 2000.0, 0.0, 1000.0)


class MelSpectrogram(tf.keras.layers.Layer):
  """Keras layer that converts a waveform to an amplitude mel spectrogram."""

  def __init__(self, config=None, name="mel_spectrogram"):
    super(MelSpectrogram, self).__init__(name=name)
    if config is None:
      config = Config()
    self.config = config

  def build(self, input_shape):
    self._stft = tf.keras.layers.Lambda(
        lambda t: tf.signal.stft(
            tf.squeeze(t, 2),
            frame_length=self.config.stft_frame_length,
            frame_step=self.config.stft_frame_step),
        name="stft",
    )
    num_spectrogram_bins = self._stft.compute_output_shape(input_shape)[-1]
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
                    name="matrix"), 1)),
        name="mel_bins",
    )

  def call(self, inputs):
    return self._bin(self._stft(inputs))

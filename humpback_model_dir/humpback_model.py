"""Keras implementation of the NOAA PIFSC humpback whale detection model.

The released SavedModel is available at

https://tfhub.dev/google/humpback_whale/1

Model.load_from_tf_hub is a convenience method that downloads the SavedModel and
initializes the Model defined here with the weights from the SavedModel.

To save redownloading, assuming a local copy is in model_dir, use:

model = humpback_model.Model()
model.load_weights(model_dir)

This is the Python code that was used to export that model. We release it to
enable users who want to do finer-grained fine-tuning than what is possible
with the SavedModel alone, for example freezing or removing layers.

The code implements a standard ResNet50 in Keras. Although off-the-shelf
implementations exist, this reimplementation was necessary to have exact control
over variable names, to load weights from a model that had been trained in TF1.
"""

import shutil
import tarfile
import tempfile
import time
from urllib import request

import tensorflow as tf

# import front_end
# import leaf_pcen
from humpback_model_dir import front_end
from humpback_model_dir import leaf_pcen

TF_HUB_URL = 'https://tfhub.dev/google/humpback_whale/1?tf-hub-format=compressed'
NUM_CLASSES = 1
NUM_AUDIO_CHANNELS = 1
CONTEXT_WAVEFORM_SHAPE = [None, 39124, NUM_AUDIO_CHANNELS]
CONTEXT_SPECTROGRAM_SHAPE = [None, 128, 64]
EMBEDDING_DIMENSION = 2048


def BatchNormalization(name=None):
  """Defaults some of the arguments of Keras' BatchNormalization layer."""
  return tf.keras.layers.BatchNormalization(
      epsilon=1e-4,
      momentum=0.9997,
      scale=False,
      center=True,
      name=name,
  )


def Conv2D(filters, kernel_size, strides=(1, 1), padding='VALID', name=None):
  """Defaults some of the arguments of Keras' Conv2D layer."""
  return tf.keras.layers.Conv2D(
      filters,
      kernel_size,
      strides,
      padding=padding,
      activation=None,
      use_bias=False,
      name=name,
  )


def _call_layers(layers, inputs):
  """Applies the function composition of layers to inputs, like Sequential."""
  t = inputs
  for layer in layers:
    t = layer(t)
  return t


class ResidualPath(tf.keras.layers.Layer):
  """Layer for the residual "skip" connection in a ResNet block."""

  def __init__(self, num_output_channels, input_stride):
    super(ResidualPath, self).__init__(name='residual_path')
    self.num_output_channels = num_output_channels
    self.input_stride = input_stride

  def build(self, input_shape):
    num_input_channels = input_shape[-1]
    if num_input_channels != self.num_output_channels:
      self._layers = [
          Conv2D(
              self.num_output_channels,
              kernel_size=1,
              strides=self.input_stride,
              padding='VALID',
              name='conv_residual'),
          BatchNormalization(name='batch_normalization_residual'),
      ]
    else:
      self._layers = []

  def call(self, inputs):
    return _call_layers(self._layers, inputs)


class MainPath(tf.keras.layers.Layer):
  """Layer for the bottleneck-and-convolutional "core" of a ResNet block."""

  def __init__(self, num_inner_channels, num_output_channels, input_stride):
    super(MainPath, self).__init__(name='main_path')
    self.num_inner_channels = num_inner_channels
    self.num_output_channels = num_output_channels
    self.input_stride = input_stride

  def build(self, input_shape):
    num_input_channels = input_shape[-1]
    if num_input_channels != self.num_output_channels:
      bottleneck_padding = 'VALID'
    else:
      bottleneck_padding = 'SAME'
    self._layers = [
        Conv2D(
            self.num_inner_channels,
            kernel_size=1,
            strides=self.input_stride,
            padding=bottleneck_padding,
            name='conv_bottleneck'),
        BatchNormalization(name='batch_normalization_bottleneck'),
        tf.keras.layers.ReLU(name='relu_bottleneck'),
        Conv2D(
            self.num_inner_channels,
            kernel_size=3,
            strides=1,
            padding='SAME',
            name='conv'),
        BatchNormalization(name='batch_normalization'),
        tf.keras.layers.ReLU(name='relu'),
        Conv2D(
            self.num_output_channels,
            kernel_size=1,
            strides=1,
            padding='SAME',
            name='conv_output'),
        BatchNormalization(name='batch_normalization_output'),
    ]

  def call(self, inputs):
    return _call_layers(self._layers, inputs)


class Block(tf.keras.layers.Layer):
  """Layer for a ResNet block."""

  def __init__(self,
               num_inner_channels,
               num_output_channels,
               input_stride=1,
               name='block'):
    super(Block, self).__init__(name=name)
    self.num_inner_channels = num_inner_channels
    self.num_output_channels = num_output_channels
    self.input_stride = input_stride

  def build(self, input_shape):
    self._residual_path = ResidualPath(self.num_output_channels,
                                       self.input_stride)
    self._main_path = MainPath(self.num_inner_channels,
                               self.num_output_channels, self.input_stride)
    self._activation = tf.keras.layers.ReLU(name='relu_output')

  def call(self, features):
    return self._activation(
        self._residual_path(features) + self._main_path(features))


class Group(tf.keras.layers.Layer):
  """Layer for a group of ResNet blocks with common inner and outer depths."""

  def __init__(self, repeats, inner_channels, output_channels, input_stride,
               name):
    super(Group, self).__init__(name=name)
    assert repeats >= 1
    self.repeats = repeats
    self.inner_channels = inner_channels
    self.output_channels = output_channels
    self.input_stride = input_stride

  def build(self, input_shape):
    self._layers = [
        Block(
            self.inner_channels,
            self.output_channels,
            input_stride=self.input_stride,
            name='block_widen')
    ]
    for i in range(1, self.repeats):
      self._layers.append(
          Block(
              self.inner_channels,
              self.output_channels,
              input_stride=1,
              name=('block_' + str(i))))

  def call(self, inputs):
    return _call_layers(self._layers, inputs)


class PreBlocks(tf.keras.layers.Layer):
  """Layer with first-layer convolutional filters."""

  def build(self, input_shape):
    self._layers = [
        tf.keras.layers.Lambda(
            lambda t: tf.expand_dims(t, 3), name='make_depth_one'),
        Conv2D(64, 7, padding='SAME', name='conv'),
        BatchNormalization(name='batch_normalization'),
        tf.keras.layers.ReLU(name='relu'),
        tf.keras.layers.MaxPool2D(3, 2, padding='SAME', name='pool'),
    ]

  def call(self, inputs):
    return _call_layers(self._layers, inputs)


class Embed(tf.keras.layers.Layer):
  """Composition of layers transforming sectrogram to embedding vector.

  The spectrogram has [128, 64] [time, frequency] bins. When the weights from TF
  Hub are used, frequency whould be on a mel scale and PCEN applied, as in the
  implementation of Model.__init__, later.
  """

  def build(self, input_shape):
    self._layers = [
        tf.keras.layers.InputLayer(input_shape=[128, 64]),
        PreBlocks(),
        Group(3, 64, 256, input_stride=1, name='group'),
        Group(4, 128, 512, input_stride=2, name='group_1'),
        Group(6, 256, 1024, input_stride=2, name='group_2'),
        Group(3, 512, 2048, input_stride=2, name='group_3'),
        tf.keras.layers.GlobalAveragePooling2D(name='pool'),
    ]

  def call(self, inputs):
    return _call_layers(self._layers, inputs)


class Model(tf.keras.Sequential):
  """Full humpback detection Keras model with supplementary signatures.

  See "Advanced Usage" on https://tfhub.dev/google/humpback_whale/1 for details
  on the reusable SavedModels attributes (front_end, features, logits).

  The "score" method is provided for variable-length inputs.
  """

  @classmethod
  def load_from_tf_hub(cls):
    """Downloads the SavedModel and initialized this class using its weights."""
    with tempfile.NamedTemporaryFile() as temp_file:
      with request.urlopen(TF_HUB_URL) as response:
        shutil.copyfileobj(response, temp_file)
      temp_file.flush()
      temp_file.seek(0)
      with tempfile.TemporaryDirectory() as temp_dir:
        with tarfile.open(temp_file.name, 'r:gz') as tar:
          tar.extractall(path=temp_dir)
        model = cls()
        model.load_weights(temp_dir)
    return model

  def __init__(self):
    super(Model, self).__init__(layers=[
        front_end.MelSpectrogram(),
        leaf_pcen.PCEN(
            alpha=0.98,
            delta=2.0,
            root=2.0,
            smooth_coef=0.025,
            floor=1e-6,
            trainable=True,
            name='pcen',
        ),
        Embed(),
        tf.keras.layers.Dense(NUM_CLASSES),
    ])
    front_end_layers = self.layers[:2]
    self._spectrogram, self._pcen = front_end_layers

    # Parts exposed through Reusable SavedModels interface.
    self.front_end = tf.keras.Sequential(
        [tf.keras.layers.InputLayer([None, 1])] + front_end_layers)
    self.features = tf.keras.Sequential(
        [tf.keras.layers.InputLayer([128, 64]), self.layers[2]])
    self.logits = tf.keras.Sequential([tf.keras.layers.InputLayer([128, 64])] +
                                      self.layers[2:])

  @tf.function(input_signature=[
      tf.TensorSpec(shape=(None, None, 1), dtype=tf.float32),
      tf.TensorSpec(shape=tuple(), dtype=tf.int64)
  ])
  def score(self, waveform, context_step_samples):
    """Scores each context window in an arbitrary-length waveform.

    This is the clip-level version of __call__. It slices out short waveform
    context windows of the duration expected by __call__, scores them as a
    batch, and returns the corresponding scores.

    Args:
      waveform: [batch, samples, channels == 1] Tensor of PCM audio.
      context_step_samples: Difference between the starts of two consecutive
        context windows, in samples.

    Returns:
      Dict {'scores': [batch, num_windows, 1]} Tensor of per-context-window
      model outputs. (Post-sigmoid, in [0, 1].)
    """
    batch_size = tf.shape(waveform)[0]
    stft_frame_step_samples = 300
    context_step_frames = tf.cast(
        context_step_samples // stft_frame_step_samples, tf.dtypes.int32)
    mel_spectrogram = self._spectrogram(waveform)
    context_duration_frames = self.features.input_shape[1]
    context_windows = tf.signal.frame(
        mel_spectrogram, context_duration_frames, context_step_frames, axis=1)
    num_windows = tf.shape(context_windows)[1]
    windows_in_batch = tf.reshape(context_windows,
                                  (-1,) + self.features.input_shape[1:])
    per_window_pcen = self._pcen(windows_in_batch)
    scores = tf.nn.sigmoid(self.logits(per_window_pcen))
    return {'scores': tf.reshape(scores, [batch_size, num_windows, 1])}

  @tf.function(input_signature=[])
  def metadata(self):
    config = self._spectrogram.config
    return {
        'input_sample_rate':
            tf.cast(config.sample_rate, tf.int64),
        'context_width_samples':
            tf.cast(
                config.stft_frame_step * (self.features.input_shape[1] - 1) +
                config.stft_frame_length, tf.int64),
        'class_names':
            tf.constant(['Mn']),
    }

if __name__ == '__main__':
    model = Model()
    model.load_weights('models/google_humpback_model')
    model.build((1, 39124, 1))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss=tf.keras.losses.BinaryCrossentropy()
    )
    print('hi')
    
import tensorflow as tf
from acodet import models


class SM(tf.keras.Sequential):
    """Full humpback detection Keras model with supplementary signatures.

    See "Advanced Usage" on https://tfhub.dev/google/humpback_whale/1 for details
    on the reusable SavedModels attributes (front_end, features, logits).

    The "score" method is provided for variable-length inputs.
    """

    def __init__(self, model):
        super(SM, self).__init__(layers=model.layers[1:])
        front_end_layers = self.layers[:2]
        self._spectrogram, self._pcen = front_end_layers

        # Parts exposed through Reusable SavedModels interface.
        self.front_end = tf.keras.Sequential(
            [tf.keras.layers.InputLayer([None, 1])] + front_end_layers
        )
        self.features = tf.keras.Sequential(
            [tf.keras.layers.InputLayer([128, 64]), self.layers[2]]
        )
        self.logits = tf.keras.Sequential(
            [tf.keras.layers.InputLayer([128, 64])] + self.layers[2:]
        )

    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=(None, None, 1), dtype=tf.float32),
            tf.TensorSpec(shape=tuple(), dtype=tf.int64),
        ]
    )
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
            context_step_samples // stft_frame_step_samples, tf.dtypes.int32
        )
        mel_spectrogram = self._spectrogram(waveform)
        context_duration_frames = self.features.input_shape[1]
        context_windows = tf.signal.frame(
            mel_spectrogram,
            context_duration_frames,
            context_step_frames,
            axis=1,
        )
        num_windows = tf.shape(context_windows)[1]
        windows_in_batch = tf.reshape(
            context_windows, (-1,) + self.features.input_shape[1:]
        )
        per_window_pcen = self._pcen(windows_in_batch)
        scores = tf.nn.sigmoid(self.logits(per_window_pcen))
        return {"scores": tf.reshape(scores, [batch_size, num_windows, 1])}

    @tf.function(input_signature=[])
    def metadata(self):
        config = self._spectrogram.config
        return {
            "input_sample_rate": tf.cast(config.sample_rate, tf.int64),
            "context_width_samples": tf.cast(
                config.stft_frame_step * (self.features.input_shape[1] - 1)
                + config.stft_frame_length,
                tf.int64,
            ),
            "class_names": tf.constant(["Mn"]),
        }


def save(model):
    model.save(
        "saved_model_1",
        signatures={
            "score": model.score,
            "metadata": model.metadata,
            "serving_default": model.score,
        },
    )


model = models.init_model(
    model_instance="GoogleMod",
    checkpoint_dir=f"../trainings/2022-11-30_01/unfreeze_no-TF",
    keras_mod_name=False,
)
path = "tests/test_files/test_audio_files/BERCHOK_SAMANA_200901_4/Samana_PU194_20090111_003000.wav"
e, r = tf.audio.decode_wav(tf.io.read_file(path))
smod = SM(model)
smod.predict(tf.expand_dims(e, 0))
save(smod)

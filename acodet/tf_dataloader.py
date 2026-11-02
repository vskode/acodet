import tensorflow as tf
import numpy as np
import librosa as lb
import acodet.global_config as conf
from acodet.combine_annotations import read_annotations_from_file, get_train_noise_df
from acodet.humpback_model_dir import front_end


class TFAudioDataset:
    def __init__(self, df, mode='train'):
        self.df = df.reset_index(drop=True)
        self.mode = mode

    def load_single_row(self, filename, start, end, label):
        # 1. Extract raw values from the TF Tensors
        # filename comes in as a byte-string tensor
        path_str = filename.numpy().decode('utf-8')
        start_val = float(start.numpy())
        end_val = float(end.numpy())
        label_val = int(label.numpy())

        # 2. Now Librosa will be happy with plain Python types
        wave, sr = lb.load(
            path=path_str,
            sr=conf.SR,
            offset=start_val,
            duration=end_val - start_val
        )

        # 3. Ensure consistent length (conf.CONTEXT_WIN)
        wave = lb.util.fix_length(wave, size=conf.CONTEXT_WIN)

        return wave.astype(np.float32), np.int32(label_val)

    def get_dataset(self):
        # 1. Create a dataset of metadata
        ds = tf.data.Dataset.from_tensor_slices((
            self.df['filename'].values.astype(str),
            self.df['start'].values,
            self.df['end'].values,
            self.df['label'].values.astype(np.int32)
        ))

        # 2. Use py_function to run Librosa in parallel
        def py_map_fn(f, s, e, l):
            audio, label = tf.py_function(
                func=self.load_single_row,
                inp=[f, s, e, l],
                Tout=[tf.float32, tf.int32]
            )
            # Crucial: Set the shapes so the rest of the pipeline doesn't crash
            audio.set_shape((conf.CONTEXT_WIN,))
            label.set_shape(())
            return audio, label

        # 3. Parallelize! num_parallel_calls is what makes the L4 fast
        ds = ds.map(py_map_fn, num_parallel_calls=tf.data.AUTOTUNE)
        return ds


class TFLoader:
    """TensorFlow equivalent of your PyTorch Loader"""
    def __init__(self, df_path):
        self.df_path = df_path

        train_df, val_df, eval_df = read_annotations_from_file(df_path)
        # noise used for the noise (mixup) augmentation, only from the
        # training set so that no validation or test audio leaks into training
        noise_df = get_train_noise_df(train_df)

        # Create datasets
        self.train = TFAudioDataset(train_df, mode='train')
        self.val = TFAudioDataset(val_df, mode='val')
        self.test = TFAudioDataset(eval_df, mode='test')
        self.noise = TFAudioDataset(noise_df, mode='train')

        # get sizes
        self.n_train = len(train_df)
        self.n_val = len(val_df)
        self.n_eval = len(eval_df)
        self.n_noise = len(noise_df)

    def train_loader(self, batch_size=None):
        dataset = self.train.get_dataset()
        # This is the key: parallelize the generator execution
        dataset = dataset.map(lambda x, y: (x, y), num_parallel_calls=tf.data.AUTOTUNE)
        return dataset

    def val_loader(self, batch_size=None):
        return self.val.get_dataset()

    def test_loader(self, batch_size=None):
        return self.test.get_dataset()

    def noise_loader(self, batch_size=None):
        return self.noise.get_dataset()


def spec():
    return tf.keras.Sequential(
        [
            tf.keras.layers.Input([conf.CONTEXT_WIN]),
            tf.keras.layers.Lambda(lambda t: tf.expand_dims(t, -1)),
            front_end.MelSpectrogram(),
        ]
    )


_spec_model = None


def get_spec():
    global _spec_model
    if _spec_model is None:
        _spec_model = spec()
    return _spec_model


def prepare(ds, batch_size, shuffle=False, shuffle_buffer=750, seed=None, AUTOTUNE=None):
    if shuffle:
        ds = ds.shuffle(shuffle_buffer, seed=seed)

    # This is the "magic" line for generators:
    # It tells TF to run the generator in multiple background threads
    ds = ds.map(lambda x, y: (x, y), num_parallel_calls=AUTOTUNE)

    ds = ds.batch(batch_size)
    return ds.prefetch(buffer_size=AUTOTUNE)


def make_spec_tensor(ds, AUTOTUNE=None, batch_size=32):
    spec_model = get_spec()
    ds = ds.batch(batch_size)

    def apply_spec(x, *y):
        # Force the shape to (Batch Size, context window length)
        # -1 allows the last batch to be smaller than batch_size
        x = tf.reshape(x, [-1, conf.CONTEXT_WIN])
        return (spec_model(x, training=False), *y)

    ds = ds.map(apply_spec, num_parallel_calls=AUTOTUNE)
    return ds.unbatch()

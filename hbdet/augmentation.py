import tensorflow as tf
from keras_cv.layers import BaseImageAugmentationLayer
import numpy as np
import yaml
from hbdet.humpback_model_dir import front_end

with open('hbdet/hbdet/config.yml', 'r') as f:
    config = yaml.safe_load(f)
AUTOTUNE = tf.data.AUTOTUNE    

class CropAndFill(BaseImageAugmentationLayer):
    def __init__(self, height: int, width: int, seed: int=None) -> None:
        """
        Augmentation class inheriting from keras' Augmentation Base class.
        This class takes images, cuts them at a random x position and then 
        appends the first section to the second section. It is intended for 
        spectrograms of labelled bioacoustics data. This way the vocalization
        in the spectrogram is time shifted and potentially cut. All of which 
        is possible to occur due to a windowing of a recording file that is 
        intended for inference. 
        It is essentially a time shift augmentation whilst preserving window 
        length and not requiring reloading data from the source file. 

        Args:
            height (int): height of image
            width (int): width of image
            seed (int, optional): create randomization seed. Defaults to None.
        """
        super().__init__()
        self.height = height
        self.width = width
        self.seed = seed
    
    def call(self, audio: tf.Tensor):
        """
        Compute time shift augmentation by creating a random slicing
        position and then returning the reordered image. 

        Args:
            audio (tf.Tensor): input image

        Returns:
            tf.Tensor: reordered image
        """
        np.random.seed(self.seed)
        beg = np.random.randint(self.width//2) + self.width//2
        
        # for debugging purposes
        if not isinstance(audio, tf.Tensor):
            audio = audio[0][0]
            
        return tf.concat([audio[beg:], audio[:beg]], 0)
    
class MixCallAndNoise(BaseImageAugmentationLayer):
    def __init__(self, noise_data: tf.data.Dataset, seed: int=None, 
                 alpha: float=0.3, **kwargs) -> None:
        super().__init__()
        self.seed = seed
        self.alpha = alpha
        self.noise_ds = noise_data
        self.len = 370 - 1
        
    def call(self, train_sample: tf.Tensor):
        np.random.seed(self.seed)
        r = np.random.randint(self.len)
        self.noise_audio, _ = next(iter(self.noise_ds.take(1)))
        return train_sample*(1-self.alpha) + self.noise_audio*self.alpha
    
    
##############################################################################
##############################################################################
##############################################################################

def time_shift():
    return tf.keras.Sequential([CropAndFill(64, 128)])

def spec():
    return tf.keras.Sequential([
        tf.keras.layers.Input([config['context_win']]),
        tf.keras.layers.Lambda(lambda t: tf.expand_dims(t, -1)),
        front_end.MelSpectrogram()])


def mix_up(noise_data):
    return tf.keras.Sequential([MixCallAndNoise(noise_data=noise_data)])


def augment(ds, augments=1, aug_func=time_shift):
    ds_augs = []
    for i in range(augments):
        ds_augs.append(ds.map(lambda x, y: (aug_func(x, training=True), y), 
                num_parallel_calls=AUTOTUNE))        
    return ds_augs

def prepare(ds, batch_size, shuffle=False, shuffle_buffer=750, augmented_data=None):
    if not augmented_data is None:
        for ds_aug in augmented_data:
            ds = ds.concatenate(ds_aug)
    if shuffle:
        ds = ds.shuffle(shuffle_buffer)    
    ds = ds.batch(batch_size)
    # create specs from audio arrays
    return ds.prefetch(buffer_size=AUTOTUNE)

def make_spec_tensor(ds):
    ds = ds.batch(1)
    ds = ds.map(lambda x, y: (spec()(x), y), num_parallel_calls=AUTOTUNE)
    return ds.unbatch()
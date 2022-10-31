import tensorflow as tf
from keras_cv.layers import BaseImageAugmentationLayer
import numpy as np
import yaml

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
    def __init__(self, noise_data: tf.data.Dataset, 
                 train_set_size: int, 
                 seed: int=None, 
                 alpha: float=0.3, 
                 batch_size: int=32, 
                 **kwargs) -> None:
        super().__init__()
        self.seed = seed
        self.alpha = alpha
        self.noise_ds = noise_data
        self.len = train_set_size - 1 
        
        np.random.seed(self.seed)
        
        self.noise_audio = []
        for _ in range(batch_size//2):
            r = np.random.randint(self.len)
            self.noise_audio.append(next(iter(self.noise_ds
                                        .skip(r)
                                        .take(1)))[0])
        
    def call(self, train_sample: tf.Tensor):
        np.random.seed(self.seed)
        r = np.random.randint(len(self.noise_audio))
        noise_mixup = self.noise_audio[r]
        return train_sample*(1-self.alpha) + noise_mixup*self.alpha
    
    
##############################################################################
##############################################################################
##############################################################################

def time_shift():
    return tf.keras.Sequential([CropAndFill(64, 128)])

def mix_up(train_set_size, noise_data):
    return tf.keras.Sequential([MixCallAndNoise(train_set_size=train_set_size,
                                                noise_data=noise_data)])

def augment(ds, augments=1, aug_func=time_shift):
    ds_augs = []
    for _ in range(augments):
        ds_augs.append(ds.map(lambda x, y: (aug_func(x, training=True), y), 
                num_parallel_calls=AUTOTUNE))        
    return ds_augs

def run_augment_pipeline(train_data, noise_data, train_set_size, 
                         n_time_augs, n_mixup_augs,
                         seed = None):
    time_aug_data = list(zip(augment(train_data, augments = n_time_augs, 
                            aug_func=time_shift()),
                            ['time_shift']*n_time_augs ))

    mixup_aug_data = list(zip(augment(train_data, augments = n_mixup_augs, 
                            aug_func=mix_up(train_set_size, noise_data)),
                            ['mix_up']*n_mixup_augs ))

    np.random.seed(seed)
    r = np.random.randint(len(time_aug_data))
    mixup_aug_data += list(zip(augment(time_aug_data[r][0], 
                                       augments = n_mixup_augs, 
                            aug_func=mix_up(train_set_size, noise_data)),
                            ['mix_up']*n_mixup_augs ))

    return [*time_aug_data, *mixup_aug_data, (noise_data, 'noise')]


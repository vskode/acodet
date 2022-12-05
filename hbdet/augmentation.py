import tensorflow as tf
from keras_cv.layers import BaseImageAugmentationLayer
from hbdet.plot_utils import plot_sample_spectrograms
import tensorflow_io as tfio

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
        tf.random.set_seed(self.seed)
    
    def call(self, audio: tf.Tensor):
        """
        Compute time shift augmentation by creating a random slicing
        position and then returning the reordered image. 

        Args:
            audio (tf.Tensor): input image

        Returns:
            tf.Tensor: reordered image
        """
        beg = tf.random.uniform(shape = [], maxval=self.width//2, 
                                dtype=tf.int32)
        
        # for debugging purposes
        # tf.print('time shift augmentation computed)
            
        return tf.roll(audio, shift=[beg], axis=[0])
    

def time_shift():
    return tf.keras.Sequential([CropAndFill(64, 128)])

def m_test(ds1, ds2, alpha=0.4):
    call, lab = ds1
    noise, l = ds2
    noise_alpha = alpha * tf.math.reduce_max(noise)
    train_alpha = (1-alpha) * tf.math.reduce_max(call) 
    # tf.print('performing mixup')
    return (call*train_alpha + noise*noise_alpha, lab)

def time_mask(x, y, spec_param=10):
    # tf.print('performing time_mask')
    return (tfio.audio.time_mask(x, param=spec_param), y)

def freq_mask(x, y, spec_param=10):
    # tf.print('performing freq_mask')
    return (tfio.audio.freq_mask(x, param=spec_param), y)

##############################################################################
##############################################################################
##############################################################################

def run_augment_pipeline(ds, noise_data, noise_set_size,
                         train_set_size, time_augs, mixup_augs,
                         seed=None, plot=False, time_start=None, 
                         spec_aug=False, spec_param=10, **kwargs):
    T = time_shift()
    if plot:
        plot_sample_spectrograms(ds, dir = time_start, name = 'train', 
                            seed=seed, ds_size=train_set_size, **kwargs)
        
    if mixup_augs:
        ds_n = (noise_data.repeat(train_set_size//noise_set_size + 1))
        if plot:
            plot_sample_spectrograms(ds_n, dir = time_start,
                                name=f'noise', seed=seed, 
                                ds_size=train_set_size, **kwargs)
            
        if plot:
            dss = tf.data.Dataset.zip((ds, ds_n))
            ds_mu = dss.map(lambda x, y: m_test(x, y),
                            num_parallel_calls=AUTOTUNE)
            plot_sample_spectrograms(ds_mu, dir = time_start,
                                name=f'augment_0-MixUp', seed=seed, 
                                ds_size=train_set_size, **kwargs)
            
        ds_n = ds_n.shuffle(train_set_size//noise_set_size + 1)
        dss = tf.data.Dataset.zip((ds, ds_n))
        ds_mu = dss.map(lambda x, y: m_test(x, y),
                        num_parallel_calls=AUTOTUNE)
        ds_mu_n = ds_mu.concatenate(noise_data)
        # ds = ds.concatenate(ds_mu_n)
        
    if time_augs:
        ds_t = ds.map(lambda x, y: (T(x, training=True), y), 
                    num_parallel_calls=AUTOTUNE) 
        if plot:
            plot_sample_spectrograms(ds_t, dir = time_start,
                                name=f'augment_0-TimeShift', seed=seed, 
                                ds_size=train_set_size, **kwargs)
        # ds = ds.concatenate(ds_t)
        
    if spec_aug:
        ds_tm = ds.map(time_mask)
        # ds = ds.concatenate(ds_tm)
        if plot:
            plot_sample_spectrograms(ds_tm, dir = time_start,
                                name=f'augment_0-TimeMask', seed=seed, 
                                ds_size=train_set_size, **kwargs)
        ds_fm = ds.map(freq_mask)
        # ds = ds.concatenate(ds_fm)
        if plot:
            plot_sample_spectrograms(ds_fm, dir = time_start,
                                name=f'augment_0-TFreqMask', seed=seed, 
                                ds_size=train_set_size, **kwargs)
    if mixup_augs:
        ds = ds.concatenate(ds_mu_n)
    if time_augs:
        ds = ds.concatenate(ds_t)
    if spec_aug:
        ds = ds.concatenate(ds_tm)
        ds = ds.concatenate(ds_fm)
        
    return ds


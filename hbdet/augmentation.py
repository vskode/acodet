import tensorflow as tf
from keras_cv.layers import BaseImageAugmentationLayer
import numpy as np

class CropAndFill(BaseImageAugmentationLayer):
    def __init__(self, height: int, width: int, seed: int=None) -> None:
        super().__init__()
        self.height = height
        self.width = width
        if seed is None:
            self.seed = 123
        else:
            self.seed = seed
    
    def call(self, audio: tf.Tensor):
        np.random.seed(self.seed)
        beg = np.random.randint(self.width)
        if not isinstance(audio, tf.Tensor):
            audio = audio[0][0]
        # aud = tf.cast(audio*2**15, 'int32')
        # aud = np.reshape(audio.numpy(), [39124])
        # new = np.append(aud[beg:], aud[:beg])
        # new_t = tf.expand_dims(new, -1)
        # return tf.cast(tf.concat(aud[beg:], aud[:beg])/2**15, 'float32')
        return tf.concat([audio[beg:], audio[:beg]], 0)
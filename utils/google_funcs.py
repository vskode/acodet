import tensorflow as tf
import tensorflow_hub as hub
import numpy as np

from funcs import *

def load_google_hub():
    return hub.load('https://tfhub.dev/google/humpback_whale/1')

def load_google_sequential():
    sequential = tf.keras.Sequential([
    tf.keras.layers.Input([39124]),
    tf.keras.layers.Lambda(lambda t: tf.expand_dims(t, -1)),
    hub.KerasLayer(load_google_hub(), trainable=False),
    tf.keras.layers.Activation('sigmoid')
    ])

    sequential.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss=tf.keras.losses.BinaryCrossentropy(),
    )
    return sequential

def predict_hub(data, model, cntxt_wn_sz, **_):
    preds = []
    for segment in data:
        tensor_sig = tf.convert_to_tensor(segment)
        tensor_sig = tf.expand_dims(tensor_sig, -1)  # makes a batch of size 1
        tensor_sig = tf.expand_dims(tensor_sig, 0)  # makes a batch of size 1
        context_step_samples = tf.cast(cntxt_wn_sz, tf.int64)
        score_fn = model.signatures['score']
        scores = score_fn(waveform=tensor_sig, 
                            context_step_samples=context_step_samples)
        preds.append(scores['scores'].numpy()[0,0,0])
    return np.array(preds)

class GooglePreds():
    def __init__(self, params):
        self.model = load_google_sequential()
        self.params = params
        
    def load_data(self, file, annots, y_test, y_noise, x_test, x_noise):
        self.x_test = x_test
        self.x_noise = x_noise
        self.y_test = y_test
        self.y_noise = y_noise
        
    def pred(self, noise = False):
        if noise:
            return self.model.predict(self.x_noise).T[0]
        else:
            return self.model.predict(self.x_test).T[0]
    
    def eval(self, noise = False):
        if noise:
            return self.model.evaluate(self.x_noise, self.y_noise)
        else:
            return self.model.evaluate(self.x_test, self.y_test)
    
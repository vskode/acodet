#%% imports
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import glob
import shutil
import sys
import json

from ketos.data_handling import selection_table as sl
import ketos.data_handling.database_interface as dbi
from ketos.data_handling.parsing import load_audio_representation
from ketos.audio.spectrogram import MagSpectrogram
from ketos.data_handling.parsing import load_audio_representation
from ketos.audio.audio_loader import AudioFrameLoader
from ketos.neural_networks.dev_utils.detection import process, save_detections

from pathlib import Path
import os
import librosa as lb
import soundfile as sf
import matplotlib.pyplot as plt
from librosa.display import specshow
from ketos.neural_networks.dev_utils.nn_interface import NNInterface
#%%
if not 'models' in os.listdir():
    os.chdir('../..')

#%% inherit from NNInterface to test black-box hub models
class MLPInterface(NNInterface):
    def __init__(self, model):
        self.model = model

#%%

import tensorflow as tf
import tensorflow_hub as hub

hub_model = hub.load('https://tfhub.dev/google/humpback_whale/1')

fine_tuning_model = tf.keras.Sequential([
  tf.keras.layers.Input([94, 129]),
  tf.keras.layers.Lambda(lambda t: tf.expand_dims(t, -1)),
  hub.KerasLayer(hub_model, trainable=True),
])

fine_tuning_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss=tf.keras.losses.BinaryCrossentropy(),
)


model = MLPInterface(fine_tuning_model)
# %%
spctrgrm_sttngs = {'type': 'MagSpectrogram',
                    'rate': 1000,
                    'window': 0.256,
                    'step': 0.032,
                    'freq_min': 0,
                    'freq_max': 1000,
                    'duration': 3.,
                    'window_func': 'hamming'}

audio_loader = AudioFrameLoader(path='Daten/model_data/test/', 
                                step=3., repres=spctrgrm_sttngs)
# %%
first_spec = next(audio_loader) #load the first 3.0-s frame
second_spec = next(audio_loader) #load the second 3.0-s frame
# %%


first_spec.plot()
second_spec.plot()
plt.show()


# %% funktioniert nicht weil der input einfach nur arrays sind und nicht spektrogramme
spec = next(audio_loader) #load a spectrogram
data = spec.get_data()    #extract the pixel values as a 2d array
output = model.run_on_instance(data) #pass the pixel values to the classifier
print(output) #print the classifier's output
# %%

#%% imports
import tensorflow as tf
import tensorflow_hub as hub
import pandas as pd
import numpy as np
import glob
import shutil
import sys
import json

from pathlib import Path
import os
import librosa as lb
import soundfile as sf
import matplotlib.pyplot as plt
from librosa.display import specshow
# %% Load model
if not 'models' in os.listdir():
    os.chdir('../..')
    
hub_model = hub.load('https://tfhub.dev/google/humpback_whale/1')


fine_tuning_model = tf.keras.Sequential([
  tf.keras.layers.Input([39124]),
  tf.keras.layers.Lambda(lambda t: tf.expand_dims(t, -1)),
  hub.KerasLayer(hub_model, trainable=False),
  tf.keras.layers.Activation('sigmoid')
])

fine_tuning_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss=tf.keras.losses.BinaryCrossentropy(),
    # loss='mse',
)


# %%
sr = 10000
cntxt_wn_sz = 39124
nr_noise_samples = 10
annots = pd.read_csv('Daten/ket_annot.csv')
files = np.unique(annots.filename)

def return_array_of_39124_segments(annots, file):
  annotations = annots[annots.filename == file]
  audio, fs = lb.load(file, sr = 10000, 
                  offset = annotations['start'].iloc[0],
                  duration = annotations['end'].iloc[-1] +\
                    annotations['start'].iloc[0] + \
                    nr_noise_samples * cntxt_wn_sz)
  seg_ar = list()
  noise_ar = list()
  for index, row in annotations.iterrows():
    beg = int(row.start*sr)
    end = int(row.start*sr + cntxt_wn_sz)
    seg_ar.append(audio[beg:end])
    if all(row == annotations.iloc[-1]):
        for i in range(nr_noise_samples):
            beg = int((row.end)*sr + i*cntxt_wn_sz)
            end = int((row.end)*sr + (i+1)*cntxt_wn_sz)
            noise_ar.append(audio[beg:end])
  noise_ar = np.array(noise_ar)
  seg_ar = np.array(seg_ar)
  return seg_ar, noise_ar

# %%
seg_ar, noise_ar = return_array_of_39124_segments(annots, files[0])
# noise_ar = return_noise_segments(annots, files[0])#.astype('float32')
x_test = seg_ar.astype("float32")
y_test = np.ones(x_test.shape[0]).astype("float32")
model = fine_tuning_model

# %%
# print("Evaluate on test data")
# results = model.evaluate(x_test, y_test, batch_size=128)
# print("test loss, test acc:", results)


# %%
print("Generate predictions for 3 samples")
predictions = model.predict(x_test)
print("predictions:\n", predictions)

preds = []
for seg in seg_ar:
  tensor_sig = tf.convert_to_tensor(seg)
  tensor_sig = tf.expand_dims(tensor_sig, -1)  # makes a batch of size 1
  tensor_sig = tf.expand_dims(tensor_sig, 0)  # makes a batch of size 1
  context_step_samples = tf.cast(39124, tf.int64)
  score_fn = hub_model.signatures['score']
  scores = score_fn(waveform=tensor_sig, 
                    context_step_samples=context_step_samples)
  preds.append(scores['scores'].numpy()[0, 0, 0])
preds = np.array(preds)

#%%

# Generate predictions (probabilities -- the output of the last layer)
# on new data using `predict`

# PATH = Path('Daten/OneDrive_1_1-24-2022')
# file_path = list(PATH.iterdir())[0]

  
# file_path_10kHz = Path(file_path.__str__()[:-4] + '_10kHz.wav')
# # check_10kHz_file_exists(file_path_10kHz)

# section = 0
# offset = section * 120 + 3
# sig_segment, fs = lb.load(file_path_10kHz, sr=None, 
#                           offset=offset, duration = 3.9124+1.9)

# print("Generate predictions for 3 samples")
# predictions = model.predict(x_test)
# print("predictions:\n", predictions)

# %%

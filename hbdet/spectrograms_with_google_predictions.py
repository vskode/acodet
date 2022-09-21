#%% imports
import numpy as np
from pathlib import Path
import os
import librosa as lb
import soundfile as sf
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow_hub as hub

from librosa.display import specshow
# %matplotlib qt


if 'Daten' not in os.listdir():
  os.chdir('..')
  
#%% import file
def check_10kHz_file_exists(path):
  if path not in path.parent.iterdir():
    file, fs = lb.load(path.__str__()[:-10] + '.wav')
    file = lb.resample(file, orig_sr = fs, target_sr = 10000)
    sf.write(path, file, 10000)

def compute_and_plt_spec(sig_segment, window_length, f_min, sr_plot, **_):
  D = np.abs(lb.stft(sig_segment, win_length = window_length))**2
  S = lb.feature.melspectrogram(S = D, sr = fs)

  fig, ax = plt.subplots(figsize = [16, 8])

  S_dB = lb.power_to_db(S, ref=np.max)

  img = specshow(S_dB, x_axis = 'time', y_axis = 'mel', sr = fs,
                win_length = window_length, fmin = f_min, fmax=sr_plot / 2, ax=ax)
  fig.colorbar(img, ax=ax, format='%+2.0f dB')

  ax.set(title='Mel-frequency spectrogram')
  return fig, ax

def predict_values(sig_segment, model, sr_plot, cntxt_wn_sz, **_):
  tensor_sig = tf.convert_to_tensor(sig_segment)
  tensor_sig = tf.expand_dims(tensor_sig, -1)  # makes a batch of size 1
  tensor_sig = tf.expand_dims(tensor_sig, 0)  # makes a batch of size 1
  context_step_samples = tf.cast(sr_plot, tf.int64)
  score_fn = model.signatures['score']
  scores = score_fn(waveform=tensor_sig, context_step_samples=context_step_samples)
  return scores

def add_predictions_to_spectrogram(fig, ax, sig_segment, scores, 
                                   section, file_path, sr_plot, f_min, **_):
  predictions = scores['scores'].numpy()[0, :, 0]
  predictions_on_frequency_scale = predictions*(sr_plot/2- f_min) + f_min
  time_vector = np.arange(len(predictions))*len(sig_segment) \
                  // sr_plot / len(predictions)
  ax.plot(time_vector, 
          predictions_on_frequency_scale, 
          color ='white', linewidth = .5)
  fig.savefig(f'{file_path.__str__()[:-4]}_plots/' \
              f'specPreds_part-{section}.png', 
              facecolor = 'white', dpi = 300)
  plt.close()


# %%

params = {
  "sr_plot" : 10000,
  "window_length" : 2**11,
  "f_min" : 100,
  "cntxt_wn_sz": 39124,
  }


model = hub.load('https://tfhub.dev/google/humpback_whale/1')

PATH = Path('Daten/OneDrive_1_1-24-2022')
for file_path in PATH.iterdir():
  if not file_path.suffix == '.wav':
    continue
  if not Path(file_path.__str__()[:-4] + '_plots') in file_path.parent.iterdir():
    Path.mkdir(Path(file_path.__str__()[:-4] + '_plots'))
    
  file_path_10kHz = Path(file_path.__str__()[:-4] + '_10kHz.wav')
  check_10kHz_file_exists(file_path_10kHz)

  for section in range(15):
    offset = section * 120 + 3
    sig_segment, fs = lb.load(file_path_10kHz, sr=None, 
                              offset=offset, duration = 3.9124+1.9)
    if len(sig_segment) == 0:
      break
    fig, ax = compute_and_plt_spec(sig_segment, **params)
    scores = predict_values(sig_segment, model, **params)
    add_predictions_to_spectrogram(fig, ax, sig_segment, scores, 
                                   section, file_path, **params)
# %%

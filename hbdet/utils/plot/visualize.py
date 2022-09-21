#%% imports
import numpy as np
import pandas as pd
from pathlib import Path
import os
import librosa as lb
import soundfile as sf
import pyfar as pf
import matplotlib.pyplot as plt
# plt.switch_backend('inline')
# %matplotlib qt
# %matplotlib.pyplot.switch_backend('Qt5Agg')
# import os
# os.chdir('..')

# %% Load Data
# DATA_PATH = Path('Daten/OneDrive_1_1-24-2022')
# file_paths = list(DATA_PATH.iterdir())
# small_file = Path('../Daten/forVincent/335564853.200313230624.wav')

def check_10kHz_file_exists(path):
  if Path(path) not in Path(path).parent.iterdir():
    file, fs = lb.load(path[:-10] + '.wav')
    file = lb.resample(file, orig_sr = fs, target_sr = 10000)
    sf.write(path, file, 10000)

# FILENAME = '../Daten/OneDrive_1_1-24-2022/channelA_2021-03-18_01-00-05.wav'
if 'Daten' not in os.listdir():
  os.chdir('..')

FILENAME = 'Daten/forVincent/335564853.200313230624_10kHz.wav'

check_10kHz_file_exists(FILENAME)
if False:

    test_file = lb.load(file_paths[0])
    fs = test_file[1]

    test_file_s = test_file[0][:1000000]

    sf.write('../Daten/temp/test_1m.wav', test_file_s, fs)

file = sf.read(FILENAME)

# plt.figure()
# plt.plot(np.arange(0, len(file[0]))/file[1], file[0])


# print('hi')

## Spectrograms


# %% Visualisierung von Spektrogram

signal  = file[0]
time    = np.arange(0, len(file[0]))/file[1]
fs      = 10000#file[1]
freq_max= 10000
freq_min= 100

# start = np.where(time>start_time)[0][0]
# end   = np.where(time>end_time)[0][0]
# for scale in ['linear', 'log']:
#     fig, ax = plt.subplots(figsize=[8, 7])
#     sig = pf.Signal(signal, fs)
#     # sig = lb.stft(signal, n_fft = 1024)

#     bin_len = 1024#int(fs*time[-1]/40)

#     freqs, times, spec = pf.dsp.spectrogram(sig, window_length = bin_len)
#     # freqs = freqs/1000

#     # ax.imshow(spec[freqs<freq_max][::-1], extent = [times[0], times[-1], freq_min, freq_max], 
#                 # interpolation='hanning')
#     # ax = pf.plot.spectrogram(sig, yscale=scale, window_length = bin_len, ax = ax)[0][0]
    
#     ax.imshow(spec, extent = [times[0], times[-1], freq_max, freq_min], 
#           interpolation='hanning')
#     # ax.vlines(time[start], freq_max, freq_min, linestyle='dashed', 
#     #                         color='tab:red', label='begin of FFT sample')
#     # ax.vlines(time[end], freq_max, freq_min, linestyle='dotted', 
#     #                         color='tab:red', label='end of FFT sample')

#     ax.set_ylabel('f in kHz')
#     ax.set_xlabel('t in s')
#     # ax.set_aspect(times[-1]/(freq_max-freq_min))
#     fig.show()

print('hi')
#%%
from librosa.display import specshow
sr_plot = 10000
w = 2**11
f_min = 100
# for sr_plot in np.arange(4, 5)*2000:
  # print('---')
  # for w in 2**np.arange(5, 12):
orig_signal, fs = sf.read(FILENAME)
orig_signal = np.array(orig_signal, dtype=np.float32)
resampled_sig = lb.resample(orig_signal, orig_sr = fs, target_sr = sr_plot)
sig_segment = resampled_sig[len(resampled_sig)//10:len(resampled_sig)//10*2]
D = np.abs(lb.stft(sig_segment,
                  win_length = w))**2
S = lb.feature.melspectrogram(S=D, sr = sr_plot)

fig, ax = plt.subplots(figsize = [16, 8])

S_dB = lb.power_to_db(S, ref=np.max)
# print(S_dB.shape, 'sr: ', sr_plot, 'win_len: ', w)
img = specshow(S_dB, x_axis='time', y_axis='mel', sr=sr_plot,
               win_length = w,
              fmin = f_min, fmax=sr_plot / 2, ax=ax)
fig.colorbar(img, ax=ax, format='%+2.0f dB')

ax.set(title='Mel-frequency spectrogram')
# fig.show()


# import plotly.express as px

# img = pf.dsp.spectrogram(sig, window_length = bin_len)
# fig = px.imshow(img[2], aspect = 'auto')
# # px.set_aspect(times[-1]/(freq_max-freq_min))
# fig.show()
# %%
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt

# FILENAME = 'gs://bioacoustics-www1/sounds/Cross_02_060203_071428.d20_7.wav'

# FILENAME = 'drive/MyDrive/Daten/temp/test_1m_10khz.wav'
# FILENAME = 'drive/MyDrive/Daten/temp/channelA_2021-03-18_01-00-05.wav'

model = hub.load('https://tfhub.dev/google/humpback_whale/1')

# waveform, sample_rate = tf.audio.decode_wav(tf.io.read_file(FILENAME))
tensor_sig = tf.convert_to_tensor(sig_segment)
tensor_sig = tf.expand_dims(tensor_sig, -1)  # makes a batch of size 1
tensor_sig = tf.expand_dims(tensor_sig, 0)  # makes a batch of size 1
context_step_samples = tf.cast(sr_plot, tf.int64)
score_fn = model.signatures['score']
scores = score_fn(waveform=tensor_sig, context_step_samples=context_step_samples)
print(scores)
# %%
out = scores['scores'].numpy()[0, :, 0]
plt.plot(np.arange(len(out))*len(sig_segment)//sr_plot/len(out), out*(sr_plot/2- f_min) + f_min, color ='white', linewidth = .5)
plt.savefig('test1.png', dpi = 300)

# %%

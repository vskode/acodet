import pandas as pd
import numpy as np

from pathlib import Path
import librosa as lb
import matplotlib.pyplot as plt
from librosa.display import specshow


def get_annots_for_file(annots, file):
    return annots[annots.filename == file].sort_values('start')

def return_cntxt_wndw_arr(annotations, file, nr_noise_samples, cntxt_wn_sz,
                            sr, **_):
    offset = annotations['start'].iloc[0]
    audio, fs = lb.load(file, sr = sr, 
                    offset = offset,
                    duration = annotations['start'].iloc[-1] +\
                                cntxt_wn_sz/sr+offset)
    
    seg_ar = list()
    for index, row in annotations.iterrows():
        beg = int((row.start - offset)*sr)
        end = int((row.start - offset)*sr + cntxt_wn_sz)
        
        if len(audio[beg:end]) == cntxt_wn_sz:
            seg_ar.append(audio[beg:end])
        else:
            end = len(audio)
            beg = end - cntxt_wn_sz
            seg_ar.append(audio[beg:end])
            break
    seg_ar = np.array(seg_ar, dtype='float32')
            
    noise_ar = return_noise_arrays(file, sr, 
                                   annotations,
                                   nr_noise_samples,
                                   cntxt_wn_sz)
    return seg_ar, noise_ar

def return_noise_arrays(file, sr, annotations,
                        nr_noise_samples, cntxt_wn_sz):
    try:
        noise, fs = lb.load(file, sr = sr, 
                offset = annotations['end'].iloc[-1],
                duration = nr_noise_samples * cntxt_wn_sz)
        noise_ar = list()
        for i in range(nr_noise_samples):
            beg = i*cntxt_wn_sz
            end = (i+1)*cntxt_wn_sz
            if len(noise[beg:end]) == cntxt_wn_sz:
                noise_ar.append(noise[beg:end])
            else:
                break
    except:
        noise_ar = list()
        
    return np.array(noise_ar, dtype='float32')

def get_file_durations():
    return pd.read_csv('Daten/file_durations.csv')

def plot_and_save_spectrogram(signal, file_path, label, 
                              fft_window_length, sr, cntxt_wn_sz, 
                              start, noise=False, **_):
    S = np.abs(lb.stft(signal, win_length = fft_window_length))
    fig, ax = plt.subplots(figsize = [6, 4])
    # limit S first dimension from [10:256], thatway isolating frequencies
    # (sr/2)/1025*10 = 48.78 to (sr/2)/1025*256 = 1248.78 for visualization
    f_min = sr/2/S.shape[0]*10
    f_max = sr/2/S.shape[0]*256
    S_dB = lb.amplitude_to_db(S[10:256, :], ref=np.max)
    img = specshow(S_dB, x_axis = 's', y_axis = 'linear', 
                   sr = sr, win_length = fft_window_length, ax=ax, 
                x_coords = np.linspace(start, start+cntxt_wn_sz/sr, S.shape[1]),
                y_coords = np.linspace(f_min, f_max, 246),
                vmin = -40)
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    file_name = f'{file_path}_spec_w_label.png'
    ax.set(title=f'spec. of random sample | prediction: {label:.4f}\n'\
                f'file: {file_path}.wav')
    
    if noise:
        dir_path = 'predictions/google/spectrograms/noise/'
        file_name = dir_path + file_name[:-4] + '_noise.png'
    else:
        dir_path = 'predictions/google/spectrograms/calls/'
        file_name = dir_path + file_name[:-4] + '_call.png'
        
    fig.savefig(file_name, 
            facecolor = 'white', dpi = 300)

def return_labels(annotations, file):
    return annotations.label.values

def calc_mse(predictions, labels):
    return sum((labels - predictions)**2) / len(labels)

def calc_rmse(predictions, labels):
    return np.sqrt(sum((labels - predictions)**2) / len(labels))

def calc_mae(predictions, labels):
    return sum(abs(labels - predictions)) / len(labels)

def get_metrics(predictions, y_test):
    if len(y_test) == 0:
        return 0, 0, 0
    mse = calc_mse(predictions, y_test)
    rmse = calc_rmse(predictions, y_test)
    mae = calc_mae(predictions, y_test)
    return mse, rmse, mae

def get_quality_of_recording(file):
    try:    
        path = 'Daten/Catherine_annotations/Detector_scanning_metadata.xlsx'
        stanton_bank = pd.read_excel(path, sheet_name = 'SBank')
        file_date = pd.to_datetime(Path(file).stem, 
                                format='PAM_%Y%m%d_%H%M%S_000')
        
        condition_date = pd.Timestamp(file_date.date()) == stanton_bank.Date
        hours = [elem.hour for elem in stanton_bank.hour]
        condition_hour = pd.DataFrame(hours) == file_date.hour
        quality = stanton_bank['quality'][
            (condition_date.values & condition_hour.values.T)[0]
            ].values[0]
        return quality
    except Exception as e:
        print(e)
        return 'unknown'
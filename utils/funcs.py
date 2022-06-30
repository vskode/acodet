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

def get_time(time):
    return f'{int(time/60)}:{np.mod(time, 60):.1f}s'

def plot_ref_spec(signal, file_path, label, 
                              fft_window_length, sr, cntxt_wn_sz,
                              start, noise=False, **_):
    S = np.abs(lb.stft(signal, win_length = fft_window_length))
    fig, ax = plt.subplots(figsize = [6, 4])
    # limit S first dimension from [10:256], thatway isolating frequencies
    # (sr/2)/1025*10 = 48.78 to (sr/2)/1025*266 = 1297.56 for visualization
    fmin = sr/2/S.shape[0]*10
    fmax = sr/2/S.shape[0]*266
    S_dB = lb.amplitude_to_db(S[10:266, :], ref=np.max)
    img = specshow(S_dB, x_axis = 's', y_axis = 'linear', 
                   sr = sr, win_length = fft_window_length, ax=ax, 
                   x_coords = np.linspace(0, cntxt_wn_sz/sr, S_dB.shape[1]),
                    y_coords = np.linspace(fmin, fmax, 2**8),
                vmin = -40)
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    file_name = f'{Path(file_path).stem}_spec_w_label.png'
    ax.set(title=f'spec. of random sample\n'\
                f'file: {Path(file_path).stem}.wav | start = {get_time(start)}')
    if noise:
        dir_path = f'predictions/reference/spectrograms/noise/'
        create_dirs(Path(dir_path))
        file_name = dir_path + file_name[:-4] + '_noise.png'
    else:
        dir_path = f'predictions/reference/spectrograms/calls/'
        create_dirs(Path(dir_path))
        file_name = dir_path + file_name[:-4] + '_call.png'
        
    fig.savefig(file_name, 
            facecolor = 'white', dpi = 300)
    plt.close(fig)


def plot_spec(spec_data, file_path, prediction, start,
                              fft_window_length, sr, cntxt_wn_sz, fmin, fmax,
                              mod_name, noise=False, **_):
    fig, ax = plt.subplots(figsize = [6, 4])
    img = specshow(spec_data, x_axis = 's', y_axis = 'linear', 
                   sr = sr, win_length = fft_window_length, ax=ax, 
                   x_coords = np.linspace(0, cntxt_wn_sz/sr, spec_data.shape[1]),
                    y_coords = np.linspace(fmin, fmax, spec_data.shape[0]))
    fig.colorbar(img, ax=ax, format='%+2.1f dB')
    file_name = f'{Path(file_path).stem}_spec_w_label.png'
    ax.set(title=f'spec. of random sample | prediction: {prediction:.4f}\n'\
            f'file: {Path(file_path).stem}.wav | start = {get_time(start)}')
    
    if noise:
        dir_path = f'predictions/{mod_name}/spectrograms/noise/'
        create_dirs(Path(dir_path))
        file_name = dir_path + file_name[:-4] + '_noise.png'
    else:
        dir_path = f'predictions/{mod_name}/spectrograms/calls/'
        create_dirs(Path(dir_path))
        file_name = dir_path + file_name[:-4] + '_call.png'
        
    fig.savefig(file_name, 
            facecolor = 'white', dpi = 300)
    plt.close(fig)
    
def generate_spectrograms(x_test, x_noise, y_test, y_noise, model, file,
                        file_annots, mod_iter, **params):
    num_c = np.random.randint(len(x_test))
    num_n = np.random.randint(len(x_noise)) if len(x_noise)>0 else 0
    
    # num = np.argmax(abs(y_test - preds['call']))
    model.spec(num_c)
    if mod_iter == 0:
        plot_ref_spec(x_test[num_c], file, y_test[num_c], 
                    start = file_annots.start.iloc[num_c], **params)
        
    if len(y_noise) > 0:
        # num = np.argmax(abs(y_noise - preds['noise']))
        model.spec(num_n, noise = True)
        if mod_iter == 0:
            plot_ref_spec(x_noise[num_n], file, 
                        y_noise[num_n], 
                        start = file_annots.start.iloc[-1] +\
                        num_n*params['cntxt_wn_sz']/params['sr'],
                        noise = True, **params)
    
def create_dirs(path):
    path.mkdir(parents = True, exist_ok=True)

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

def collect_all_metrics(mtrxs, preds, y_test, y_noise):
    mtrxs['mse'], mtrxs['rmse'], mtrxs['mae'] = get_metrics(preds['call'], 
                                                            y_test)
    mtrxs['mse_n'], mtrxs['rmse_n'], mtrxs['mae_n'] = get_metrics(preds['noise'], 
                                                                y_noise)
    mtrxs['mse_t'], mtrxs['rmse_t'], mtrxs['mae_t'] = get_metrics(preds['thresh'], 
                                                                y_test)
    mtrxs['mse_t_n'], mtrxs['rmse_t_n'], mtrxs['mae_t_n'] = get_metrics(preds['thresh_noise'],
                                                                        y_noise)
    return mtrxs

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
    
def get_dicts():
    mtrxs = {'mse':0, 'rmse':0, 'mae':0}
    mtrxs.update({f'{m}_t': 0 for m in mtrxs})
    mtrxs.update({f'{m}_n': 0 for m in mtrxs})
    mtrxs.update({'bin_cross_entr': 0, 'bin_cross_entr_n': 0})

    preds = {'call':[], 'thresh': [], 'noise': [], 'thresh_noise': []}
    return preds, mtrxs
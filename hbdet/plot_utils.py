import numpy as np
import matplotlib.pyplot as plt 
import time
from pathlib import Path
import json
import librosa as lb
import yaml

with open('hbdet/hbdet/config.yml', 'r') as f:
    config = yaml.safe_load(f)

def plot_model_results(datetime, **kwargs):
    r, c = 2, 4
    fig, ax = plt.subplots(ncols = c, nrows = r, figsize = [15, 8])

    checkpoint_paths = Path(f"trainings/{datetime}").glob('unfreeze_*')
    for checkpoint_path in checkpoint_paths:
        unfreeze = checkpoint_path.stem.split('_')[-1]

        if not Path(f"{checkpoint_path}/results.json").exists():
            continue
        with open(f"{checkpoint_path}/results.json", 'r') as f:
            results = json.load(f)

        for i, key in enumerate(results.keys()):
            ax[i//c, i%c].plot(results[key], 
                            label = f'{unfreeze}')
            
            # axis handling depending on subplot index
            if i//c == 1 and i%c == 0:
                ax[i//c, i%c].set_ylim([0, 2])
            if i//c == 0:
                ax[i//c, i%c].set_title(f'{key}')
            if i//c == i%c == 0:
                ax[i//c, i%c].set_ylabel('training')
                ax[i//c, i%c].legend()
            elif i//c == 1 and i%c == 0:
                ax[i//c, i%c].set_ylabel('val')

    info_string = ''
    for key, val in kwargs.items():
        info_string += f' | {key}: {val}'
    
    today = time.ctime()
    fig.suptitle(f'Model Results{info_string}'
                '\n'
                f'{today}')
    ref_time = time.strftime('%Y%m%d', time.gmtime())
    fig.savefig(f'trainings/{datetime}/model_results_{ref_time}.png')


def plot_spec_from_file(file, start, sr, cntxt_wn_sz = 39124, **kwArgs):
    audio, sr = lb.load(file, sr = sr, offset = start/sr, 
                        duration = cntxt_wn_sz/sr)
    return simple_spec(audio, sr = sr, cntxt_wn_sz=cntxt_wn_sz, **kwArgs)

def save_rndm_spectrogram(dataset,  path, sr = config['preproc']['sr']):
    ds_size = sum(1 for _ in dataset)
    
    r, c = 4, 4 
    sample = dataset.skip(np.random.randint(ds_size)).take(1)
    sample = next(iter(sample))[0][:r*c]
    
    max_freq_bin = 128//(config['preproc']['sr']//2000)
    
    fmin = sr/2/sample[0].numpy().shape[0]
    fmax = sr/2/sample[0].numpy().shape[0]*max_freq_bin
    fig, axes = plt.subplots(nrows = r, ncols = c, figsize=[12, 10])
    
    for i, samp in enumerate(sample):
        ar = samp.numpy()[:,1:max_freq_bin].T
        axes[i//r][i%c].imshow(ar, origin='lower', interpolation='nearest',
                                aspect='auto')
        if i//r == r-1 and i%c == 0:
            axes[i//r][i%c].set_xticks(np.linspace(0, ar.shape[1], 5))
            xlabs = np.linspace(0, 3.9, 5).astype(str)
            axes[i//r][i%c].set_xticklabels(xlabs)
            axes[i//r][i%c].set_xlabel('time in s')
            axes[i//r][i%c].set_yticks(np.linspace(0, ar.shape[0]-1, 7))
            ylabs = np.linspace(fmin, fmax, 7).astype(int).astype(str)
            axes[i//r][i%c].set_yticklabels(ylabs)
            axes[i//r][i%c].set_ylabel('freq in Hz')
        else:
            axes[i//r][i%c].set_xticks([])
            axes[i//r][i%c].set_xticklabels([])         
            axes[i//r][i%c].set_yticks([])
            axes[i//r][i%c].set_yticklabels([])
            
    fig.suptitle('Random sample of 16 spectrograms')
    fig.savefig(path)

def simple_spec(signal, ax = None, fft_window_length=2**11, sr = 10000, 
                cntxt_wn_sz = 39124, fig = None, colorbar = True):
    S = np.abs(lb.stft(signal, win_length = fft_window_length))
    if not ax:
        fig_new, ax = plt.subplots()
    if fig:
        fig_new = fig
    # limit S first dimension from [10:256], thatway isolating frequencies
    # (sr/2)/1025*10 = 48.78 to (sr/2)/1025*266 = 1297.56 for visualization
    fmin = sr/2/S.shape[0]*10
    fmax = sr/2/S.shape[0]*266
    S_dB = lb.amplitude_to_db(S[10:266, :], ref=np.max)
    img = lb.display.specshow(S_dB, x_axis = 's', y_axis = 'linear', 
                   sr = sr, win_length = fft_window_length, ax=ax, 
                   x_coords = np.linspace(0, cntxt_wn_sz/sr, S_dB.shape[1]),
                    y_coords = np.linspace(fmin, fmax, 2**8),
                vmin = -60)
    
    if colorbar:
        fig_new.colorbar(img, ax=ax, format='%+2.0f dB')
        return fig_new, ax
    else:
        return ax
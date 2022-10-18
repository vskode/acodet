import numpy as np
import matplotlib.pyplot as plt 
import time
from pathlib import Path
import json
import librosa as lb

def plot_model_results(datetime, **kwargs):

    fig, ax = plt.subplots(ncols = 4, nrows = 2, figsize = [15, 8])

    checkpoint_paths = Path(f"trainings/{datetime}").glob('unfreeze_*')
    for checkpoint_path in checkpoint_paths:
        unfreeze = checkpoint_path.stem.split('_')[-1]

        if not Path(f"{checkpoint_path}/results.json").exists():
            continue
        with open(f"{checkpoint_path}/results.json", 'r') as f:
            results = json.load(f)

        for i, m in enumerate(results.keys()):
            row = i // 4
            col = np.mod(i, 4)
            if row == 1 and col == 0:
                ax[row, col].set_ylim([0, 2])
            ax[row, col].plot(results[m], 
                            label = f'{unfreeze}')
            if row == 0:
                ax[row, col].set_title(f'{m}')
            if row == col == 0:
                ax[row, col].set_ylabel('training')
            elif row == 1 and col == 0:
                ax[row, col].set_ylabel('val')
    ax[0, 0].legend()

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

def save_rndm_spectrogram(dataset,  path, sr = 10000):
    ds_size = sum(1 for _ in dataset)
    
    sample = dataset.skip(np.random.randint(ds_size)).take(1)
    sample = next(iter(sample))[0][:16]
    
    fmin = sr/2/sample[0].numpy().shape[0]
    fmax = sr/2/sample[0].numpy().shape[0]*(128//5)
    fig, axes = plt.subplots(nrows = 4, ncols = 4, figsize=[12, 10])
    
    for i, samp in enumerate(sample):
        ar = samp.numpy()[:,1:128//5].T
        axes[i//4][i%4].imshow(ar, origin='lower', interpolation='nearest',
                                aspect='auto')
        if i//4 == 3 and i%4 == 0:
            axes[i//4][i%4].set_xticks(np.linspace(0, ar.shape[1], 5))
            xlabs = np.linspace(0, 3.9, 5).astype(str)
            axes[i//4][i%4].set_xticklabels(xlabs)
            axes[i//4][i%4].set_xlabel('time in s')
            axes[i//4][i%4].set_yticks(np.linspace(0, ar.shape[0]-1, 7))
            ylabs = np.linspace(fmin, fmax, 7).astype(int).astype(str)
            axes[i//4][i%4].set_yticklabels(ylabs)
            axes[i//4][i%4].set_ylabel('freq in Hz')
        else:
            axes[i//4][i%4].set_xticks([])
            axes[i//4][i%4].set_xticklabels([])         
            axes[i//4][i%4].set_yticks([])
            axes[i//4][i%4].set_yticklabels([])
            
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

if __name__ == '__main__':
    plot_model_results('2022-10-04_15', dataset = 'good and poor data, 5 shifts from 0s - 2s',
                                        begin_lr = '0.005', end_lr = '1e-5')
import numpy as np
import matplotlib.pyplot as plt 
from matplotlib.gridspec import GridSpec
import time
from pathlib import Path
import json
import librosa as lb
import yaml
from . import funcs
from . import tfrec
import tensorflow as tf

with open('hbdet/hbdet/config.yml', 'r') as f:
    config = yaml.safe_load(f)

def plot_model_results(datetime, **kwargs):
    r, c = 2, 4
    fig, ax = None, None

    checkpoint_paths = Path(f"trainings/{datetime}").glob('unfreeze_*')
    for i, checkpoint_path in enumerate(checkpoint_paths):
        unfreeze = checkpoint_path.stem.split('_')[-1]

        if not Path(f"{checkpoint_path}/results.json").exists():
            continue
        with open(f"{checkpoint_path}/results.json", 'r') as f:
            results = json.load(f)
        
        if i == 0:
            c = len(list(results.keys()))//2
            fig, ax = plt.subplots(ncols = c, nrows = r, figsize = [15, 8])
        
        for i, key in enumerate(results.keys()):
            ax[i//c, i%c].plot(results[key], 
                            label = f'{unfreeze}')
            ax[i//c, i%c].set_ylim([.5, 1])
            
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
    fig.tight_layout()
    fig.savefig(f'trainings/{datetime}/model_results_{ref_time}.png')


def plot_spec_from_file(file, start, sr, cntxt_wn_sz = 39124, **kwArgs):
    audio, sr = lb.load(file, sr = sr, offset = start/sr, 
                        duration = cntxt_wn_sz/sr)
    return simple_spec(audio, sr = sr, cntxt_wn_sz=cntxt_wn_sz, **kwArgs)

def plot_sample_spectrograms(dataset, *, dir, name, ds_size=None,
                          random=True, seed=None, sr=config['sr'], 
                          rows=4, cols=4):
    r, c = rows, cols 
    if random:
        if ds_size is None: ds_size = sum(1 for _ in dataset)
        np.random.seed(seed)
        rand_skip = np.random.randint(ds_size)
        sample = dataset.skip(rand_skip).take(r*c)
    else:
        sample = dataset.take(r*c)
    
    max_freq_bin = 128//(config['sr']//2000)
    
    fmin = sr/2/next(iter(sample))[0].numpy().shape[0]
    fmax = sr/2/next(iter(sample))[0].numpy().shape[0]*max_freq_bin
    fig, axes = plt.subplots(nrows = r, ncols = c, figsize=[12, 10])
    
    for i, (aud, lab) in enumerate(sample):
        ar = aud.numpy()[:,1:max_freq_bin].T
        axes[i//r][i%c].imshow(ar, origin='lower', interpolation='nearest',
                                aspect='auto')
        axes[i//r][i%c].set_title(f'label: {lab}')
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
            
    fig.suptitle(f'{name} sample of 16 spectrograms. random={random}')
    fig.savefig(f'trainings/{dir}/{name}_sample.png')

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
    
def plot_conf_matr(labels, preds, ax, training_run):
    heat = tf.math.confusion_matrix(labels, preds).numpy()
    ax.imshow(heat, cmap='Greys')
    value_string = '{}\n{:.0f}%'
    for row in range(2):
        for col in range(2):
            ax.text(col, row, 
                     value_string.format(heat[row, col], 
                                         heat[row, col]/np.sum(heat[row])*100), 
                     ha='center', va='center', color='orange')
    ax.set_xticks([0, 1], labels=['TP', 'TN'])
    ax.set_yticks([0, 1], labels=['pred. P', 'pred. N'])
    ax.set_title(Path(training_run).stem)
    return ax

def plot_pr_curve(labels, preds, ax, training_path, **kwArgs):
    pr = dict()
    for met in ('Recall', 'Precision'):
        pr.update({met: funcs.get_pr_arrays(labels, preds, met)})
    
    if 'load_untrained_model' in kwArgs:
        ax.plot(pr['Recall'], pr['Precision'], label='untrained_model')
    else:
        ax.plot(pr['Recall'], pr['Precision'], label=f'{training_path.stem}')
        
    ax.set_ylabel('precision')
    ax.set_xlabel('recall')
    ax.set_ylim([0.3, 1])
    ax.set_xlim([0.3, 1])
    ax.legend()
    ax.grid(True)
    ax.set_title('Precision and Recall Curves')
    return ax
    
def plot_evaluation_metric(model_instance, training_runs, val_data, 
                           fig, plot_pr=True, plot_cm=False, 
                           plot_untrained=False, 
                           **kwargs):
    r = plot_cm+plot_pr
    c = len(training_runs)
    gs = GridSpec(r, c, figure=fig)
    if plot_pr:
        ax_pr = fig.add_subplot(gs[0, :])
        
    for i, run in enumerate(training_runs):
        labels, preds = funcs.get_labels_and_preds(model_instance, run, 
                                                   val_data, **kwargs)            
        if not plot_pr:
            plot_conf_matr(labels, preds, fig.add_subplot(gs[-1, i]), run)
        else:
            ax_pr = plot_pr_curve(labels, preds, 
                                  ax_pr, run, **kwargs)
            if plot_untrained:
                ax_pr = plot_pr_curve(labels, preds, ax_pr, run,
                                load_untrained_model=True, **kwargs)
            if plot_cm:
                plot_conf_matr(labels, preds, fig.add_subplot(gs[-1, i]), run)
            
        print('creating pr curve for ', run.stem)
    return fig
    
def create_and_save_figure(model_instance, tfrec_path, batch_size, train_date,
                            debug=False, plot_pr=True, plot_cm=False, 
                            **kwargs):
    
    training_runs = list(Path(f'trainings/{train_date}').glob('unfreeze*'))
    val_data = tfrec.get_val_data(tfrec_path, batch_size, debug=debug)
    
    fig = plt.figure(constrained_layout=True)

    plot_evaluation_metric(model_instance, training_runs, val_data, 
                           fig = fig, plot_pr=plot_pr, plot_cm=plot_cm, 
                           **kwargs)
    
    info_string = ''
    for key, val in kwargs.items():
        info_string += f' | {key}: {val}'
    fig.suptitle(f'Evaluation Metrics{info_string}')
    
    fig.savefig(f'{training_runs[0].parent}/eval_metrics.png')
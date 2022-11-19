import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
import time
from pathlib import Path
import seaborn as sns
import json
import librosa as lb
import yaml
from . import funcs
from . import tfrec
import tensorflow as tf

with open('hbdet/hbdet/config.yml', 'r') as f:
    config = yaml.safe_load(f)

def plot_model_results(datetimes, labels=None, fig=None, legend=True, **kwargs):
    if fig == None:
        fig = plt.figure(figsize=(15, 8))
        savefig = True
    else:
        savefig = False
        
    
    if not isinstance(datetimes, list):
        datetimes = [datetimes]
        
    checkpoint_paths = []
    for datetime in datetimes:
        checkpoint_paths += list(Path(f"trainings/{datetime}")
                                 .glob('unfreeze_*'))
        
    r, c = 2, 6
    for j, checkpoint_path in enumerate(checkpoint_paths):
        unfreeze = checkpoint_path.stem.split('_')[-1]

        if not Path(f"{checkpoint_path}/results.json").exists():
            if j == 0:
                ax = fig.subplots(ncols = c, nrows = r)
            continue
        with open(f"{checkpoint_path}/results.json", 'r') as f:
            results = json.load(f)
        
        if j == 0:
            c = len(list(results.keys()))//2
            ax = fig.subplots(ncols = c, nrows = r)
            
        if not labels is None:
            label = labels[j]
        else:
            label = f'{checkpoint_path.parent.stem}_{unfreeze}'
        
        for i, key in enumerate(results.keys()):
            if 'val_' in key: 
                row = 1
            else:
                row = 0
            if 'loss' in key:
                col = 0
            else:
                col += 1
                
            ax[row, col].plot(results[key], label = label)
            if not col == 0:
                ax[row, col].set_ylim([.5, 1.01])
            
            # axis handling depending on subplot index
            if row == 1 and col == 0:
                ax[row, col].set_ylim([0, 1.5])
            if row == 0:
                ax[row, col].set_title(f'{key}')
            if row == col == 0:
                ax[row, col].set_ylim([0, .5])
                ax[row, col].set_ylabel('training')
            elif row == 1 and col == 0:
                ax[row, col].set_ylabel('val')
            if legend and row == 1 and col == c-1:
                ax[row, col].legend(loc='center left', 
                                     bbox_to_anchor=(1, 0.5))

    info_string = ''
    for key, val in kwargs.items():
        info_string += f' | {key}: {val}'
    
    today = time.ctime()
    fig.suptitle(f'Model Results{info_string}'
                '\n'
                f'{today}')
    ref_time = time.strftime('%Y%m%d', time.gmtime())
    if savefig:
        fig.tight_layout()
        fig.savefig(f'trainings/{datetime}/model_results_{ref_time}.png')
    else:
        return fig


def plot_spec_from_file(file, start, sr, cntxt_wn_sz = 39124, **kwArgs):
    audio, sr = lb.load(file, sr = sr, offset = start/sr, 
                        duration = cntxt_wn_sz/sr)
    return simple_spec(audio, sr = sr, cntxt_wn_sz=cntxt_wn_sz, **kwArgs)

def plot_sample_spectrograms(dataset, *, dir, name, ds_size=None,
                          random=True, seed=None, sr=config['sr'], 
                          rows=4, cols=4, plot_meta=False, **kwargs):
    r, c = rows, cols 
    if isinstance(dataset, tf.data.Dataset):
        if random:
            if ds_size is None: ds_size = sum(1 for _ in dataset)
            np.random.seed(seed)
            rand_skip = np.random.randint(ds_size)
            sample = dataset.skip(rand_skip).take(r*c)
        else:
            sample = dataset.take(r*c)
    elif isinstance(dataset, list):
        sample = dataset
    
    max_freq_bin = 128//(config['sr']//2000)
    
    fmin = sr/2/next(iter(sample))[0].numpy().shape[0]
    fmax = sr/2/next(iter(sample))[0].numpy().shape[0]*max_freq_bin
    fig, axes = plt.subplots(nrows = r, ncols = c, figsize=[12, 10])
    
    for i, (aud, *lab) in enumerate(sample):
        if i == r*c:
            break
        ar = aud.numpy()[:,1:max_freq_bin].T
        axes[i//r][i%c].imshow(ar, origin='lower', interpolation='nearest',
                                aspect='auto')
        if len(lab) == 1:
            axes[i//r][i%c].set_title(f'label: {lab[0]}')
        elif len(lab) == 3:
            label, file, t = (v.numpy() for v in lab)
            axes[i//r][i%c].set_title(f'label: {label}; t in f: {funcs.get_time(t)}\n'
                                      f'file: {Path(file.decode()).stem}')
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
    
def plot_conf_matr(labels, preds, ax, training_run, iteration, **kwargs):
    bin_preds = list(map(lambda x: 1 if x >= config['thresh'] else 0, preds))
    heat = tf.math.confusion_matrix(labels, bin_preds).numpy()
    rearrange = lambda x:  np.array([[x[1, 1], x[1, 0]], 
                                [x[0, 1], x[0, 0]]])
    rearranged_head = rearrange(heat)
    value_string = '{}\n{:.0f}%'
    heat_annot = [[[], []], [[], []]]
    for row in range(2):
        for col in range(2):
            heat_annot[row][col] = value_string.format(heat[row, col], 
                                         heat[row, col]/np.sum(heat[row])*100)
    rearranged_annot = rearrange(np.array(heat_annot))
    
    ax = sns.heatmap(rearranged_head, annot=rearranged_annot, fmt='', 
                     cbar=False, ax=ax, xticklabels=False, yticklabels=False)
    ax.set_yticks([0.5, 1.5], labels=['TP', 'TN'])
    ax.set_xticks([1.5, 0.5], labels=['pred. N', 'pred.P'])
    color = list(mcolors.TABLEAU_COLORS.keys())[iteration]
    ax.set_title(Path(training_run).parent.stem 
                 + Path(training_run).stem.split('_')[-1], color=color)
    return ax

def plot_pr_curve(labels, preds, ax, training_path, iteration=0, **kwargs):
    pr = dict()
    for met in ('Recall', 'Precision'):
        pr.update({met: funcs.get_pr_arrays(labels, preds, met)})
    
    if 'plot_labels' in kwargs:
        if isinstance(kwargs['plot_labels'], list):
            label = kwargs['plot_labels'][iteration] 
        else:
            label = kwargs['plot_labels']
    elif 'load_untrained_model' in kwargs:
        label='untrained_model'
    else:
        label = str(training_path.parent.stem
                    + training_path.stem.split('_')[-1])
        
    ax.plot(pr['Recall'], pr['Precision'], label=label)
        
    ax.set_ylabel('precision')
    ax.set_xlabel('recall')
    ax.set_ylim([0.5, 1])
    ax.set_xlim([0.7, 1])
    ax.legend()
    ax.grid(True)
    ax.set_title('Precision and Recall Curves')
    return ax
    
def plot_evaluation_metric(model_instance, training_runs, val_data, 
                           fig, plot_pr=True, plot_cm=False, 
                           plot_untrained=False, legend=False,
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
            plot_conf_matr(labels, preds, fig.add_subplot(gs[-1, i]), 
                           run, iteration=i)
        else:
            ax_pr = plot_pr_curve(labels, preds, 
                                  ax_pr, run, iteration=i, **kwargs)
            if plot_untrained:
                ax_pr = plot_pr_curve(labels, preds, ax_pr, run,
                                load_untrained_model=True, iteration=i, 
                                **kwargs)
            if plot_cm:
                plot_conf_matr(labels, preds, fig.add_subplot(gs[-1, i]), 
                               run, iteration=i)
            
        print('creating pr curve for ', run.stem)
    if legend:
        ax_pr.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    return fig
    
def create_and_save_figure(model_instance, tfrec_path, batch_size, train_date,
                            debug=False, plot_pr=True, plot_cm=False, 
                            **kwargs):
    
    training_runs = list(Path(f'trainings/{train_date}').glob('unfreeze*'))
    val_data = tfrec.run_data_pipeline(tfrec_path, 'val', return_spec=False)
    
    fig = plt.figure(constrained_layout=True)

    plot_evaluation_metric(model_instance, training_runs, val_data, 
                           fig = fig, plot_pr=plot_pr, plot_cm=plot_cm, 
                           **kwargs)
    
    info_string = ''
    for key, val in kwargs.items():
        info_string += f' | {key}: {val}'
    fig.suptitle(f'Evaluation Metrics{info_string}')
    
    fig.savefig(f'{training_runs[0].parent}/eval_metrics.png')
    
def plot_pre_training_spectrograms(train_data, test_data, 
                                   augmented_data, 
                                   time_start, seed):
    
    plot_sample_spectrograms(train_data, dir = time_start, name = 'train_all', 
                            seed=seed)
    for i, (augmentation, aug_name) in enumerate(augmented_data):
        plot_sample_spectrograms(augmentation, dir = time_start, 
                                name=f'augment_{i}-{aug_name}', seed=seed)
    plot_sample_spectrograms(test_data, dir = time_start, name = 'test')
    
    
def compare_random_spectrogram(filenames, dataset_size = config['tfrecs_lim']):
    r = np.random.randint(dataset_size)
    dataset = (
        tf.data.TFRecordDataset(filenames)
        .map(tfrec.parse_tfrecord_fn)
        .skip(r)
        .take(1)
    )

    sample = next(iter(dataset))
    aud, file, lab, time = (sample[k].numpy() for k in list(sample.keys()))
    file = file.decode()

    fig, ax = plt.subplots(ncols = 2, figsize = [12, 8])
    ax[0] = funcs.simple_spec(aud, fft_window_length = 512, sr = 10000, 
                                ax = ax[0], colorbar = False)
    _, ax[1] = funcs.plot_spec_from_file(file, ax = ax[1], start = time, 
                                        fft_window_length = 512, sr = 10000, 
                                        fig = fig)
    ax[0].set_title(f'Spec of audio sample from \ntfrecords array nr. {r}'
                    f' | label: {lab}')
    ax[1].set_title(f'Spec of audio sample from file: \n{Path(file).stem}'
                    f' | time in file: {funcs.get_time(time/10000)}')

    fig.suptitle('Comparison between tfrecord audio and file audio')
    fig.savefig(f'{tfrec.TFRECORDS_DIR}_check_imgs/comp_{Path(file).stem}.png')
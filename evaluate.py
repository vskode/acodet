from hbdet.plot_utils import (plot_evaluation_metric, 
                              plot_model_results, 
                              plot_sample_spectrograms)
from hbdet import models
from hbdet.funcs import get_labels_and_preds
from hbdet.tfrec import run_data_pipeline, spec
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
import pandas as pd
import time
import numpy as np
from hbdet.humpback_model_dir import front_end
import hbdet.global_config as conf
tfrec_path =list(Path(conf.TFREC_DESTINATION).iterdir())

train_dates = [
    # '2022-11-22_03', 
    # '2022-11-22_06', 
    # '2022-11-22_11', 
    # '2022-11-22_17', 
    # # '2022-11-23_12', 
    # # '2022-11-23_17', 
    # # '2022-11-23_20', 
    # # '2022-11-24_00', 
    # '2022-11-24_02', 
    # '2022-11-24_10', 
    # # '2022-11-24_13', 
    # # '2022-11-24_14', 
    # '2022-11-24_16', 
    # '2022-11-25_00', 
    # '2022-11-24_12', 
    # '2022-11-24_17'
    # '2022-11-29_13',
    # '2022-11-29_17',
    # '2022-11-29_19',
    # '2022-11-29_21',
    # '2022-11-29_22',
    # '2022-11-30_11',
    # '2022-11-30_20',
    # '2022-11-30_13',
    # '2022-11-30_17',
    # '2022-12-01_02',
    # '2022-11-30_22',
    # '2022-12-01_00',
    '2022-11-30_01'
    ]

display_keys = [
    # 'data_path', 
    # 'batch_size', 
    'Model', 
    'keras_mod_name', 
    'epochs', 
    'init_lr', 
    'final_lr',    
    # 'weight_clipping', 
    ]

def get_info(date):
    keys = ['data_path', 'batch_size', 'epochs', 'Model',
            'keras_mod_name', 'load_weights', 'training_date',
            'steps_per_epoch', 'f_score_beta', 'f_score_thresh', 
            'bool_SpecAug', 'bool_time_shift', 'bool_MixUps', 
            'weight_clipping', 'init_lr', 'final_lr', 'unfreezes', 
            'preproc blocks']    
    path = Path(f'../trainings/{date}')
    f = pd.read_csv(path.joinpath('training_info.txt'), sep='\t')
    l, found = [], 0
    for key in keys:
        found = 0
        for s in f.values:
            if key in s[0]:
                l.append(s[0])
                found = 1
        if found == 0:
            l.append(f'{key}= nan')
    return {key: s.split('= ')[-1] for s, key in zip(l, keys)}

def write_trainings_csv():
    trains = list(Path('../trainings').iterdir())
    try:
        df = pd.read_csv('../trainings/20221124_meta_trainings.csv')
        new = [t for t in trains if t.stem not in df['training_date'].values]
        i = len(trains) - len(new) + 1
        trains = new
    except Exception as e:
        print('file not found', e)
        df = pd.DataFrame()
        i = 0
    for path in trains:
        try:
            f = pd.read_csv(path.joinpath('training_info.txt'), sep='\t')
            for s in f.values:
                if '=' in s[0]:
                    df.loc[i, s[0].split('= ')[0].replace(' ', '')] = s[0].split('= ')[-1]
                    df.loc[i, 'training_date'] = path.stem
            i += 1
        except Exception as e:
            print(e)
        df.to_csv('../trainings/20221124_meta_trainings.csv')
            


def create_overview_plot(train_dates, val_set, display_keys, plot_metrics=True):
    
    df = pd.read_csv('../trainings/20221124_meta_trainings.csv')
    df.index = df['training_date']
    # info_dicts = [get_info(date) for date in train_dates]

    # val_s = ''.join([Path(s).stem.split('_2khz')[0]+';' for s in val_set])
    if isinstance(val_set, list):
        val_label = 'all'
    else:
        val_label = Path(val_set).stem
    string = str(
        # 'batch:{}; ' 
        'Model:{}; ' 
        'keras_mod_name:{}; ' 
        'epochs:{}; ' 
        # 'training_date:{}; ' 
        'init_lr:{}; ' 
        'lr_end:{}; ' 
        # 'clip:{} ; '
        f'val: {val_label}')
    if conf.THRESH != 0.5:
        string += f' thr: {conf.THRESH}'


    # labels = [string.format(*[d[k] for k in display_keys]) for d in info_dicts]
    labels = [string.format(*df.loc[df['training_date'] == d, 
                                    display_keys].values[0]) for d in train_dates]

    training_runs = []
    for i, train in enumerate(train_dates):
        training_runs += list(Path(f'../trainings/{train}').glob('unfreeze*'))
        for _ in range(len(list(Path(f'../trainings/{train}').glob('unfreeze*')))):
            labels += labels[i]
    val_data = run_data_pipeline(val_set, 'val', return_spec=False)


    model_name = [df.loc[df['training_date'] == d, 'Model'].values[0] for d in train_dates]
    keras_mod_name = [df.loc[df['training_date'] == d, 'keras_mod_name'].values[0] for d in train_dates]
    model_class = list(map(lambda x: getattr(models, x), model_name))

    time_start = time.strftime('%Y%m%d_%H%M%S', time.gmtime())
    
    if plot_metrics:
        fig = plt.figure(constrained_layout=True, figsize=(15, 15))
        subfigs = fig.subfigures(2, 1)#, wspace=0.07, width_ratios=[1, 1])
        plot_model_results(train_dates, labels, fig=subfigs[0], legend=False)#, **info_dict)
        eval_fig = subfigs[1]
    else:
        fig = plt.figure(constrained_layout=True, figsize=(5, 5))
        eval_fig = fig
        display_keys = ['keras_mod_name']
        table_df=df.loc[train_dates, display_keys]
        table_df.iloc[-1] = 'GoogleModel'
        # titles = [t[0] for t in table_df.values]
        titles = [f'config {i}' for i in range(1, 9)]
    plot_evaluation_metric(model_class, training_runs, val_data, plot_labels=labels,
                            fig = eval_fig, plot_pr=False, plot_cm=True, titles=titles,
                            train_dates=train_dates, label=None, legend=False, 
                            keras_mod_name=keras_mod_name)

    fig.savefig(f'../trainings/{train_dates[-1]}/{time_start}_results_combo.png', dpi=150)

def create_incorrect_prd_plot(model_instance, train_date, val_data_path, **kwargs):
    training_run = Path(f'../trainings/{train_date}').glob('unfreeze*')
    val_data = run_data_pipeline(val_data_path, 'val', return_spec=False)
    labels, preds = get_labels_and_preds(model_instance, training_run, 
                                         val_data, **kwargs)
    preds = preds.reshape([len(preds)])
    bin_preds = list(map(lambda x: 1 if x >= conf.THRESH else 0, preds))
    false_pos, false_neg = [], []
    for i in range(len(preds)):
        if bin_preds[i] == 0 and labels[i] == 1:
            false_neg.append(i)
        if bin_preds[i] == 1 and labels[i] == 0:
            false_pos.append(i)
            
    offset = min([false_neg[0], false_pos[0]])
    val_data = run_data_pipeline(val_data_path, 'val', return_spec=False, 
                                 return_meta=True)
    val_data = val_data.batch(1)
    val_data = val_data.map(lambda x, y, z, w: (spec()(x), y, z, w))
    val_data = val_data.unbatch()
    data = list(val_data.skip(offset))
    fp = [data[i-offset] for i in false_pos]
    fn = [data[i-offset] for i in false_neg]
    plot_sample_spectrograms(fn, dir = train_date,
                    name=f'False_Negative', plot_meta=True, **kwargs)
    plot_sample_spectrograms(fp, dir = train_date,
                    name=f'False_Positive', plot_meta=True, **kwargs)
   
def create_table():
    time_start = time.strftime('%Y%m%d_%H%M%S', time.gmtime())
    df = pd.read_csv('../trainings/20221124_meta_trainings.csv')
    df.index = df['training_date']
    # display_keys = ['bool_SpecAug', 'bool_time_shift', 'bool_MixUps']
    # col_labels = ['Spec Aug', 'time shift', 'MixUp']
    display_keys = ['keras_mod_name']
    col_labels = ['model name']
    table_df=df.loc[train_dates, display_keys]
    table_df.iloc[-1] = 'GoogleModel'
    f, ax_tb = plt.subplots()
    bbox = [0, 0, 1, 1]
    ax_tb.axis('off')
    font_size = 20
    import seaborn as sns
    color = list(sns.color_palette())
    mpl_table = ax_tb.table(cellText=table_df.values, rowLabels=['     ']*len(table_df), 
                        bbox=bbox, colLabels=col_labels,
                        rowColours = color)
    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(font_size)
    f.tight_layout()
    f.savefig(f'../trainings/{train_dates[-1]}/{time_start}_results_table.png', dpi=150)

# create_incorrect_prd_plot(GoogleMod, train_dates[0], tfrec_path)
# for path in tfrec_path:
# write_trainings_csv()
    # try:
# df = pd.read_csv('../trainings/20221124_meta_trainings.csv')
# string = str('bool_SpecAug:{}; ' 'bool_time_shift:{}; ' 'bool_MixUps:{}; ')
# labels = [string.format(*df.loc[df['training_date'] == d, 
#             ['bool_SpecAug', 'bool_time_shift', 'bool_MixUps']].values[0]) for d in train_dates]
# plot_model_results(train_dates, legend=False)
# create_table()
create_overview_plot(train_dates, tfrec_path, display_keys, plot_metrics=False)
    # except:
    #     print(path, 'not working')
    #     continue

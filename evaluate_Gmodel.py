from hbdet.plot_utils import plot_evaluation_metric, plot_model_results, create_and_save_figure
from hbdet.google_funcs import GoogleMod
from hbdet.funcs import load_config
from hbdet import tfrec
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import time
from hbdet.humpback_model_dir import front_end

config = load_config()

tfrec_path =[
    # 'Daten/Datasets/ScotWest_v5_2khz', 
    # 'Daten/Datasets/ScotWest_v4_2khz',
    # 'Daten/Datasets/Mixed_v1_2khz',
    # 'Daten/Datasets/Mixed_v2_2khz',
    'Daten/Datasets/Benoit_v1_2khz',
    ]

train_dates = [
    # '2022-05-00_00',
    # '2022-11-07_16',
    # '2022-11-07_21',
    # '2022-11-08_03',
    # '2022-11-08_11',
    # '2022-11-08_16',
    # '2022-11-08_19',
    # '2022-11-09_03',
    # '2022-11-14_16',
    # '2022-11-10_18',
    '2022-11-16_09'
              ]

display_keys = [
    # 'data_path', 
    # 'batch_size', 
    'bool_time_shift', 
    'bool_MixUps', 
    'init_lr', 
    'final_lr',    
    # 'weight_clipping', 
    ]

def get_info(date):
    keys = ['data_path', 'batch_size', 'epochs', 'load_weights', 
            'steps_per_epoch', 'f_score_beta', 'f_score_thresh', 
            'bool_time_shift', 'bool_MixUps', 'weight_clipping', 
            'init_lr', 'final_lr', 'unfreezes', 'preproc blocks']    
    path = Path(f'trainings/{date}')
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


def create_overview_plot(train_dates, val_set, display_keys):
    info_dicts = [get_info(date) for date in train_dates]

    val_s = ''.join([Path(s).stem.split('_2khz')[0]+';' for s in val_set])
    string = str(
        # 'batch:{}; ' 
        't_aug:{}; ' 
        'mixup:{}; ' 
        'lr_beg:{}; ' 
        'lr_end:{}; ' 
        # 'clip:{} ; '
        f'val: {val_s}')
    if config.thresh != 0.5:
        string += f' thr: {config.thresh}'


    labels = [string.format(*[d[k] for k in display_keys]) for d in info_dicts]

    training_runs = []
    for i, train in enumerate(train_dates):
        training_runs += list(Path(f'trainings/{train}').glob('unfreeze*'))
        for _ in range(len(list(Path(f'trainings/{train}').glob('unfreeze*')))):
            labels += labels[i]
    val_data = tfrec.run_data_pipeline(val_set, 'val', return_spec=False)


    time_start = time.strftime('%Y%m%d_%H%M%S', time.gmtime())
    fig = plt.figure(constrained_layout=True, figsize=(15, 15))
    subfigs = fig.subfigures(2, 1)#, wspace=0.07, width_ratios=[1, 1])

    plot_model_results(train_dates, labels, fig=subfigs[0], legend=False)#, **info_dict)
    plot_evaluation_metric(GoogleMod, training_runs, val_data, plot_labels=labels,
                            fig = subfigs[1], plot_pr=True, plot_cm=True, 
                            train_dates=train_dates, label=None)

    fig.savefig(f'trainings/{train_dates[-1]}/{time_start}_results_combo.png')



create_overview_plot(train_dates, tfrec_path, display_keys)



    # except Exception as e:
    #     print(train, ' failed', e)
    #     continue

# import tensorflow as tf
# class EffNet:
#     def __init__(self, **kwargs) -> None:
#         self.model = tf.keras.applications.EfficientNetB5(
#             include_top=True,
#             weights=None,
#             input_tensor=None,
#             input_shape=[128, 64, 1],
#             pooling=None,
#             classes=1,
#             classifier_activation="sigmoid"
#         )
    
#     def load_ckpt(self, ckpt_path, ckpt_name='last'):
#         try:
#             file_path = Path(ckpt_path).joinpath(f'cp-{ckpt_name}.ckpt.index')
#             if not file_path.exists():
#                 ckpts = list(ckpt_path.glob('cp-*.index'))
#                 ckpts.sort()
#                 ckpt = ckpts[-1]
#             else:
#                 ckpt = file_path
#             self.model.load_weights(
#                 str(ckpt).replace('.index', '')
#                 ).expect_partial()
#         except Exception as e:
#             print('Checkpoint not found.', e)
            
#     def change_input_to_array(self):
#         """
#         change input layers of model after loading checkpoint so that a file
#         can be predicted based on arrays rather than spectrograms, i.e.
#         reintegrate the spectrogram creation into the model. 

#         Args:
#             model (tf.keras.Sequential): keras model

#         Returns:
#             tf.keras.Sequential: model with new arrays as inputs
#         """
#         model_list = self.model.layers
#         model_list.insert(0, tf.keras.layers.Input([7755]))
#         model_list.insert(1, tf.keras.layers.Lambda(
#                             lambda t: tf.expand_dims(t, -1)))
#         model_list.insert(2, front_end.MelSpectrogram())
#         self.model = tf.keras.Sequential(layers=[layer for layer in model_list])

# fig = plt.figure(constrained_layout=True, figsize=(15, 15))
# subfigs = fig.subfigures(2, 1)#, wspace=0.07, width_ratios=[1, 1])

# plot_model_results(train_dates, labels, fig=subfigs[0], legend=False)#, **info_dict)
# plot_evaluation_metric(EffNet, training_runs, val_data, plot_labels=labels,
#                         fig = subfigs[1], plot_pr=True, plot_cm=True, 
#                         train_dates=train_dates, label=None,
#                 )

# fig.savefig(f'trainings/{train_dates[-1]}/model_results_combo.png')
    # except Exception as e:
    #     print(train, ' failed', e)
    #     continue
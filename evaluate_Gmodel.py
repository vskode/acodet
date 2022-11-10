from hbdet.plot_utils import plot_evaluation_metric, plot_model_results
from hbdet.google_funcs import GoogleMod
from hbdet import tfrec
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd

def get_info(date, keys):
    path = Path(f'trainings/{date}')
    f = pd.read_csv(path.joinpath('training_info.txt'), sep='\t')
    l = [[s[0] for s in f.values if key in s[0]] for key in keys]
    return {key: s[0].split('= ')[-1] for s, key in zip(l, keys)}

# for train in Path('trainings').iterdir():
#     try:
tfrec_path = ['Daten/Datasets/ScotWest_v5_2khz', 
              'Daten/Datasets/ScotWest_v4_2khz',
              'Daten/Datasets/ScotWest_v6_2khz',
              'Daten/Datasets/ScotWest_v7_2khz']

train_dates = ['2022-11-07_13',
              '2022-11-07_16',
              '2022-11-07_21',
              '2022-11-08_03',
              '2022-11-08_11',
              '2022-11-08_16',
              '2022-11-08_19',
              '2022-11-09_03']

batch_size = [64, 32, 64, 32, 32, 64, 32, 64]
time_augs = [False, True, True, True]*2
mixup_augs = [True, False, True, True]*2
init_lr = [1e-4, 1e-3] *4
final_lr = [5e-6, 1e-6] *4
weight_clip = [1]*4 + [0.8] *4

string = str('batch:{}; ' 't_aug:{}; ' 'mixup:{}; ' 'lr_0:{}; ' 'lr_1:{}; ' 'clip:{}')
# string = str('batch size:{}\n'
#           'time shift augment:{}\n'
#           'mix up augment:{}\n'
#           'starting lr:{}\n'
#           'final lr:{}\n'
#           'clip val:{}')

labels = [string.format(batch_size[i], time_augs[i], mixup_augs[i], 
                        init_lr[i], final_lr[i], weight_clip[i])
          for i in range(len(time_augs))]

# batch_size = 32
# info_dict = get_info(train_date, ['epochs', 'final_lr', 'num_of_shifts'])
# info_dict['shifts'] = info_dict.pop('num_of_shifts')
# create_and_save_figure(GoogleMod, tfrec_path, batch_size, train_date, 
#                         plot_cm=True, **info_dict)
# info_dict = get_info(train_date, ['dataset', 'final_lr'])
training_runs = []
for train in train_dates:
    training_runs += list(Path(f'trainings/{train}').glob('unfreeze*'))
val_data = tfrec.run_data_pipeline(tfrec_path, 'val', return_spec=False)

fig = plt.figure(constrained_layout=True, figsize=(15, 15))
subfigs = fig.subfigures(2, 1)#, wspace=0.07, width_ratios=[1, 1])

plot_model_results(train_dates, labels, fig=subfigs[0], legend=False)#, **info_dict)
plot_evaluation_metric(GoogleMod, training_runs, val_data, plot_labels=labels,
                        fig = subfigs[1], plot_pr=True, plot_cm=True, 
                        train_dates=train_dates, label=None)

fig.savefig(f'trainings/{train_dates[-1]}/model_results_combo.png')
    # except Exception as e:
    #     print(train, ' failed', e)
    #     continue

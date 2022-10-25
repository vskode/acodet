from hbdet.plot_utils import create_and_save_figure, plot_model_results
from hbdet.google_funcs import GoogleMod
from pathlib import Path
import pandas as pd

def get_info(date, keys):
    path = Path(f'trainings/{date}')
    f = pd.read_csv(path.joinpath('training_info.txt'), sep='\t')
    l = [[s[0] for s in f.values if key in s[0]] for key in keys]
    return {key: s[0].split('= ')[-1] for s, key in zip(l, keys)}

# for train in Path('trainings').iterdir():
#     try:
tfrec_path = 'Daten/Datasets/ScotWest_v1/tfrecords_0s_shift'
train_date = '2022-10-24_09'
batch_size = 32
info_dict = get_info(train_date, ['epochs', 'final_lr', 'num_of_shifts'])
info_dict['shifts'] = info_dict.pop('num_of_shifts')
create_and_save_figure(GoogleMod, tfrec_path, batch_size, train_date, 
                        plot_cm=True, **info_dict)
info_dict = get_info(train_date, ['dataset', 'final_lr'])
plot_model_results(train_date, **info_dict)
    # except Exception as e:
    #     print(train, ' failed', e)
    #     continue

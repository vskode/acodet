import yaml
import time
from pathlib import Path
from hbdet.google_funcs import GoogleMod
from hbdet.funcs import get_files, init_model, gen_raven_annotation

with open('hbdet/hbdet/config.yml', 'r') as f:
    config = yaml.safe_load(f)
    
if __name__ == '__main__':
    time_start = time.strftime('%Y-%m-%d_%H', time.gmtime())
    tfrec_path = 'Daten/Datasets/ScotWest_v1/tfrecords_0s_shift'
    train_date = '2022-10-21_15'
    files = get_files(search_str='*_01-00*')
    for file in files:
        # TODO init model in die function reinschieben damit ich die selbe function wie vorher schon nutzen kann
        model = init_model(GoogleMod, f'trainings/{train_date}/unfreeze_no-TF', 
                           load_g_ckpt=False)
        gen_raven_annotation(file, model, f'{train_date}_no-TF', time_start)
import yaml
import time
from hbdet.google_funcs import GoogleMod
from hbdet.funcs import get_files, gen_annotations

with open('hbdet/hbdet/config.yml', 'r') as f:
    config = yaml.safe_load(f)
    
if __name__ == '__main__':
    time_start = time.strftime('%Y-%m-%d_%H', time.gmtime())
    tfrec_path = 'Daten/Datasets/ScotWest_v1/tfrecords_0s_shift'
    train_date = '2022-10-21_15'
    files = get_files(search_str='*_01-00*')
    for file in files:
        gen_annotations(file, GoogleMod, training_path='trainings', 
                        mod_label=train_date, time_start=time_start)
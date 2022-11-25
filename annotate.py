import time
from hbdet import models
from hbdet.funcs import get_files, gen_annotations, init_model, get_dt_filename
from hbdet import global_config as conf
import pandas as pd
import numpy as np
    
if __name__ == '__main__':
    time_start = time.strftime('%Y-%m-%d_%H', time.gmtime())
    train_date = '2022-11-24_17'
    files = get_files(location=conf.SOUND_FILES_SOURCE,
                      search_str='**/*wav')
    # files = get_files(location='/media/vincent/Extreme SSD/MA/for_manual_annotation/src_to_be_annotated/resampled_2kHz',
    #                   search_str='**/*wav')
    df = pd.read_csv('../trainings/20221124_meta_trainings.csv')
    row = df.loc[df['training_date'] == train_date]
    model_name = row.Model.values[0]
    keras_mod_name = row.keras_mod_name.values[0]
    model_class = getattr(models, model_name)
    
    model = init_model(model_class, 
                       f'../trainings/{train_date}/unfreeze_no-TF', 
                       keras_mod_name=keras_mod_name)
    df = pd.DataFrame(columns=['Date', 'Daily_Presence', 
                                *['%.2i' % i for i in np.arange(24)]])
    for i, file in enumerate(files):
        try:
            
            annot = gen_annotations(file, model, mod_label=train_date, 
                                 time_start=time_start)
            
            file_dt = get_dt_filename(file)
            df.loc[i+1, 'Date'] = str(file_dt.date())
            # df.loc[i+1, '%.2i'%file_dt.hour] = return_hourly_presence(annot)
            
        except Exception as e:
            print(f"{file} couldn't be loaded, continuing with next file.\n", e)
            continue
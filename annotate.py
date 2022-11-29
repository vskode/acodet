import time
from hbdet import models
from hbdet.funcs import get_files, gen_annotations, init_model, get_dt_filename
from hbdet import global_config as conf
import pandas as pd
import numpy as np
from pathlib import Path


class MetaData:
    def __init__(self):
        self.filename = 'filename'
        self.f_dt = 'date from timestamp'
        self.n_pred_col = 'number of predictions'
        self.avg_pred_col = 'average prediction value'
        self.n_pred08_col = 'number of predictions with thresh>0.8'
        self.n_pred09_col =  'number of predictions with thresh>0.9'
        self.df = pd.DataFrame(columns=[self.filename, 
                                        self.f_dt, 
                                        self.n_pred_col, 
                                        self.avg_pred_col, 
                                        self.n_pred08_col, 
                                        self.n_pred09_col])
        
    def append_and_save_meta_file(self, annot):
        self.df.loc[f_ind, self.f_dt] = str(get_dt_filename(file).date())
        self.df.loc[f_ind, self.filename] = Path(file).relative_to(
                                                        conf.SOUND_FILES_SOURCE)
        self.df.loc[f_ind, self.n_pred_col] = len(annot)
        self.df.loc[f_ind, self.avg_pred_col] = np.mean(annot[conf.ANNOTATION_COLUMN])
        self.df.loc[f_ind, self.n_pred08_col] = len(annot.loc[annot[
                                                conf.ANNOTATION_COLUMN]>0.8])
        self.df.loc[f_ind, self.n_pred09_col] = len(annot.loc[annot[
                                                conf.ANNOTATION_COLUMN]>0.9])
        self.df.to_csv(f'../generated_annotations/{time_start}/stats.csv')
    
if __name__ == '__main__':
    time_start = time.strftime('%Y-%m-%d_%H', time.gmtime())
    train_date = '2022-11-25_16'
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
    mdf = MetaData()
    f_ind = 0
    for i, file in enumerate(files):    
        try:
            f_ind += 1
            annot = gen_annotations(file, model, mod_label=train_date, 
                                 time_start=time_start)
            mdf.append_and_save_meta_file(annot)

        except Exception as e:
            print(f"{file} couldn't be loaded, continuing with next file.\n", e)
            continue
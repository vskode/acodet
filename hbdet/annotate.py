import time
from hbdet import models
from hbdet.funcs import (get_files, gen_annotations, 
                         get_dt_filename, 
                         remove_str_flags_from_predictions)
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
        self.time_per_file = 'computing time [s]'
        self.df = pd.DataFrame(columns=[self.filename, 
                                        self.f_dt, 
                                        self.n_pred_col, 
                                        self.avg_pred_col, 
                                        self.n_pred08_col, 
                                        self.n_pred09_col])
        
    def append_and_save_meta_file(self, file, annot, f_ind, time_start,
                                  relativ_path = conf.SOUND_FILES_SOURCE,
                                  computing_time = 'not calculated'):
        self.df.loc[f_ind, self.f_dt] = str(get_dt_filename(file).date())
        self.df.loc[f_ind, self.filename] = Path(file).relative_to( #TODO relative_path muss noch dauerhaft geändert werden
                                                        relativ_path)
        self.df.loc[f_ind, self.n_pred_col] = len(annot)
        df_clean = remove_str_flags_from_predictions(annot)
        self.df.loc[f_ind, self.avg_pred_col] = np.mean(df_clean[conf.ANNOTATION_COLUMN])
        self.df.loc[f_ind, self.n_pred08_col] = len(df_clean.loc[df_clean[
                                                conf.ANNOTATION_COLUMN]>0.8])
        self.df.loc[f_ind, self.n_pred09_col] = len(df_clean.loc[df_clean[
                                                conf.ANNOTATION_COLUMN]>0.9])
        self.df.loc[f_ind, self.time_per_file] = computing_time
        self.df.to_csv(Path(conf.GEN_ANNOTS_DIR).joinpath(time_start)
                       .joinpath('stats.csv'))
    
def run_annotation(train_date=None):
    time_start = time.strftime('%Y-%m-%d_%H-%M-%S', time.gmtime())
    files = get_files(location=conf.SOUND_FILES_SOURCE,
                      search_str='**/*')
    
    if not train_date:
        model = models.init_model()
        mod_label = conf.MODEL_NAME
    else:
        df = pd.read_csv('../trainings/20221124_meta_trainings.csv')
        row = df.loc[df['training_date'] == train_date]
        model_name = row.Model.values[0]
        keras_mod_name = row.keras_mod_name.values[0]
        
        model = models.init_model(model_instance=model_name, 
                        checkpoint_dir=f'../trainings/{train_date}/unfreeze_no-TF', 
                        keras_mod_name=keras_mod_name)
        mod_label = train_date
    mdf = MetaData()
    f_ind = 0
    for i, file in enumerate(files):    
        if file.is_dir():
            continue
        try:
            f_ind += 1
            start = time.time()
            annot = gen_annotations(file, model, mod_label=mod_label, 
                                    time_start=time_start)
            computing_time = time.time() - start
            mdf.append_and_save_meta_file(file, annot, f_ind, time_start,
                                            computing_time=computing_time)

        except Exception as e:
            print(f"{file} couldn't be loaded, continuing with next file.\n", 
                  e)
            continue
    return time_start    
    
def filter_annots_by_thresh(time_dir=None):
    if not time_dir:
        path = conf.GEN_ANNOT_SRC
    else:
        path = Path(conf.GEN_ANNOTS_DIR).joinpath(time_dir)
    files = get_files(location=path, search_str='**/*txt')
    files = [f for f in files if 'thresh_0.5' in str(f.parent)]
    for i, file in enumerate(files):
        try:
            annot = pd.read_csv(file, sep='\t')        
            annot = annot.loc[annot[conf.ANNOTATION_COLUMN] >= conf.THRESH]
            save_dir = (Path(path)
                        .joinpath(f'thresh_{conf.THRESH}')
                        .joinpath(file.relative_to(Path(path)
                                                   .joinpath('thresh_0.5'))).parent)
            save_dir.mkdir(exist_ok=True, parents=True)
            
            annot.index  = np.arange(1, len(annot)+1)
            annot.index.name = 'Selection'
            annot.to_csv(save_dir.joinpath(file.stem+file.suffix), sep='\t')
            print(f'Writing file {i+1}/{len(files)}')
        except Exception as e:
            print('Could not process file, maybe not an annotation file?', 
                  'Error: ', e)

def generate_stats():
    files = get_files(location=conf.GEN_ANNOT_SRC, search_str='**/*txt')
    mdf = MetaData()
    f_ind = 0
    for i, file in enumerate(files):    
        f_ind += 1
        annot = pd.read_csv(file, sep='\t')
        mdf.append_and_save_meta_file(file, annot, f_ind, 
                                        Path(conf.GEN_ANNOT_SRC).stem,
                                        relativ_path=conf.GEN_ANNOT_SRC)

if __name__ == '__main__':
    train_dates = [
        '2022-11-30_01'
        ]

    for train_date in train_dates:
        start = time.time()
        run_annotation(train_date)
        end = time.time()
        print(end-start)
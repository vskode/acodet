import pandas as pd
import glob
from ketos.data_handling import selection_table as sl
from pathlib import Path
import os
from . import funcs

config = funcs.load_config()
annotation_files = Path(config.annotations_root_dir).glob('/**/*.txt')

def compensate_for_naming_inconsistencies(hard_drive_path):
    if not file_path:
        new_file = file.stem.split("Table")[0] + 'wav'
        file_tolsta = '336097327.'+new_file[6:].replace('_000', '').replace('_', '')
        file_path = glob.glob(f'{hard_drive_path}/**/{file_tolsta}',
                    recursive = True)
        
        if not file_path :
            file_tolsta = '335564853.'+new_file[6:].replace('5_000', '4').replace('_', '')
            file_path = glob.glob(f'{hard_drive_path}/**/{file_tolsta}',
                        recursive = True)
            if not file_path :
                return f'{file.stem.split("Table")[0]}wav'
    return file_path

def get_corresponding_sound_file(file):
    hard_drive_path = config.files_root_dir
    file_path = glob.glob(f'{hard_drive_path}/**/{file.stem.split("Table")[0]}wav',
                      recursive = True)
    
    file_path = compensate_for_naming_inconsistencies(hard_drive_path)
        
    if len(file_path) > 1:
        for path in file_path:
            if file.parent.parent.stem in path:
                file_path = path
    else:
        file_path = file_path[0]

    return file_path
    
def standardize_annotations(file):
    ann = pd.read_csv(file, sep = '\t')

    ann['filename'] = get_corresponding_sound_file(file)
    ann['label']    = 1
    map_to_ketos_annot_std = {'Begin Time (s)': 'start', 
                              'End Time (s)': 'end',
                              'Low Freq (Hz)' : 'freq_min', 
                              'High Freq (Hz)' : 'freq_max',} 
    std_annot_train = sl.standardize(table=ann,
                                    mapper = map_to_ketos_annot_std, 
                                    trim_table=True)
    return std_annot_train
    
def save_ket_annot_only_existing_paths(df):
    check_if_full_path_func = lambda x: x[0] == '/'
    df[list( map(check_if_full_path_func, 
        df.index.get_level_values(0)) )].to_csv(
        'Daten/ket_annot_file_exists.csv')

if __name__ == '__main__':
    df = pd.DataFrame()
    for file in list(annotation_files):
        
        df = df.append(standardize_annotations(file))
        
    df.to_csv('Daten/ket_annot.csv')
    save_ket_annot_only_existing_paths(df)
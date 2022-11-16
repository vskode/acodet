import pandas as pd
import glob
from ketos.data_handling import selection_table as sl
from pathlib import Path
import os
from hbdet import funcs

config = funcs.load_config()
# annotation_files = Path(config.annotations_root_dir).glob('/**/*.txt')
annotation_files = Path(r'/mnt/f/Daten/20221019-Benoit/').glob('**/*.txt')
# annotation_files = Path(r'generated_annotations/2022-11-04_12/').glob('ch*.txt')

def compensate_for_naming_inconsistencies(hard_drive_path, file):
    
    split_file = file.stem.split("Table")[0]
    file_path = glob.glob(f'{hard_drive_path}/**/{split_file}*wav',
            recursive = True)
    
    if not file_path:
        file_tolsta = '336097327.'+split_file[6:].replace('_000', '').replace('_', '')
        file_path = glob.glob(f'{hard_drive_path}/**/{file_tolsta}*wav',
                    recursive = True)
    
    if not file_path :
        file_tolsta = '335564853.'+split_file[6:].replace('5_000', '4').replace('_', '')
        file_path = glob.glob(f'{hard_drive_path}/**/{file_tolsta}*wav',
                    recursive = True)
        
    if not file_path :
        file_new_annot = file.stem.split('_annot')[0]
        file_path = glob.glob(f'{hard_drive_path}/**/{file_new_annot}*wav',
                    recursive = True)
        
    if not file_path :
        split_file_underscore = file.stem.split('_')[0]
        file_path = glob.glob(f'{hard_drive_path}/**/{file_new_annot}*wav',
                    recursive = True)
        if not file_path :
            file_new_annot = split_file_underscore.split('.')[-1]
            file_path = glob.glob(f'{hard_drive_path}/**/*{file_new_annot}*wav',
                        recursive = True)
        
    if not file_path :
        return False
    return file_path

def get_corresponding_sound_file(file):
    hard_drive_path = config.files_root_dir
    file_path = glob.glob(f'{hard_drive_path}/**/{file.stem}*wav',
                      recursive = True)
    if not file_path:
        file_path = compensate_for_naming_inconsistencies(hard_drive_path, file)
        
    if len(file_path) > 1:
        for path in file_path:
            if file.parent.parent.stem in path:
                file_path = path
    else:
        file_path = file_path[0]

    return file_path
    
def standardize_annotations(file, all_noise=False):
    ann = pd.read_csv(file, sep = '\t')

    try:
        ann['filename'] = get_corresponding_sound_file(file)
    except:
        print(f'corresponding sound file for annotations file: {file} not found')
        
    ann['label']    = 0 if all_noise else 1
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
        
def leading_underscore_in_parent_dirs(file):
    return '_' in [f.stem[0] for f in list(file.parents)[:-1]]

if __name__ == '__main__':
    df = pd.DataFrame()
    files = list(annotation_files)
    for i, file in enumerate(files):
        if leading_underscore_in_parent_dirs(file):
            print(file, ' skipped due to leading underscore in parent dir.')
            continue
        df = pd.concat([df, standardize_annotations(file, all_noise=False)])
        print(f'Completed file {i}/{len(files)}.', end='\r')
        
    # TODO include date in path by default
    df.to_csv('Daten/Benoit_annots.csv')
    save_ket_annot_only_existing_paths(df)

import pandas as pd
import glob
from ketos.data_handling import selection_table as sl
from pathlib import Path
import os
from hbdet.funcs import load_config
import numpy as np

config = load_config()
annotation_files = Path(config.annotation_source).glob('**/*.txt')
# annotation_files = Path(r'/mnt/f/Daten/20221019-Benoit/').glob('**/*.txt')
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
    hard_drive_path = config.sound_files_source
    file_path = glob.glob(f'{hard_drive_path}/**/{file.stem}*wav',
                      recursive = True)
    if not file_path:
        file_path = compensate_for_naming_inconsistencies(hard_drive_path, file)
        
    if len(file_path) > 1:
        p_dir = list(file.relative_to(config.annotation_source).parents)[-2]
        p_dir_main = str(p_dir).split('_')[0]
        for path in file_path:
            if p_dir_main in path:
                file_path = path
    else:
        file_path = file_path[0]
        
    if isinstance(file_path, list) and len(file_path) > 1:
        file_path = file_path[0]
        print('WARNING: Multiple sound files for annotations file found.'
              ' Because pattern could not be resolved, first file is chosen.'
              f'\nannotations file name: \n{file}\n'
              f'sound file name: \n{file_path}\n')

    return file_path

annotation_column = 'Prediction/Comments'
def differentiate_label_flags(df, flag=None):
    df.loc[df[annotation_column]=='c', 'label'] = 1
    df.loc[df[annotation_column]=='n', 'label'] = 'explicit 0'
    df = df.drop(df.loc[df[annotation_column]=='u'].index)
    if flag == 'noise':
        df.loc[df[annotation_column] > 0.9, 'label'] = 'explicit 0'
    return df

def get_labels(file, df, active_learning=False):
    if not active_learning:
        df['label'] = 1
    else:
        noise_flag, annotated_flag, calls_flag = ['_allnoise', '_annotated', '_allcalls']
        df = df.iloc[df.Selection.drop_duplicates().index]
        if calls_flag in file.stem:
            df['label'] = 1
            df = differentiate_label_flags(df, flag='calls')
        elif noise_flag in file.stem:
            df['label'] = 0
            df = differentiate_label_flags(df, flag='noise')
        elif annotated_flag in file.stem:
            df = differentiate_label_flags(df)
    return df
            
def standardize(df, *, mapper, filename_col='filename',
                selection_col='Selection'):
    keep_cols = ['label', 'start', 'end', 'freq_min', 'freq_max']
    df = df.rename(columns=mapper)
    out_df = df[keep_cols]
    out_df.index = pd.MultiIndex.from_arrays(arrays=(df[filename_col], 
                                                     df[selection_col]))
    return out_df.astype(dtype=np.float64)
    
    
    
def finalize_annotation(file, all_noise=False, **kwargs):
    ann = pd.read_csv(file, sep = '\t')

    ann['filename'] = get_corresponding_sound_file(file)
    # if not ann['filename']:
    #     print(f'corresponding sound file for annotations file: {file} not found')
        
    ann = get_labels(file, ann, **kwargs)
    map_to_ketos_annot_std = {'Begin Time (s)': 'start', 
                              'End Time (s)': 'end',
                              'Low Freq (Hz)' : 'freq_min', 
                              'High Freq (Hz)' : 'freq_max',} 
    # std_annot_train1 = sl.standardize(table=ann,
    #                                 mapper = map_to_ketos_annot_std, 
    #                                 trim_table=True)
    ann_explicit_noise = ann.loc[ann['label']=='explicit 0', :]
    ann_explicit_noise['label'] = 0
    ann = ann.drop(ann.loc[ann['label']=='explicit 0'].index)
    std_annot_train = standardize(ann, mapper=map_to_ketos_annot_std)
    std_annot_enoise = standardize(ann_explicit_noise, 
                                   mapper=map_to_ketos_annot_std)
    return std_annot_train, std_annot_enoise
    
def save_ket_annot_only_existing_paths(df):
    check_if_full_path_func = lambda x: x[0] == '/'
    df[list( map(check_if_full_path_func, 
        df.index.get_level_values(0)) )].to_csv(
        'Daten/ket_annot_file_exists.csv')
        
def leading_underscore_in_parent_dirs(file):
    return '_' in [f.stem[0] for f in list(file.parents)[:-1]]

def get_active_learning_files(files):
    cases = ['_allnoise', '_annotated', '_allcalls']
    cleaned_files = [f for f in files if [True for c in cases if c in f.stem]]
    drop_cases = ['_tobechecked']
    final_cleanup = [f for f in cleaned_files if not \
                                    [True for d in drop_cases if d in f.stem]]
    return final_cleanup

def main(annotation_files, active_learning=False):
    files = list(annotation_files)
    if active_learning:
        files = get_active_learning_files(files)
    folders, counts = np.unique([list(f.relative_to(config.annotation_source)
                                .parents)[-2] for f in files],
                        return_counts=True)
    files.sort()
    ind = 0
    for i, folder in enumerate(folders):
        df_t, df_n = pd.DataFrame(), pd.DataFrame()
        for _ in range(counts[i]):
            if leading_underscore_in_parent_dirs(files[ind]):
                print(files[ind], ' skipped due to leading underscore in parent dir.')
                continue
            df_train, df_enoise = finalize_annotation(files[ind], all_noise=False, 
                                                active_learning=active_learning)
            df_t = pd.concat([df_t, df_train])
            df_n = pd.concat([df_n, df_enoise])
            print(f'Completed file {ind}/{len(files)}.', end='\r')
            ind += 1
        
    # TODO include date in path by default
        save_dir = Path(config.annotation_destination).joinpath(folder)
        save_dir.mkdir(exist_ok=True, parents=True)
        df_t.to_csv(save_dir.joinpath('combined_annotations.csv'))
        df_n.to_csv(save_dir.joinpath('explicit_noise.csv'))
    # save_ket_annot_only_existing_paths(df)
    
if __name__ == '__main__':
    main(annotation_files, active_learning=True)

import pandas as pd
import glob
from pathlib import Path
import os
import numpy as np
import hbdet.global_config as conf

annotation_files = Path(conf.ANNOTATION_SOURCE).glob('**/*.txt')
# TODO aufraeumen
annotation_column = 'Prediction/Comments'
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
    hard_drive_path = conf.SOUND_FILES_SOURCE
    file_path = glob.glob(f'{hard_drive_path}/**/{file.stem}*wav',
                      recursive = True)
    if not file_path:
        file_path = compensate_for_naming_inconsistencies(hard_drive_path, file)
        
    if not file_path:
        return 'empty'
        
    if len(file_path) > 1:
        p_dir = list(file.relative_to(conf.ANNOTATION_SOURCE).parents)[-2]
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

def differentiate_label_flags(df, flag=None):
    df.loc[df[annotation_column]=='c', 'label'] = 1
    df.loc[df[annotation_column]=='n', 'label'] = 'explicit 0'
    for b in df.loc[df['End Time (s)']-df['Begin Time (s)'] \
        > conf.CONTEXT_WIN/conf.SR]:
        if df.loc[b.index, annotation_column] == 'n':
            1 # TODO noise aufschneiden und einfuegen
        
    df.loc[df[annotation_column]=='n' \
           and df['End Time (s)']-df['Begin Time (s)'] > conf.CONTEXT_WIN/conf.SR,
           'label'] = 0
    df = df.drop(df.loc[df[annotation_column]=='u'].index)
    if flag == 'noise':
        # df['not_float'] = ~((df[annotation_column] == 'u') ^ (df[annotation_column] == 'n') ^ (df[annotation_column] == 'c'))
        # index = df['not_float'] * df[annotation_column].astype(float) > 0.9
        df.loc[df[annotation_column] == 'u', annotation_column] = -9
        df.loc[df[annotation_column] == 'n', annotation_column] = -8
        df.loc[df[annotation_column] == 'c', annotation_column] = -7
        df.loc[df[annotation_column].astype(float) > 0.9, 'label'] = 'explicit 0'
        # df.loc[df[annotation_column] == -9] = 'u'
        # df.loc[df[annotation_column] == -8] = 'n'
        # df.loc[df[annotation_column] == -7] = 'c'
    return df

def get_labels(file, df, active_learning=False, **kwargs):
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
            df['label'] = 1
            df = differentiate_label_flags(df, flag='calls')
    return df
            
def standardize(df, *, mapper, filename_col='filename',
                selection_col='Selection'):
    keep_cols = ['label', 'start', 'end', 'freq_min', 'freq_max']
    df = df.rename(columns=mapper)
    out_df = df[keep_cols]
    out_df.index = pd.MultiIndex.from_arrays(arrays=(df[filename_col], 
                                                     df[selection_col]))
    return out_df.astype(dtype=np.float64)
    
def clean_benoits_data(df):
    df = df.loc[df['High Freq (Hz)'] <= 2000]
    df = df.loc[df['End Time (s)']-df['Begin Time (s)'] >= 0.4]
    return df
    
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
    if 'benoit' in kwargs:
        ann = clean_benoits_data(ann)
        
    ann_explicit_noise = ann.loc[ann['label']=='explicit 0', :]
    ann_explicit_noise['label'] = 0
    ann = ann.drop(ann.loc[ann['label']=='explicit 0'].index)
    std_annot_train = standardize(ann, mapper=map_to_ketos_annot_std)
    std_annot_enoise = standardize(ann_explicit_noise, 
                                   mapper=map_to_ketos_annot_std)
    
    return std_annot_train, std_annot_enoise
        
def leading_underscore_in_parent_dirs(file):
    return '_' in [f.stem[0] for f in list(file.parents)[:-1]]

def get_active_learning_files(files):
    cases = ['_allnoise', '_annotated', '_allcalls']
    cleaned_files = [f for f in files if [True for c in cases if c in f.stem]]
    drop_cases = ['_tobechecked']
    final_cleanup = [f for f in cleaned_files if not \
                                    [True for d in drop_cases if d in f.stem]]
    return final_cleanup

def main(annotation_files, active_learning=False, **kwargs):
    files = list(annotation_files)
    if active_learning:
        files = get_active_learning_files(files)
    folders, counts = np.unique([list(f.relative_to(conf.ANNOTATION_SOURCE)
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
            df_train, df_enoise = finalize_annotation(files[ind], 
                                                      all_noise=False, 
                                            active_learning=active_learning, 
                                                      **kwargs)
            df_t = pd.concat([df_t, df_train])
            df_n = pd.concat([df_n, df_enoise])
            print(f'Completed file {ind}/{len(files)}.', end='\r')
            ind += 1
        
    # TODO include date in path by default
        save_dir = Path(conf.ANNOTATION_DESTINATION).joinpath(folder)
        save_dir.mkdir(exist_ok=True, parents=True)
        df_t.to_csv(save_dir.joinpath('combined_annotations.csv'))
        df_n.to_csv(save_dir.joinpath('explicit_noise.csv'))
    # save_ket_annot_only_existing_paths(df)
    
if __name__ == '__main__':
    main(annotation_files, active_learning=False, benoit=True)

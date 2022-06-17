#%% imports
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import glob
import shutil
import sys

from ketos.data_handling import selection_table as sl
import ketos.data_handling.database_interface as dbi
from ketos.data_handling.parsing import load_audio_representation
from ketos.audio.spectrogram import MagSpectrogram
from ketos.data_handling.parsing import load_audio_representation

from pathlib import Path
import os
import librosa as lb
import soundfile as sf
import matplotlib.pyplot as plt
from librosa.display import specshow

#%% functions

def get_file_durations(annotations):
    # get all unique folders containing audio files
    audio_parent_folders = np.unique(list(map(lambda x: Path(x).parent,  
                            annots.index.get_level_values(0))))
    f_dur_tst = pd.DataFrame()
    for folder in audio_parent_folders:
        f_dur_tst = f_dur_tst.append( sl.file_duration_table(folder) )
    return f_dur_tst

# %% PIPELINE
# read ketos annotation table, normalize length, create negatives, 
# combine the two tables

annots = pd.read_csv('Daten/ket_annot_file_exists.csv')
# needs to be standardized again, because Multiindexes are not saved in csv format
annots = sl.standardize(annots)

# normlize the lengths
positives_test = sl.select(annotations=annots, 
                           length = 3.,
                           step = 0,
                           center = False)

file_durations = get_file_durations(annots)

negatives_test = sl.create_rndm_backgr_selections(annotations=annots,
                                                  files = file_durations,
                                                  length = 3.,
                                                  num = len(annots) // 100,
                                                  trim_table = True)

selections_test = positives_test.append(negatives_test, sort = False)
#%% copy files to local folder
all_files = np.unique(annots.index.get_level_values(0))
for num, file in enumerate(all_files):
    file = Path(file)
    shutil.copyfile(file, f'Daten/model_data/test/{file.stem}{file.suffix}')
    sys.stdout.write(f"\rsuccessfully copied file {num}/{len(all_files)}")
    sys.stdout.flush()
    # print(f'successfully copied file {num}/{len(all_files)}')

# %%
spctrgrm_sttngs = {'type': 'MagSpectrogram',
                    'rate': 1000,
                    'window': 0.256,
                    'step': 0.032,
                    'freq_min': 0,
                    'freq_max': 1000,
                    'window_func': 'hamming'}

# spec_cfg = load_audio_representation('spec_config.json', name="spectrogram")


dbi.create_database(output_file='Daten/model_data/test_database.h5', 
                    data_dir='Daten/model_data/test',
                    dataset_name='test',selections=selections_test,
                    audio_repres=spctrgrm_sttngs)
                              


# %%

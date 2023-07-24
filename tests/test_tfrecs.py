import os, sys
import unittest
from pathlib import Path
import pandas as pd
sys.path.insert(0, os.path.abspath("."))    

########### MODIFY SESSION SETTINGS BEFORE GLOBAL CONFIG IS IMPORTED #########
from AcoDet.create_session_file import create_session_file
create_session_file()
import json
with open('AcoDet/files/tmp_session.json', 'r') as f:
    session = json.load(f)
session['sound_files_source'] = 'tests/test_files/test_audio_files'
session['generated_annotation_source'] = 'tests/test_files/test_generated_annotations'
session['annotation_destination'] = 'tests/test_files/test_generated_annotations'
session['generated_annotations_folder'] = 'tests/test_files/test_generated_annotations'

session['reviewed_annotation_source'] = 'tests/test_files/test_generated_annotations'
session['tfrecords_destination_folder'] = 'tests/test_files/test_tfrecords'

with open('AcoDet/files/tmp_session.json', 'w') as f:
    json.dump(session, f)
##############################################################################


from AcoDet.annotate import run_annotation, filter_annots_by_thresh
from AcoDet.funcs import return_windowed_file, get_train_set_size
from AcoDet.models import GoogleMod
from AcoDet.combine_annotations import generate_final_annotations
from AcoDet.tfrec import write_tfrec_dataset
from AcoDet.train import run_training
from AcoDet import global_config as conf

class TestTFRecords(unittest.TestCase):
    def test_tfrec_creation(self):
        file = list(Path(conf.SOUND_FILES_SOURCE)
                .joinpath('BERCHOK_SAMANA_200901_4')
                .glob('**/*.wav'))[0]
        arr, t = return_windowed_file(file)
        self.assertEqual(arr.shape[1], conf.CONTEXT_WIN)
        self.assertEqual(arr.shape[0], len(t))

class TestTFRecordCreation(unittest.TestCase):
    
    def test_tfrecord_writing(self):
        write_tfrec_dataset()
        self.assertEqual((Path(conf.TFREC_DESTINATION)
                        .joinpath(f'thresh_{conf.THRESH}')
                        .joinpath('dataset_meta_train.json').exists()),
                        1, 'TFRecords files have not been generated.')
            
    def test_combined_annotation_creation(self):
        generate_final_annotations()
        self.assertEqual((Path(conf.ANNOT_DEST)
                        .joinpath(f'thresh_{conf.THRESH}')
                        .joinpath('combined_annotations.csv').exists()),
                        1, 'csv file containing combined_annotations does not exist.')

class TestDetection(unittest.TestCase):
    def test_annotation(self):
        self.time_stamp = run_annotation()
        df = pd.read_csv((Path(conf.GEN_ANNOTS_DIR).joinpath(self.time_stamp)
                       .joinpath('stats.csv')))
        self.assertEqual(df['number of predictions with thresh>0.8'][0],
                           326, 
                           'Number of predictions is not what it should be.')
        
        filter_annots_by_thresh(self.time_stamp)
        file = list(Path(conf.GEN_ANNOT_SRC).joinpath(
            self.time_stamp
            ).joinpath(f'thresh_{conf.THRESH}').glob('**/*.txt'))[0]
        df = pd.read_csv(file)
        self.assertEqual(len(df),
                         309, 
                         'Number of predictions from filtered thresholds '
                         'is incorrect.')

class TestTraining(unittest.TestCase):
    def test_model_load(self):
        model = GoogleMod(load_g_ckpt=False).model
        self.assertGreater(len(model.layers), 15)
        
    def test_tfrecord_loading(self):
        data_dir = list(Path(conf.TFREC_DESTINATION).iterdir())
        n_train, n_noise = get_train_set_size(data_dir)
        self.assertEqual(n_train, 200)
        self.assertEqual(n_noise, 19)

if __name__ == '__main__':
    unittest.main()
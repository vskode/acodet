import os, sys
import unittest
import tensorflow as tf
from pathlib import Path
import pandas as pd
sys.path.insert(0, os.path.abspath("."))
file = 'Daten/Tim/2020-11-17/channelA_2020-11-17_00-00-04.wav'
from hbdet import global_config as conf
from hbdet.annotate import run_annotation, filter_annots_by_thresh
from hbdet.funcs import return_windowed_file, get_train_set_size
from hbdet.models import GoogleMod
from hbdet.combine_annotations import generate_final_annotations
from hbdet.tfrec import write_tfrec_dataset
from hbdet.train import run_training

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
        time_stamp = run_annotation()
        df = pd.read_csv((Path(conf.GEN_ANNOTS_DIR).joinpath(time_stamp)
                       .joinpath('stats.csv')))
        self.assertGreater(df['number of predictions with thresh>0.8'][0],
                           100, 
                           'Number of predictions is far lower than it should be.')
        
    def test_filter_threshs(self):
        filter_annots_by_thresh()
        file = list(Path(conf.GEN_ANNOT_SRC)
                .joinpath('thresh_0.5/BERCHOK_SAMANA_200901_4')
                .glob('**/*.txt'))[0]
        self.assertEqual((Path(conf.GEN_ANNOT_SRC)
                        .joinpath(f'thresh_{conf.THRESH}')
                        .joinpath(Path(file).relative_to(Path(conf.GEN_ANNOT_SRC)
                                                   .joinpath('thresh_0.5'))).parent.exists()),
                        1, 'Directory is not created for filtered thresholds.')

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
import unittest
from pathlib import Path
import tensorflow as tf
class TestPaths(unittest.TestCase):
    def test_tfrecords_exist(self):
        tfrecord_files = list(Path('Daten/tfrecords').glob('*tfrec'))
        self.assertGreater(len(tfrecord_files), 0, 
                           'Could not find TFRecord files.')
        
    def test_gpu_connected(self):
        gpus = tf.config.experimental.list_physical_devices('GPU')
        self.assertGreater(len(gpus), 0, 
                           'No GPU device connected. This might mean slow trainings.')

if __name__ == '__main__':
    unittest.main()
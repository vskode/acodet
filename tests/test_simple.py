import unittest
from pathlib import Path

class TestPaths(unittest.TestCase):
    def test_tfrecords_exist(self):
        tfrecord_files = list(Path('Daten/tfrecords').glob('*tfrec'))
        self.assertGreater(len(tfrecord_files), 0, 
                           'Could not find TFRecord files.')

if __name__ == '__main__':
    unittest.main()
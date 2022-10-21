from hbdet.hbdet.funcs import return_windowed_file
import unittest
from pathlib import Path
file = 'Daten/Tim/2020-11-17/channelA_2020-11-17_00-00-04.wav'

class TestPaths(unittest.TestCase):
    def test_tfrec_creation(self):
        arr, t = return_windowed_file(file, sr=10000, cntxt_wn_sz=39124)
        self.assertEqual(arr.shape[1], 39124)
        self.assertEqual(arr.shape[0], len(t))

if __name__ == '__main__':
    unittest.main()
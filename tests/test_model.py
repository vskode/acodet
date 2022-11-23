import unittest
from hbdet.hbdet.models import GoogleMod

class TestModel(unittest.TestCase):
    def test_model_load(self):
        model = GoogleMod({'load_g_ckpt':False}).model
        self.assertGreater(len(model.layers), 15)
        
        
if __name__ == '__main__':
    unittest.main()
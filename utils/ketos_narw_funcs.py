import pandas as pd
import ketos.neural_networks.dev_utils.detection as det
from ketos.audio.spectrogram import MagSpectrogram
from ketos.audio.audio_loader import AudioFrameLoader
from ketos.neural_networks.resnet import ResNetInterface
from ketos.neural_networks.dev_utils.detection import process, save_detections
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import librosa as lb
from utils.funcs import *


def load_ketos_model():
    model_path = 'models/ketos_narw_classifier/narw.kt'
    model, _ = ResNetInterface.load_model_file(model_file=model_path,
                                                overwrite = True,
                    new_model_folder=Path(model_path).parent.joinpath('tmp'),
                                        load_audio_repr=True)
    return model

def get_2d_array(signal, fft_window_length, **params):
    S = np.abs(lb.stft(signal, win_length = fft_window_length))
    S_dB = lb.amplitude_to_db(S, ref=np.max)
    return S_dB

        
class NarwMod():
    def __init__(self, params):
        self.params = params
        self.model = load_ketos_model()
        self.model.model.compile('Adam', 'BinaryCrossentropy')
        self.params['fmin'] = 0
        self.params['fmax'] = 5000
        
    def load_data(self, file, annots, y_test, y_noise, x_test, x_noise):
        self.file = file
        self.annots = annots
        self.x_test = x_test
        self.x_noise = x_noise
        self.y_test = y_test
        self.y_noise = y_noise
        
    def pred(self, noise = False):
        if noise:
            self.x_noise_ar = []
            for ar in self.x_noise:
                self.x_noise_ar.append( get_2d_array(ar, **self.params) )
            self.x_noise_ar = np.array(self.x_noise_ar)

            self.preds_noise =  self.model.run_on_batch(self.x_noise_ar)[1]
            return self.preds_noise
        else:
            self.x_test_ar = []
            for ar in self.x_test:
                self.x_test_ar.append( get_2d_array(ar, **self.params) )
            self.x_test_ar = np.array(self.x_test_ar)
            
            self.preds_call =  self.model.run_on_batch(self.x_test_ar)[1]
            return self.preds_call
        

    
    def eval(self, noise = False):
        if noise:
            return self.model.model.evaluate(np.expand_dims(self.x_noise_ar, -1), 
                                             self.y_noise)
        else:
            return self.model.model.evaluate(np.expand_dims(self.x_test_ar, -1), 
                                             self.y_test)
    
    def spec(self, num, noise = False):
        if noise:
            prediction = self.preds_noise[num]
            start = self.annots['end'].iloc[-1] + \
                    self.params['cntxt_wn_sz']*num / self.params['sr']
            spec_data = self.x_noise_ar[num]
        else:
            prediction = self.preds_call[num]
            start = self.annots['start'].iloc[num]
            spec_data = self.x_test_ar[num]
        
        
        plot_spec(spec_data, self.file, prediction = prediction,
                    start = start, noise = noise, 
                    mod_name = type(self).__name__, **self.params)
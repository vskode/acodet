#%%
from email.mime import audio
import torch
import torch.nn as nn
from funcs import *
import numpy as np
import os
import sys
sys.path.insert(1, '/home/vincent/Code/MA/models/Benoit_orca_spot/ORCA-SPOT/orca_spot')

from collections import OrderedDict
from models.classifier import Classifier
from data.audiodataset import StridedAudioDataset
from models.residual_encoder import ResidualEncoder as Encoder

#%%


#%%
class HumpSpotHelper():
    def __init__(self, params):
        self.model = self.load_HUMP_SPOT()
        self.params = params
        
    def get_audio_len(self):
        last_annot = self.annots['start'].iloc[-1]
        first_annot  = self.annots['start'].iloc[0]
        sr = self.params['sr']
        return first_annot + last_annot + self.params['cntxt_wn_hop']/sr

    def make_data_loader(self, x_test, x_noise, offset, 
                                  sequence_len, cntxt_wn_hop, sr, 
                                  fft_window_length, fft_hop, 
                                  n_freq_bins, freq_cmpr, 
                                  fmin, fmax, **_):
        # sequence length = context windows length
        audio_len = self.get_audio_len()
        dataset = StridedAudioDataset(
            self.file.strip(),
            offset = offset,
            sequence_len=sequence_len,
            audio_len = audio_len,
            hop=cntxt_wn_hop,
            sr=sr,
            fft_size=fft_window_length,
            fft_hop=fft_hop,
            n_freq_bins=n_freq_bins,
            freq_compression=freq_cmpr,
            f_min=fmin,
            f_max=fmax,
            min_max_normalize=True,
            annotations = self.annots,
            x_test = x_test, 
            x_noise = x_noise, 
        )
        
        batch_size = max(len(x_test), len(x_noise))
        signal_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=4,
            pin_memory=True,
        )
        return signal_loader, dataset

    def get_preds(self):
        with torch.no_grad():
            for file in self.data_loader:
                out = self.model(file)
                pred = torch.nn.functional.softmax(out, dim=1).numpy()[:, 0]
        return pred

    def bin_cross_entropy(self, noise = False):
        bce = nn.BCELoss()
        if noise:
            return bce(torch.tensor(self.preds_noise), torch.tensor(self.y_noise))
        else:
            return bce(torch.tensor(self.preds_call), torch.tensor(self.y_test))
        
    @staticmethod
    def load_HUMP_SPOT():
        if not 'Daten' in os.listdir():
            os.chdir('../../../..')
        model_path = 'models/Benoit_orca_spot/20200830_HB_model.pk'

        model_dict = torch.load(model_path)
        dataOpts = model_dict["dataOpts"]
        encoder = Encoder(model_dict["encoderOpts"])
        encoder.load_state_dict(model_dict["encoderState"])
        classifier = Classifier(model_dict["classifierOpts"])
        classifier.load_state_dict(model_dict["classifierState"])
        model = nn.Sequential(
            OrderedDict([("encoder", encoder), ("classifier", classifier)])
        )
        return model
    
    
class BenoitMod(HumpSpotHelper):

    def load_data(self, file, annots, y_test, y_noise,
                  x_test = None, x_noise = None):
        self.file = file
        self.annots = annots
        self.audio_len = self.get_audio_len()
        offset = self.annots['start'].iloc[0]
        self.y_test = y_test
        self.y_noise = y_noise
        self.data_loader, self.dataset = self.make_data_loader(x_test, x_noise,
                                                          offset = offset, 
                                                        **self.params)
    def spec(self, num, noise = False):
        if noise:
            self.dataset.get_noise = True
        else:
            self.dataset.get_noise = False
        with torch.no_grad():
            for file in self.data_loader:
                file = file
        spec_data = file[num, 0].T
        if noise:
            prediction = self.preds_noise[num]
            start = self.annots['end'].iloc[-1] + \
                    self.params['cntxt_wn_sz']*num / self.params['sr']
        else:
            prediction = self.preds_call[num]
            start = self.annots['start'].iloc[num]
        plot_spec(spec_data.numpy(), self.file, prediction = prediction,
                start = start, noise = noise, 
                mod_name = type(self).__name__, **self.params)
        
    def pred(self, noise = False):
        if noise:
            self.dataset.get_noise = True
            self.preds_noise = self.get_preds()
            return self.preds_noise
        else:
            self.dataset.get_noise = False
            self.preds_call = self.get_preds()
            return self.preds_call
    
    def eval(self, noise = False):
        return self.bin_cross_entropy(noise = noise).item()
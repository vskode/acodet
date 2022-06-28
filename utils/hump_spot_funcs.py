#%%
from email.mime import audio
import torch
import torch.nn as nn
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

    def create_data_loader_benoit(self, offset, sequence_len,
                                cntxt_wn_hop, sr, fft_window_length, 
                                fft_hop, n_freq_bins, freq_cmpr, 
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
            annotations = self.annots
        )
        
        batch_size = len(self.annots)
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=4,
            pin_memory=True,
        )
        return data_loader

    def get_preds(self):
        with torch.no_grad():
            for file in self.data_loader:
                out = self.model(file)
                pred = torch.nn.functional.softmax(out, dim=1).numpy()[:, 0]
        return pred

    def bin_cross_entropy(self):
        bce = nn.BCELoss()
        return bce(torch.tensor(self.preds), torch.tensor(self.labels))
        
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
    
    
class HumpSpot(HumpSpotHelper):

    def load_data(self, file, annots):
        self.file = file
        self.annots = annots
        self.audio_len = self.get_audio_len()
        offset = self.annots['start'].iloc[0]
        self.data_loader = self.create_data_loader_benoit(offset = offset, 
                                                        **self.params)
        
    def pred(self, noise = False):
        self.preds = self.get_preds()
        self.labels = np.ones(len(self.preds), dtype='float32')
        return self.preds
    
    def eval(self, noise = False):
        return self.bin_cross_entropy().item()
#%%
import torch
import torch.nn as nn
import os
import sys
sys.path.insert(1, '/home/vincent/Code/MA/models/Benoit_orca_spot/ORCA-SPOT/orca_spot')

from collections import OrderedDict
from models.classifier import Classifier
from data.audiodataset import StridedAudioDataset
from models.residual_encoder import ResidualEncoder as Encoder

#%%
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


def create_data_loader_benoit(file, offset, sequence_len, cntxt_wn_hop,
                              sr, fft_window_length, fft_hop, 
                              n_freq_bins, freq_cmpr, 
                              fmin, fmax, **_):
    dataset = StridedAudioDataset(
        file.strip(),
        offset = offset,
        sequence_len=sequence_len,
        hop=cntxt_wn_hop,
        sr=sr,
        fft_size=fft_window_length,
        fft_hop=fft_hop,
        n_freq_bins=n_freq_bins,
        freq_compression=freq_cmpr,
        f_min=fmin,
        f_max=fmax,
        min_max_normalize=True
    )
    
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=int(sequence_len//cntxt_wn_hop),
        num_workers=4,
        pin_memory=True,
    )
    return data_loader

def get_HOMP_SPOT_preds(model, data_loader):
    model = load_HUMP_SPOT()
    with torch.no_grad():
        for file in data_loader:
            out = model(file)
            pred = torch.nn.functional.softmax(out, dim=1).numpy()[:,0]
    return pred

import timm
from types import SimpleNamespace
# from nnAudio.features.mel import MelSpectrogram
from .humpback_model_dir.torch_PCEN import PCEN as torch_PCEN

import torch
import torch.nn as nn
import torchaudio as ta

import numpy as np

from . import global_config as conf

class TorchModel(nn.Module):
    
    def __init__(self, effnet='b3', num_classes=2, **kwargs):
        super(TorchModel, self).__init__()
        # DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.backbone = timm.create_model(
            f"efficientnet_{effnet}",
            pretrained=False,
            num_classes=num_classes,
            in_chans=1
        )
        # self.backbone.to(DEVICE)
        
        cfg = {
            'n_classes': num_classes,
            'sample_rate': conf.SR,
            'n_fft': conf.STFT_FRAME_LEN,
            'window_size': conf.STFT_FRAME_LEN,
            'hop_length': conf.FFT_HOP,
            'fmin': 0,
            'fmax': conf.SR / 2,
            'n_mels': 64,
            'power': 2,
            'mel_normalized': False,
            'top_db': 80.0,
        }
        cfg = SimpleNamespace(**cfg)
        
        self.front_end = MelSpecTorch(cfg)
        self.pcen = torch_PCEN(
            num_channels=1,
            alpha=0.98,
            smooth_coef=0.025,
            delta=2.0,
            root=2.0,
            floor=1e-6,
            trainable=True
        )
        

        # if init_backbone:
        #     # Initialize pre-trained CNN
        #     # Input and output layers are automatically adjusted
        #     self.backbone = timm.create_model(
        #         cfg.backbone,
        #         pretrained=cfg.pretrained,
        #         num_classes=cfg.n_classes,
        #         in_chans=cfg.in_chans,
        #     )

        # Spectrogram augmentation
        # Chooses one of MaskFrequency and MaskTime with a probability of cfg.specaug_prob
        # These functions mask a certain segment (frequency or time) in each sample of the batch.
        # self.specaug = Compose(
        #     [OneOf(
        #         [MaskFrequency(p=1),
        #          MaskTime(p=1)],
        #         p=cfg.specaug_prob),
        #     ])

        # # Mixup augmentation
        # # Mixes two random samples in the batch with a random mixing ratio
        # # Not only changes the spectrogram, but also turn the 1Hot label vector into probabilities
        # self.mixup = Mixup(cfg.mixup_prob)
        

    def forward(self, x, y=None):
        # (bs, channel, time)
        # Add channel dimension for CNN input
        x = x[:, None, :]

        # (bs, channel, mel, time)
        x = self.front_end.wav2timefreq(x)
        # x = x.unsqueeze(1)
        x = self.pcen(x) # TODO why all nans?

        # if self.cfg.minmax_norm:
        #     x = (x - self.cfg.min) / (self.cfg.max - self.cfg.min)

        # if self.training:
        #     # Mixup augmentation
        #     if self.cfg.mixup:
        #         # Mixup returns adapted spectrogram and label probabilities
        #         x, y = self.mixup(x, y)
        #     # Spectrogram augmentation, e.g. MaskFrequency/MaskTime
        #     if self.cfg.specaug:
        #         x = self.specaug(x, None)

        # Forward pass through the CNN backbone
        logits = self.backbone(x)

        # if self.training:
        #     # During training has to also return the label in case they got modified during mixup
        #     return logits, y
        # else:
        return logits
    
    
class MelSpecTorch(nn.Module):
    def __init__(self, cfg):
        """
        Pytorch network class containing the transformation from waveform to
        mel spectrogram, as well as the forward pass through a CNN backbone.

        Data augmentation like mixup or masked frequency or time can also be
        applied here.

        Parameters
        ----------
        cfg: SimpleNameSpace containing all configurations
        init_backbone: bool (Default=True). Whether to download and initialize the backbone.
                       Not always necessary when debugging.
        """
        super().__init__()

        self.cfg = cfg
        self.n_classes = cfg.n_classes

        # Initializes the transformation from waveform to mel spectrogram
        self.mel_spec = ta.transforms.MelSpectrogram(
            sample_rate=cfg.sample_rate,
            n_fft=cfg.n_fft,
            win_length=cfg.window_size,
            hop_length=cfg.hop_length,
            f_min=cfg.fmin,
            f_max=cfg.fmax,
            n_mels=cfg.n_mels,
            power=cfg.power,
            normalized=cfg.mel_normalized,
        )
        # self.mel_spec = MelSpectrogram(
        #     sr=cfg.sample_rate,
        #     n_fft=cfg.n_fft,
        #     n_mels=cfg.n_mels,
        #     trainable_mel=True,
        #     trainable_STFT=True
        # )
        
        # the above melspec doesn't seem to work well with batching
        # cause this prints false, whereas the ta one prints true
        # seq=[]
        # for xx in x:
        #     seq.append(self.front_end.mel_spec(xx).detach().cpu())
        # seq = torch.Tensor(np.array(seq))
        # bat = self.front_end.mel_spec(x).detach().cpu()
        # torch.allclose(seq, bat, atol=1e-7)

        self.amplitude_to_db = ta.transforms.AmplitudeToDB(
            top_db=cfg.top_db
            )
        self.wav2timefreq = torch.nn.Sequential(
            self.mel_spec,
            self.amplitude_to_db
            )
        


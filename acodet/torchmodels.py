import timm
from types import SimpleNamespace
from .humpback_model_dir.torch_PCEN import PCEN as torch_PCEN

import torch.nn as nn
import torchaudio as ta
import torch


from . import global_config as conf
from .torch_augment import TorchAugment

class TorchModel(nn.Module):
    
    def __init__(self, effnet='b3', num_classes=1, **kwargs):
        super(TorchModel, self).__init__()
        # DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.backbone = timm.create_model(
            f"efficientnet_{effnet}",
            pretrained=False,
            num_classes=num_classes,
            in_chans=1
        )
        
        cfg = {
            'n_classes': num_classes,
            'sample_rate': conf.SR,
            'n_fft': conf.STFT_FRAME_LEN,
            'window_size': conf.STFT_FRAME_LEN,
            'hop_length': conf.FFT_HOP,
            'fmin': 50,
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
        
        self.augment = TorchAugment(64, conf.N_TIME_BINS)


    def forward(
        self, x, y=None, noise=None, 
        path=None, start=None, training=False,
        plot=False
        ):
        # (bs, channel, time)        
        x = x[:, None, :]

        x_spec = self.front_end.mel_spec(x)

        if training:
            noise_spec = self.front_end.mel_spec(noise)
            if plot and 1 in y:
                self.augment.plot_augmented(x_spec, y, paths=path, starts=start)
            x_spec = self.augment(x_spec, noise_spec, y, plot=plot)
            if plot and 1 in y:
                self.augment.plot_augmented(x_spec, y, paths=path, starts=start, augmented=True)

        x_processed = self.pcen(x_spec)
        if plot and 1 in y:
            self.augment.plot_augmented(x_spec, y, paths=path, starts=start, augmented='with_pcen')
        
        logits = []
        if len(x_processed.shape) == 3:
            x_processed = x_processed.unsqueeze(1)
            
        for idx in range(0, x_processed.shape[0] // conf.BATCH_SIZE):
            # Forward pass through the CNN backbone
            logits.append(self.backbone(x_processed[
                idx*conf.BATCH_SIZE : (idx+1)*conf.BATCH_SIZE
                ]))
        
        logits = torch.vstack(logits)
        
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
        # using the torchaudio MelSpectrogram implementation
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
        
        ##  nnAudio implementation - would have the advantage that the
        # spectrogram creation is learnable, but there seem to be issues
        # that I was not able to fix yet.

        ## If you want to try it you'll need the nnAudio package and load
        # the MelSpectrogram class
        # from nnAudio.features.mel import MelSpectrogram
        
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

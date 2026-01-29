import torch
import numpy as np
import torch.nn as nn
import torchaudio.transforms as T

from acodet import global_config as conf


class TimeShift(nn.Module):
    def __init__(self, max_shift_ratio=0.5, p=0.5):
        super().__init__()
        self.max_shift_ratio = max_shift_ratio
        self.p = p

    def forward(self, x):
        """
        x shape: (Batch, Freq, Time)
        """            
        B, F, T = x.shape
        max_shift = int(T * self.max_shift_ratio)
        
        # 1. Generate a random shift for the whole batch 
        shift = torch.randint(low=0, high=max_shift, size=(1,)).item()
        
        # 2. Roll along the Time axis
        return torch.roll(x, shifts=shift, dims=-1)

class NoiseAugment(nn.Module):
    def __init__(self, alpha=0.4, p=0.5):
        super().__init__()
        self.alpha = alpha
        self.p = p

    def forward(self, calls, noise):
        """
        Implements the specific scaling logic
        mix = call * train_alpha + noise * noise_alpha
        """
        # in case we're on the last batch and it has less elements
        nr = len(calls)

        flat_calls = calls[:nr].flatten(1)
        flat_noise = noise[:nr].flatten(1)
        
        # Get values (B,)
        max_call = flat_calls.max(dim=1).values
        max_noise = flat_noise.max(dim=1).values

        # 2. Reshape for Broadcasting
        # Dynamically create a shape like (B, 1, 1) or (B, 1, 1, 1)
        # depending on input dimensions so multiplication works.
        target_shape = [-1] + [1] * (calls.ndim - 1)
        
        max_call = max_call.view(target_shape)
        max_noise = max_noise.view(target_shape)

        # # get max call and max noise to specify how to overlay them
        # max_call = calls.view(calls.size(0), -1).max(dim=1).values
        # max_noise = noise.view(noise.size(0), -1).max(dim=1).values

        # max_call = max_call.view(-1, 1, 1)
        # max_noise = max_noise.view(-1, 1, 1)

        noise_alpha = self.alpha * max_noise
        train_alpha = (1 - self.alpha) * max_call

        return (calls * train_alpha) + (noise[:nr] * noise_alpha)

class TorchAugment(nn.Module):
    
    def __init__(self, n_mels, n_tbins):
        super().__init__()
        self.do_timeshift = conf.TIME_AUGS
        self.do_specaug = conf.MIXUP_AUGS
        self.do_noiseaug = conf.SPEC_AUG
        self.specaug = T.SpecAugment(
            # nr of time/freq masks
            n_time_masks=1,
            n_freq_masks=1,
            # max width/height of masks
            time_mask_param=int(n_tbins / 4), 
            freq_mask_param=int(n_mels / 4),
            # Independent and Identically Distributed (i.i.d.) mask 
            # for every individual sample in the batch. If false
            # all the augmentationas are identical - which we don't want
            iid_masks=True,
            )
        
        # 2. Time Shift (CropAndFill)
        self.timeshift = TimeShift(max_shift_ratio=0.5, p=0.5)
        
        # 3. Mixup
        self.noiseaug = NoiseAugment(alpha=0.4, p=0.5)
        
    def forward(self, x, noise=None):
        # x shape comes in as: (Batch, Channels, Freq, Time)
        # Augmentations expect: (Batch, Freq, Time)
        if x.dim() == 4:
            x = x.squeeze(1)

        if self.do_timeshift:
            x = self.timeshift(x)
        
        if self.do_specaug:
            x = self.specaug(x)
            
        if self.do_noiseaug:
            x = self.noiseaug(x, noise)
        
        # Restore Channel dimension for CNN: (Batch, 1, Freq, Time)
        return x.unsqueeze(1)
        
    def plot_augmented(self, x, y=None, paths=None, starts=None):
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(figsize=[12, 12], ncols=4, nrows=4)
        axes = axes.reshape(16)
        for idx, ax in enumerate(axes):
            ax.imshow(x[idx].detach().cpu().squeeze(), origin='lower')
            if y is not None and paths is not None and starts is not None:
                from pathlib import Path
                ax.set_title(
                    f'{Path(paths[idx]).stem} @ {starts[idx]:.0f}, {np.argmax(y[idx].detach().cpu())}'
                    )
        fig.tight_layout()
        fig.suptitle('Sample of augmented spectrograms')
        fig.savefig('augmentated spectrograms.png')
        plt.close(fig)
        
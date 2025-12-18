import random
import math

import torch
import torch.nn as nn
from torch.distributions import Beta
from torchaudio.transforms import FrequencyMasking, TimeMasking
import numpy as np
import colorednoise as cn
from tqdm import tqdm


def get_max_amplitude_window_index(path: str, waveform=None, samplerate=None, window_length_sec=5, scan_param=50,
                                   verbose=True):
    """
    Returns index of waveform that starts the window of length window_length_sec*samplerate,
    with the highest summed amplitude.
    only scans at certain scan intervals, to speed up the calculation

    Parameters
    ----------
    path: str, path to data, as in torchaudio.load().
    waveform: torch.tensor of a single waveform.
    samplerate: samplerate of the waveform.
    window_length_sec: window length to calculate sum over absolute amplitudes.
    scan_param: samplerate should be divisible by scan_param.
    verbose: bool, to print return index in seconds.

    Returns
    -------
    max_index (int): start index of window with max amplitudes

    """
    if waveform is None:
        waveform, samplerate = torchaudio.load(path)

    waveform_length = waveform[0].numpy().shape[0]
    window_length = math.floor(window_length_sec * samplerate)

    if window_length >= waveform_length:
        return 0

    # divide available waveform length by scan_param, to construct scan array
    scan_length = math.floor((waveform_length - window_length) / scan_param)

    max_sum = 0
    max_index = 0

    # in every scan interval: calculate sum over window and save max
    for x in range(scan_length):
        tmp = np.sum(abs(waveform[0].numpy()[x * scan_param:x * scan_param + window_length]))
        if tmp > max_sum:
            max_sum = tmp
            max_index = x * scan_param

    if verbose:
        print('window starts at:', max_index / samplerate, 'seconds')
    return max_index


def get_min_max(cfg, dm, model):
    """
    Helper function to obtain the minimum and maximum
    spectrogram values necessary for min max normalization.

    Parameters
    ----------
    cfg: SimpleNameSpace containing all configurations
    dm: Lightning Datamodule class, e.g. as defined in data.py.
    model: Pytorch network class, e.g. SpectrogramCNN() defined in net.py.

    Returns
    -------
    total_min: float
    total_max: float

    """
    dm = dm(cfg=cfg)
    model = model(cfg, init_backbone=False)
    dm.setup()
    dm.train.mode = 'val'
    total_max = -1e3
    total_min = 1e3
    print('Gather min max statistics:')
    for batch in tqdm(dm.train_dataloader()):
        spec = model.wav2timefreq(batch['wave'][:, None, :])
        if spec.max() > total_max:
            total_max = spec.max()
        if spec.min() < total_min:
            total_min = spec.min()
    return total_min, total_max


def min_max_norm(x, min_val=-39.4655, max_val=53.6203):
    # Min max normalization of an array x
    return (x - min_val) / (max_val - min_val)


def batch_to_device(batch, device):
    # Helper function to move a Pytorch batch to a specific device,
    # e.g. a GPU.
    batch_dict = {key: batch[key].to(device) for key in batch}
    return batch_dict


def get_state_dict(sd_fp):
    """
    Helper function to load and preprocess Pytorch state dicts, which
    contains for instance model weights.
    When using pretrained models from other libraries key mismatches
    can happen which requires the replacement of certain keys.

    Parameters
    ----------
    sd_fp: str. Filepath to a state dict.

    Returns
    -------
    sd: Torch tensor containing model weights.
    """
    sd = torch.load(sd_fp, map_location="cpu")['state_dict']
    # When saving model weights during training, the prefix 'model.'
    # is appended which leads to errors during loading and has to be removed.
    sd = {k.replace("model.", ""): v for k, v in sd.items()}
    sd.pop("loss_fn.weight")
    return sd


class Compose:
    def __init__(self, transforms: list):
        """
        Base class to chain data augmentation methods.

        Based and modified from
        https://github.com/albumentations-team/albumentations/blob/master/albumentations/core/composition.py
        and
        https://github.com/tattaka/birdclef-2021/blob/7fed19356f8e4cc499ed29dbcdd7b8e960de6cef/src/stage1/main.py
        under MIT license.

        Parameters
        ----------
        transforms: list of augmentation methods, e.g. [NoiseInjection(), GaussianNoise()]
        """
        self.transforms = transforms

    def __call__(self, y: np.ndarray, sr):
        for trns in self.transforms:
            y = trns(y, sr)
        return y


class OneOf(Compose):
    def __init__(self, transforms, p=0.5):
        """
        Chooses one transformation/augmentation method from a provided
        list with probability p.

        Based and modified from
        https://github.com/albumentations-team/albumentations/blob/master/albumentations/core/composition.py
        and
        https://github.com/tattaka/birdclef-2021/blob/7fed19356f8e4cc499ed29dbcdd7b8e960de6cef/src/stage1/main.py
        under MIT license.

        Parameters
        ----------
        transforms: list of augmentation methods, e.g. [NoiseInjection(), GaussianNoise()].
        p: float. Probability to apply augmentation.
        """
        super().__init__(transforms)
        self.p = p
        transforms_ps = [t.p for t in transforms]
        s = sum(transforms_ps)
        self.transforms_ps = [t / s for t in transforms_ps]

    def __call__(self, y: np.ndarray, sr):
        data = y
        if self.transforms_ps and (random.random() < self.p):
            random_state = np.random.RandomState(random.randint(0, 2 ** 32 - 1))
            t = random_state.choice(self.transforms, p=self.transforms_ps)
            data = t(y, sr)
        return data


class AudioTransform:
    def __init__(self, always_apply=False, p=0.5):
        """
        Base class for data augmentations on waveforms or spectrogram's.

        Based and modified from
        https://github.com/tattaka/birdclef-2021/blob/7fed19356f8e4cc499ed29dbcdd7b8e960de6cef/src/stage1/main.py
        under MIT license.

        Parameters
        ----------
        always_apply: bool. Can be turned to True for debugging purposes.
        p: float between 0-1. Probability to apply the augmentation.
        """
        self.always_apply = always_apply
        self.p = p

    def __call__(self, y: np.ndarray, sr):
        if self.always_apply:
            return self.apply(y, sr=sr)
        else:
            if np.random.rand() < self.p:
                return self.apply(y, sr=sr)
            else:
                return y

    def apply(self, y: np.ndarray, **params):
        raise NotImplementedError


class NoiseInjection(AudioTransform):
    def __init__(self, always_apply=False, p=0.5, max_noise_level=0.5):
        """
        Noise injection augmentation to be applied on waveforms.

        Based and modified from
        https://github.com/tattaka/birdclef-2021/blob/7fed19356f8e4cc499ed29dbcdd7b8e960de6cef/src/stage1/main.py
        under MIT license.

        Parameters
        ----------
        always_apply: bool. Can be turned to True for debugging purposes.
        p: float between 0-1. Probability to apply the augmentation.
        max_noise_level: float. Maximum noise level of the injection.
        """
        super().__init__(always_apply, p)

        self.noise_level = (0.0, max_noise_level)

    def apply(self, y: np.ndarray, **params):
        noise_level = torch.distributions.Uniform(*self.noise_level).sample([1])

        noise = torch.randn(len(y))
        augmented = y + noise * noise_level
        return augmented


class GaussianNoise(AudioTransform):
    def __init__(self, always_apply=False, p=0.5, min_snr=5, max_snr=20):
        """
        Gaussian noise augmentation to be applied on waveforms.

        Based and modified from
        https://github.com/tattaka/birdclef-2021/blob/7fed19356f8e4cc499ed29dbcdd7b8e960de6cef/src/stage1/main.py
        under MIT license.

        Parameters
        ----------
        always_apply: bool. Can be turned to True for debugging purposes.
        p: float between 0-1. Probability to apply the augmentation.
        min_snr: int. Minimum signal-to-noise ratio.
        max_snr: int. Maximum signal-to-noise ratio.
        """
        super().__init__(always_apply, p)

        self.min_snr = min_snr
        self.max_snr = max_snr

    def apply(self, y: np.ndarray, **params):
        snr = torch.distributions.Uniform(self.min_snr, self.max_snr).sample([1])
        a_signal = torch.sqrt(y ** 2).max()
        a_noise = a_signal / (10 ** (snr / 20))

        white_noise = torch.randn(len(y))
        a_white = torch.sqrt(white_noise ** 2).max()
        augmented = y + white_noise * 1 / a_white * a_noise
        return augmented


class PinkNoise(AudioTransform):
    def __init__(self, always_apply=False, p=0.5, min_snr=5, max_snr=20):
        """
        Pink noise augmentation to be applied on waveforms.

        Based and modified from
        https://github.com/tattaka/birdclef-2021/blob/7fed19356f8e4cc499ed29dbcdd7b8e960de6cef/src/stage1/main.py
        under MIT license.

        Parameters
        ----------
        always_apply: bool. Can be turned to True for debugging purposes.
        p: float between 0-1. Probability to apply the augmentation.
        min_snr: int. Minimum signal-to-noise ratio.
        max_snr: int. Maximum signal-to-noise ratio.
        """
        super().__init__(always_apply, p)

        self.min_snr = min_snr
        self.max_snr = max_snr

    def apply(self, y: np.ndarray, **params):
        snr = torch.distributions.Uniform(self.min_snr, self.max_snr).sample([1])
        a_signal = torch.sqrt(y ** 2).max()
        a_noise = a_signal / (10 ** (snr / 20))

        pink_noise = torch.from_numpy(cn.powerlaw_psd_gaussian(1, len(y)))
        a_pink = torch.sqrt(pink_noise ** 2).max()
        augmented = y + (pink_noise * 1 / a_pink * a_noise).type(y.dtype)
        return augmented


class MaskFrequency(AudioTransform):
    def __init__(self, always_apply=False, p=0.5, freq_mask_param=40):
        """
        Masks a random frequency range (freq_mask_param) in individual
        samples of a batch (iid_masks=True).

        Parameters
        ----------
        always_apply: bool. Can be turned to True for debugging purposes.
        p: float between 0-1. Probability to apply the augmentation.
        freq_mask_param: int. Maximum frequency range to be masked in a sample.
        """
        super().__init__(always_apply, p)
        self.masking = FrequencyMasking(freq_mask_param=freq_mask_param, iid_masks=True)

    def apply(self, y: np.ndarray, **params):
        return self.masking(y)


class MaskTime(AudioTransform):
    def __init__(self, always_apply=False, p=0.5, freq_mask_param=40):
        """
        Masks a random time range (freq_mask_param) in individual
        samples of a batch (iid_masks=True).

        Parameters
        ----------
        always_apply: bool. Can be turned to True for debugging purposes.
        p: float between 0-1. Probability to apply the augmentation.
        freq_mask_param: int. Maximum time range to be masked in a sample.
        """
        super().__init__(always_apply, p)
        self.masking = TimeMasking(time_mask_param=freq_mask_param, iid_masks=True)

    def apply(self, y: np.ndarray, **params):
        return self.masking(y)


class Mixup(nn.Module):
    def __init__(self, mix_beta):
        """
        Performs Mixup augmentation on spectrograms and 1hot labels.

        Based and modified from
        https://github.com/ChristofHenkel/kaggle-birdclef2021-2nd-place/blob/26438069466242e9154aacb9818926dba7ddc7f0/models/model_utils.py#L35
        under MIT license.

        A beta distribution is used to draw mixup probabilities.

        Parameters
        ----------
        mix_beta: float. Parameter to initialize a symmetric beta distribution.
        """
        super(Mixup, self).__init__()
        self.beta_distribution = Beta(mix_beta, mix_beta)

    def forward(self, X, Y, weight=None):
        bs = X.shape[0]
        n_dims = len(X.shape)
        perm = torch.randperm(bs)
        coeffs = self.beta_distribution.rsample(torch.Size((bs,))).to(X.device)
        if n_dims == 2:
            X = coeffs.view(-1, 1) * X + (1 - coeffs.view(-1, 1)) * X[perm]
        elif n_dims == 3:
            X = coeffs.view(-1, 1, 1) * X + (1 - coeffs.view(-1, 1, 1)) * X[perm]
        else:
            X = coeffs.view(-1, 1, 1, 1) * X + (1 - coeffs.view(-1, 1, 1, 1)) * X[perm]

        Y = coeffs.view(-1, 1) * Y + (1 - coeffs.view(-1, 1)) * Y[perm]

        if weight is None:
            return X, Y
        else:
            weight = coeffs.view(-1) * weight + (1 - coeffs.view(-1)) * weight[perm]
            return X, Y, weight
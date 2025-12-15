import numpy as np
import pandas as pd
import torch
import librosa
from torch.utils.data import DataLoader, Dataset
from torch.nn.functional import one_hot
import torchaudio as ta

# from .utils import Compose, OneOf, NoiseInjection, GaussianNoise, PinkNoise, get_max_amplitude_window_index
from acodet import global_config as conf

def collate_fn(batch):
    # Helper function to collate individual samples into batches
    return list({
        'wave': torch.stack([x['wave'] for x in batch]),
        'labels': torch.stack([x['labels'] for x in batch])
    }.values())


class AudioDataset(Dataset):
    def __init__(self, df, mode: str = 'train'):
        """
        Custom pytorch Dataset class, which loads a single sample and applies pre-processing steps.
        Pre-processing includes initialization of the 1Hot label vector,
        additional zero-padding when waveform is too short, as well as data augmentation in form of
        either Noise injections, Gaussian noise or pink noise, respectively.

        Parameters
        ----------
        df: Pandas dataframe, containing the path to the .wav files as well as the label.
        cfg: SimpleNameSpace containing all configurations
        mode: str (Default=train). To differentiate between training and validation/test runs.
              When not 'train', no data augmentation is applied.
        """
        self.df = df
        # self.cfg = cfg
        self.filepaths = df["filename"].values
        self.starts = df['start'].values * conf.SR
        self.offsets = df['end'].values * conf.SR - self.starts
        self.labels = torch.Tensor(np.ones([len(self.starts)]))
        # self.labels = torch.zeros((df.shape[0], cfg.n_classes))
        # self.weights = None
        # self.mode = mode

        # Pre-loads 1Hot label vectors
        # self.setup()

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        wave, _ = ta.load(
            self.filepaths[idx],
            frame_offset=self.starts[idx],
            num_frames=self.offsets[idx]
            )
        wave = wave.squeeze()
        if len(wave) < conf.CONTEXT_WIN:
            wave = librosa.util.fix_length(
                wave, 
                size=conf.CONTEXT_WIN,
                mode='mininum'
                )
        elif len(wave) > conf.CONTEXT_WIN:
            wave = wave[:conf.CONTEXT_WIN]
            

        # start = 0
        # # Cross-check whether file is as long as expected, e.g. 5s
        # # If not, apply zero-padding
        # max_time = int(self.cfg.wav_crop_len * sample_rate)
        # if wave.shape[0] <= max_time:
        #     pad = max_time - wave.shape[0]
        #     wave = torch.from_numpy(np.pad(wave, (0, pad)))
        # else:
        #     if self.mode == 'test' and self.cfg.max_amp:
        #         # Allows to get the audio window with the maximum average amplitude
        #         # Can be helpful during inference and evaluation
        #         start = get_max_amplitude_window_index('',
        #                                                waveform=wave,
        #                                                samplerate=sample_rate,
        #                                                window_length_sec=self.cfg.wav_crop_len,
        #                                                scan_param=50,
        #                                                verbose=False)

        # # Only necessary due to the max amplitude method
        # wave = wave[start:start + max_time]

        # if self.mode == 'train':
        #     # When in training mode, apply data augmentation
        #     wave = self.wave_transforms(wave, sample_rate)

        sample = {'wave': wave,
                  'labels': self.labels[idx],
                  }
        return sample

    # def setup(self):
    #     # Sets up 1Hot label vector
    #     if self.mode == 'train' or self.mode == 'val':
    #         self.labels = one_hot(
    #             torch.tensor(self.df['label'].values),
    #             num_classes=self.cfg.n_classes
    #             ).float()


class Loader(DataLoader):
    def __init__(self, df_path):
        """
        Lightning DataModule, containing any Pytorch Dataloader classes that are necessary during training,
        e.g. a train and validation dataloader.
        Deals with the corresponding splitting into train and validation and initializes pre-defined
        Pytorch dataloaders.
        The dataloaders deal with any necessary logic such as batching, parallelization across workers etc.

        The functions train_dataloader() and val_dataloader() will be internally called by the Lightning Trainer class
        during training.

        Parameters
        ----------
        cfg: SimpleNameSpace containing all configurations
        """
        super(DataLoader, self).__init__()
        self.df = pd.read_csv(df_path)
            
        self.train = AudioDataset(
            self.df[
                self.df['subset'] == 'train'
                ], 
            mode='train',
            )

        self.val = AudioDataset(
            self.df[
                self.df['subset'] == 'val'
                ], 
            mode='val',
            )

    def train_loader(self):
        return DataLoader(
            self.train, 
            batch_size=32,#self.cfg.batch_size, 
            shuffle=True, 
            pin_memory=True,
            num_workers=1,#self.cfg.num_workers, 
            persistent_workers=True, 
            collate_fn=collate_fn
            )

    def val_loader(self):
        return DataLoader(
            self.val, 
            batch_size=32,#self.cfg.batch_size, 
            shuffle=False, 
            pin_memory=True,
            num_workers=1,#self.cfg.num_workers, 
            persistent_workers=True, 
            collate_fn=collate_fn
            )
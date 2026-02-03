import numpy as np
import pandas as pd
import torch
import librosa
from torch.utils.data import DataLoader, Dataset
import torchaudio as ta

from pathlib import Path
from acodet import global_config as conf

def collate_fn(batch):
    waves = torch.stack([x['wave'] for x in batch])
    labels = torch.stack([x['labels'] for x in batch])
    starts = torch.tensor([x['start'] for x in batch], dtype=torch.float32)
    
    # this is a list of strings, can't be a torch tensor
    paths = [x['path'] for x in batch]
    
    return waves, labels, paths, starts


class AudioDataset(Dataset):
    def __init__(self, df, mode: str = 'train'):
        """
        Custom pytorch Dataset class, which loads a single sample and applies padding if needed.

        Parameters
        ----------
        df: Pandas dataframe, containing the path to the .wav files as well as the label.
        """
        self.df = df

        self.filepaths = df["filename"].values
        self.starts = df['start'].values * conf.SR
        self.offsets = df['end'].values * conf.SR - self.starts
        self.labels = torch.Tensor(np.zeros([2, len(self.starts)]))
        self.labels = torch.tensor(df.label.values)
        # idxs = np.arange(len(df))
        # self.labels[0, idxs[vals==0]] = torch.ones(len(vals[vals==0]))
        # self.labels[1, idxs[vals==1]] = torch.ones(len(vals[vals==1]))

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        wave, sr = ta.load(
            self.filepaths[idx],
            frame_offset=self.starts[idx],
            num_frames=self.offsets[idx]
            )
        if not sr == conf.SR:
            wave = ta.functional.resample(wave, sr, conf.SR)
        wave = wave.squeeze()
        if len(wave) < conf.CONTEXT_WIN:
            wave = librosa.util.fix_length(wave, 
                                           size=conf.CONTEXT_WIN,
                                           mode='minimum')
            wave = torch.from_numpy(wave)
        elif len(wave) > conf.CONTEXT_WIN:
            wave = wave[:conf.CONTEXT_WIN]
    
        sample = {
            'wave': wave,
            'labels': self.labels[idx],
            'path': self.filepaths[idx],
            'start': float(self.starts[idx] / conf.SR) 
        }
        return sample

class Loader(DataLoader):
    def __init__(self, df_path):
        """
        Deals with the corresponding splitting into train and validation and initializes pre-defined
        Pytorch dataloaders.
        """
        self.df_path = df_path
        super(DataLoader, self).__init__()
        
        combined_annots = Path(df_path) / 'combined_annotations.csv'
        explicit_noise = Path(df_path) / 'explicit_noise.csv'
        
        ca_df = pd.read_csv(combined_annots)
        en_df = pd.read_csv(explicit_noise)
        
        if not 'subset' in ca_df.columns:
            files = ca_df.filename.unique()
            eval_files = files[-int(len(files) * 0.2):]
            ca_df['subset'] = [''] * len(ca_df)
            ca_df.loc[ca_df.filename.isin(eval_files), 'subset'] = 'eval'
        
        if not 'subset' in en_df.columns:
            files = en_df.filename.unique()
            eval_files = files[-int(len(files) * 0.2):]
            en_df['subset'] = [''] * len(en_df)
            en_df.loc[en_df.filename.isin(eval_files), 'subset'] = 'eval'
            
        df = pd.concat([ca_df, en_df], ignore_index=True)
        
        df = df[df.subset != 'eval']
        
        rand_ints = np.random.permutation(len(df))
        border = int(len(df) * 0.8)
        
        train, val = df.iloc[rand_ints[:border]], df.iloc[rand_ints[border:]]
        train.subset = 'train'
        val.subset = 'val'
        train, val = train, val
        
        df = pd.concat([train, val], ignore_index=True)
            
        self.train = AudioDataset(
            df[
                df['subset'] == 'train'
                ], 
            mode='train',
            )

        self.val = AudioDataset(
            df[
                df['subset'] == 'val'
                ], 
            mode='val',
            )
        
        
        eval_df = pd.concat([ca_df, en_df], ignore_index=True)
        eval_df = eval_df[eval_df.subset == 'eval']
        
        
        eval_df = eval_df[:20]
        self.test = AudioDataset(
            eval_df,
            mode='test',
            )
        
    def noise_loader(self):
        # Filter strictly for Explicit Noise
        en_df = pd.read_csv(Path(self.df_path) / 'explicit_noise.csv')
        
        noise_dataset = AudioDataset(en_df, mode='train')
        
        return DataLoader(
            noise_dataset,
            batch_size=conf.BATCH_SIZE,
            shuffle=True,
            num_workers=1,
            pin_memory=True,
            drop_last=True, # Ensure we don't get tiny batches
            collate_fn=collate_fn
        )            

    def train_loader(self):
        return DataLoader(
            self.train, 
            batch_size=conf.BATCH_SIZE,
            shuffle=True, 
            pin_memory=True,
            num_workers=1,
            persistent_workers=True, 
            collate_fn=collate_fn
            )

    def val_loader(self):
        return DataLoader(
            self.val, 
            batch_size=conf.BATCH_SIZE,
            shuffle=False, 
            pin_memory=True,
            num_workers=1,
            persistent_workers=True, 
            collate_fn=collate_fn
            )
        
    def test_loader(self):
        return DataLoader(
            self.test, 
            batch_size=conf.BATCH_SIZE,
            shuffle=False, 
            pin_memory=True,
            num_workers=1,
            persistent_workers=True, 
            collate_fn=collate_fn
            )
    
    
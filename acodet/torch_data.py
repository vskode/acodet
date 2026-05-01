import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
import librosa as lb
from pathlib import Path
from acodet import global_config as conf
import torchaudio as ta

np.random.seed(42)

def collate_fn(batch):
    waves = torch.stack([x['wave'] for x in batch])
    labels = torch.stack([x['labels'] for x in batch])
    starts = torch.tensor([x['start'] for x in batch], dtype=torch.float32)
    
    # this is a list of strings, can't be a torch tensor
    paths = [x['path'] for x in batch]
    
    return waves, labels, paths, starts


class AudioDataset(Dataset):
    def __init__(self, df, mode: str = 'train'):
        self.mode = mode

        rows = []
        for _, row in df.iterrows():
            clip_duration = row['end'] - row['start']
            frame_duration = conf.CONTEXT_WIN / conf.SR
            
            # skip clips that are too short to be meaningful
            if clip_duration < 0.5:  # adjust threshold to your needs
                continue
            
            num_frames = max(1, int(np.ceil(clip_duration / frame_duration)))
            
            for i in range(num_frames):
                frame_start = row['start'] + i * frame_duration
                # clamp so we don't seek past the actual annotation end
                frame_start = min(frame_start, row['end'] - frame_duration)
                rows.append({
                    'filename': row['filename'],
                    'label': row['label'],
                    'start': max(0, frame_start),  # also guard against negative offsets
                    'duration': frame_duration
                })

        expanded_df = pd.DataFrame(rows)

        self.filepaths = expanded_df['filename'].values
        self.starts = expanded_df['start'].values
        self.durations = expanded_df['duration'].values
        self.labels = torch.tensor(expanded_df['label'].values)

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        wave, sr = ta.load(
            self.filepaths[idx], 
            frame_offset=self.starts[idx] * 48_000, 
            num_frames=self.durations[idx] * 48_000
            )
        wave = wave.mean(dim=0).numpy()  # convert to mono if needed

        if sr != self.sample_rate:
            wave = ta.functional.resample(
                torch.tensor(wave), orig_freq=sr, new_freq=conf.SR
            ).numpy()
        
        # wave, sr = lb.load(
        #     path=self.filepaths[idx],
        #     sr=conf.SR,
        #     offset=self.starts[idx],
        #     duration=self.durations[idx]
        # )
        # wave = torch.tensor(wave).squeeze()

        # Only the last frame of a long clip may be short
        if len(wave) < conf.CONTEXT_WIN:
            wave = torch.tensor(
                lb.util.fix_length(
                    wave.numpy(), 
                    size=conf.CONTEXT_WIN, 
                    mode='wrap'
                    )
            )

        return {
            'wave': wave,
            'labels': self.labels[idx],
            'path': self.filepaths[idx],
            'start': self.starts[idx]
        }

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
        
        
        # eval_df = eval_df[:20]
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
            num_workers=2,
            prefetch_factor=2,
            persistent_workers=True, 
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
            num_workers=2,
            prefetch_factor=2,
            persistent_workers=True, 
            collate_fn=collate_fn
            )

    def val_loader(self):
        return DataLoader(
            self.val, 
            batch_size=conf.BATCH_SIZE,
            shuffle=False, 
            pin_memory=True,
            num_workers=2,
            prefetch_factor=2,
            persistent_workers=True, 
            collate_fn=collate_fn
            )
        
    def test_loader(self):
        return DataLoader(
            self.test, 
            batch_size=conf.BATCH_SIZE,
            shuffle=False, 
            pin_memory=True,
            num_workers=2,
            prefetch_factor=2,
            persistent_workers=True, 
            collate_fn=collate_fn
            )
    
    
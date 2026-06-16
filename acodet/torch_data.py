import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
import librosa as lb
from pathlib import Path
from acodet import global_config as conf
from acodet.combine_annotations import read_annotations_from_file

NUM_CORES = os.cpu_count() or 10
if hasattr(os, "sched_getaffinity"):
    # This function is only available on certain platforms. When running with Slurm, it can tell us the true
    # number of cores we have access to.
    NUM_CORES = len(os.sched_getaffinity(0))
print(f"Using {NUM_CORES} cores.")


def collate_fn(batch):
    # Filter examples that aren't the full size.
    expected_sz = len(batch[0]['wave'])
    batch = [x for x in batch if len(x['wave']) == expected_sz]
    waves = torch.stack([x['wave'] for x in batch])
    labels = torch.stack([x['labels'] for x in batch])
    starts = torch.tensor([x['start'] for x in batch], dtype=torch.float32)
    
    # this is a list of strings, can't be a torch tensor
    paths = [x['path'] for x in batch]
    
    return waves, labels, paths, starts


class AudioDataset(Dataset):
    def __init__(self, df, mode: str = 'train'):
        assert len(df) > 0
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
        
        import time, random

        for attempt in range(3):
            try:
                wave, sr = lb.load(
                    path=self.filepaths[idx],
                    sr=conf.SR,
                    offset=self.starts[idx],
                    duration=self.durations[idx]
                )
                wave = torch.tensor(wave).squeeze()
                break
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(random.uniform(0.1, 0.5))
        
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
        super(DataLoader, self).__init__()
        self.df_path = df_path

        train_df, val_df, eval_df = read_annotations_from_file(df_path)

        self.train = AudioDataset(train_df, mode='train') if len(train_df) > 0 else pd.DataFrame()
        self.val = AudioDataset(val_df, mode='val') if len(val_df) > 0 else pd.DataFrame()
        self.test = AudioDataset(eval_df, mode='test') if len(eval_df) > 0 else pd.DataFrame()

    def noise_loader(self):
        # Filter strictly for Explicit Noise
        en_df = pd.read_csv(Path(self.df_path) / 'explicit_noise.csv')
        
        noise_dataset = AudioDataset(en_df, mode='train')
        
        return DataLoader(
            noise_dataset,
            batch_size=conf.BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_CORES,
            prefetch_factor=4,
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
            num_workers=NUM_CORES,
            prefetch_factor=4,
            persistent_workers=True, 
            collate_fn=collate_fn
            )

    def val_loader(self):
        return DataLoader(
            self.val, 
            batch_size=conf.BATCH_SIZE,
            shuffle=False, 
            pin_memory=True,
            num_workers=NUM_CORES,
            prefetch_factor=4,
            persistent_workers=True, 
            collate_fn=collate_fn
            )
        
    def test_loader(self):
        return DataLoader(
            self.test, 
            batch_size=conf.BATCH_SIZE,
            shuffle=False, 
            pin_memory=True,
            num_workers=NUM_CORES,
            prefetch_factor=4,
            persistent_workers=True, 
            collate_fn=collate_fn
            )
    
    
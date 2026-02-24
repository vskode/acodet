import tensorflow as tf
import numpy as np
import pandas as pd
from pathlib import Path
import torchaudio as ta
ta.set_audio_backend("soundfile")  # Avoid torchcodec
import librosa
import torch
import acodet.global_config as conf

class TFAudioDataset:
    def __init__(self, df, mode='train'):
        """TensorFlow equivalent of your PyTorch AudioDataset"""
        self.df = df.reset_index(drop=True)  # Important for iteration
        self.mode = mode
        
    def audio_generator(self):
        """Generator that yields audio samples"""
        for idx, row in self.df.iterrows():
            # Load audio segment
            start_frame = int(row['start'] * conf.SR)
            end_frame = int(row['end'] * conf.SR)
            num_frames = end_frame - start_frame
            
            wave, sr = ta.load(
                row['filename'],
                frame_offset=start_frame,
                num_frames=num_frames
            )
            
            # Resample if needed
            if sr != conf.SR:
                wave = ta.functional.resample(wave, sr, conf.SR)
            
            wave = wave.squeeze().numpy()
            
            # Padding/truncation
            if len(wave) < conf.CONTEXT_WIN:
                wave = librosa.util.fix_length(
                    wave, 
                    size=conf.CONTEXT_WIN, 
                    mode='minimum'
                )
            elif len(wave) > conf.CONTEXT_WIN:
                wave = wave[:conf.CONTEXT_WIN]
            
            yield wave.astype(np.float32), np.int32(row['label'])
    
    def get_dataset(self):
        """Create tf.data.Dataset from generator"""
        dataset = tf.data.Dataset.from_generator(
            self.audio_generator,
            output_signature=(
                tf.TensorSpec(shape=(conf.CONTEXT_WIN,), dtype=tf.float32),
                tf.TensorSpec(shape=(), dtype=tf.int32)
            )
        )
        return dataset


class TFLoader:
    """TensorFlow equivalent of your PyTorch Loader"""
    def __init__(self, df_path):
        self.df_path = df_path
        
        combined_annots = Path(df_path) / 'combined_annotations.csv'
        explicit_noise = Path(df_path) / 'explicit_noise.csv'
        
        ca_df = pd.read_csv(combined_annots)
        en_df = pd.read_csv(explicit_noise)
        
        # Add subset column if missing
        if 'subset' not in ca_df.columns:
            files = ca_df.filename.unique()
            eval_files = files[-int(len(files) * 0.2):]
            ca_df['subset'] = ''
            ca_df.loc[ca_df.filename.isin(eval_files), 'subset'] = 'eval'
        
        if 'subset' not in en_df.columns:
            files = en_df.filename.unique()
            eval_files = files[-int(len(files) * 0.2):]
            en_df['subset'] = ''
            en_df.loc[en_df.filename.isin(eval_files), 'subset'] = 'eval'
        
        df = pd.concat([ca_df, en_df], ignore_index=True)
        df = df[df.subset != 'eval']
        
        # Split train/val
        rand_ints = np.random.permutation(len(df))
        border = int(len(df) * 0.8)
        
        train_df = df.iloc[rand_ints[:border]].copy()
        val_df = df.iloc[rand_ints[border:]].copy()
        train_df['subset'] = 'train'
        val_df['subset'] = 'val'
        
        # Eval set
        eval_df = pd.concat([ca_df, en_df], ignore_index=True)
        eval_df = eval_df[eval_df.subset == 'eval'][:20]
        
        # Create datasets
        self.train = TFAudioDataset(train_df, mode='train')
        self.val = TFAudioDataset(val_df, mode='val')
        self.test = TFAudioDataset(eval_df, mode='test')
        
        # get sizes
        self.n_train = len(train_df)
        self.n_val = len(val_df)
        self.n_eval = len(eval_df)
        self.n_noise = len(df[df.label==0])
            

    
    def train_loader(self, batch_size=None):
        batch_size = batch_size or conf.BATCH_SIZE
        dataset = self.train.get_dataset()
        # dataset = dataset.shuffle(
        #     buffer_size=1000, 
        #     reshuffle_each_iteration=True
        # )
        # dataset = dataset.batch(batch_size, drop_remainder=True)
        # dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset
    
    def val_loader(self, batch_size=None):
        batch_size = batch_size or conf.BATCH_SIZE
        dataset = self.val.get_dataset()
        # dataset = dataset.batch(batch_size)
        # dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset
    
    def test_loader(self, batch_size=None):
        batch_size = batch_size or conf.BATCH_SIZE
        dataset = self.test.get_dataset()
        # dataset = dataset.batch(batch_size)
        # dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset
    
    def noise_loader(self, batch_size=None):
        batch_size = batch_size or conf.BATCH_SIZE
        en_df = pd.read_csv(Path(self.df_path) / 'explicit_noise.csv')
        noise_dataset = TFAudioDataset(en_df, mode='train')
        
        dataset = noise_dataset.get_dataset()
        # dataset = dataset.shuffle(buffer_size=1000)
        # dataset = dataset.batch(batch_size, drop_remainder=True)
        # dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset
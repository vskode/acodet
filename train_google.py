#%%
import pandas as pd
import numpy as np
from pathlib import Path
import os

from utils.funcs import *
from utils.tfrecord_funcs import *
from utils.google_funcs import GoogleMod

params = {
    "sr" : 10000,
    "cntxt_wn_hop": 39124,
    "cntxt_wn_sz": 39124,
    "fft_window_length" : 2**11,
    "n_freq_bins": 2**8,
    "freq_cmpr": 'linear',
    "fmin":  50,
    "fmax":  1000,
    "nr_noise_samples": 100,
    "sequence_len" : 39124,
    "fft_hop": 300,
}

TFRECORDS_DIR = 'Daten/tfrecords'

def prepare_sample(features):
    return features["audio"], features["label"]

def get_dataset(filenames, batch_size):
    dataset = (
        tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTOTUNE)
        .map(parse_tfrecord_fn, num_parallel_calls=AUTOTUNE)
        .map(prepare_sample, num_parallel_calls=AUTOTUNE)
        .batch(batch_size * 10)
        .take(10)
        # .prefetch(AUTOTUNE)
    )
    return dataset

#%%

filenames = tf.io.gfile.glob(f"{TFRECORDS_DIR}/*.tfrec")
batch_size = 32
epochs = 1
steps_per_epoch = 50
AUTOTUNE = tf.data.AUTOTUNE

G = GoogleMod(params)
model = G.model
data = get_dataset(filenames, batch_size)

dset = tf.data.TFRecordDataset(f"{TFRECORDS_DIR}/file_01.tfrec").map(parse_tfrecord_fn)
ll = list()
for features in dset:
    ll.append(list(features["audio"].numpy()))
e = model.predict(np.array(ll, dtype='float32'))

dataset = (
    tf.data.TFRecordDataset(filenames[:1], num_parallel_reads=AUTOTUNE)
    .map(parse_tfrecord_fn, num_parallel_calls=AUTOTUNE)
    .map(prepare_sample, num_parallel_calls=AUTOTUNE).batch(1)
    )
a = model.predict(dataset, verbose =2)


model.evaluate(data,
               batch_size = batch_size, verbose =2)
model.predict(x = get_dataset(filenames, batch_size))

model.fit(x=get_dataset(filenames, batch_size), 
          epochs = epochs, 
          steps_per_epoch=steps_per_epoch, 
          verbose=1)

print('end')
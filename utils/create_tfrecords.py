import numpy as np
import tensorflow as tf
import pandas as pd
from funcs import *
from pathlib import Path

params = {
    "sr" : 10000,
    "cntxt_wn_sz": 39124,
    "nr_noise_samples": 100,
}

num_tfrecords = 1
tfrecords_dir = 'Daten/tfrecords'
if not Path(tfrecords_dir).exists():
    Path(tfrecords_dir).mkdir()

annots = pd.read_csv('Daten/ket_annot.csv')
files = np.unique(annots.filename) 
    
def audio_feature(list_of_floats):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=list_of_floats))

def label_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[int(value)]))


def create_example(audio, label):
    feature = {
        "audio": audio_feature(audio),
        "label": label_feature(label)
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))


def parse_tfrecord_fn(example):
    feature_description = {
        "audio": tf.io.FixedLenFeature([params['cntxt_wn_sz']], tf.float32),
        "label": tf.io.FixedLenFeature([], tf.int64),
    }
    example = tf.io.parse_single_example(example, feature_description)
    return example

def read_tfrecords():
    raw_dataset = tf.data.TFRecordDataset(f"{tfrecords_dir}/file_00-270.tfrec")
    parsed_dataset = raw_dataset.map(parse_tfrecord_fn)


def read_raw_files():
    for i, file in enumerate(files[:1]):
        
        file_annots = get_annots_for_file(annots, file)
        file_annots.start -= params['cntxt_wn_sz'] / params['sr'] / 2
        
        x_test, x_noise = return_cntxt_wndw_arr(file_annots, file, **params)        
        y_test = np.ones(len(x_test), dtype = 'float32')
        y_noise = np.zeros(len(x_noise), dtype = 'float32')
    
    
def write_tfrecords(audio, label):
    for tfrec_num in range(num_tfrecords):
        samples = zip(audio, label)

        with tf.io.TFRecordWriter(
                tfrecords_dir + "/file_%.2i.tfrec" % tfrec_num) as writer:
            for audio, label in samples:
                example = create_example(audio, label)
                writer.write(example.SerializeToString())


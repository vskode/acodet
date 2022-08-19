import numpy as np
import tensorflow as tf
import pandas as pd
from utils.funcs import *

from pathlib import Path

params = {
    "sr" : 10000,
    "cntxt_wn_sz": 39124,
    "nr_noise_samples": 100,
}

# num_tfrecords = 1
TFRECORDS_DIR = 'Daten/tfrecords'
FILE_ARRAY_LIMIT = 600
if not Path(TFRECORDS_DIR).exists():
    Path(TFRECORDS_DIR).mkdir()

annots = pd.read_csv('Daten/ket_annot.csv')
files = np.unique(annots.filename) 
    
def audio_feature(list_of_floats):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=list_of_floats))

def int_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[int(value)]))
 
def string_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[bytes(value, 'utf-8')]))

# def string_feature(value):
#     return tf.train.Feature(string_list=tf.train.StringList(value=value))

def create_example(audio, label, file, time):
    feature = {
        "audio": audio_feature(audio),
        "label": int_feature(label),
        "file" : string_feature(file),
        "time" : int_feature(time)
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))


def parse_tfrecord_fn(example):
    feature_description = {
        "audio": tf.io.FixedLenFeature([params['cntxt_wn_sz']], tf.float32),
        "label": tf.io.FixedLenFeature([], tf.int64),
        "file" : tf.io.FixedLenFeature([], tf.string),
        "time" : tf.io.FixedLenFeature([], tf.int64)
    }
    example = tf.io.parse_single_example(example, feature_description)
    return example

def read_tfrecords():
    raw_dataset = tf.data.TFRecordDataset(f"{TFRECORDS_DIR}/file_01.tfrec")
    parsed_dataset = raw_dataset.map(parse_tfrecord_fn)


def read_raw_file(file):
        
    file_annots = get_annots_for_file(annots, file)
    file_annots.start -= params['cntxt_wn_sz'] / params['sr'] / 2

    x_call, x_noise, times_c, times_n = return_cntxt_wndw_arr(file_annots, 
                                                              file, 
                                                        return_times =True,
                                                            **params)    
    y_call = np.ones(len(x_call), dtype = 'float32')
    y_noise = np.zeros(len(x_noise), dtype = 'float32')
        
    
    return (x_call, y_call, times_c), (x_noise, y_noise, times_n)
    
def write_tfrecords(files):
    # for tfrec_num in range(num_tfrecords):
    tfrec_num = 0
    array_per_file = 0
    for i, file in enumerate(files):
        call_tup, noise_tup = read_raw_file(file)
        x_call, y_call, times_c = call_tup
        x_noise, y_noise, times_n = noise_tup
        
        calls = zip(x_call, y_call, [file]*len(x_call), times_c)
        noise = zip(x_noise, y_noise, [file]*len(x_noise), times_n)

        
        for samples in [calls, noise]:
            for audio, label, file, time in samples:
                if array_per_file > FILE_ARRAY_LIMIT or \
                                        array_per_file == tfrec_num == 0:
                    tfrec_num += 1
                    writer = get_tfrecords_writer(tfrec_num)
                    array_per_file = 0
                example = create_example(audio, label, file, time)
                writer.write(example.SerializeToString())
                array_per_file += 1

def get_tfrecords_writer(num):
    return tf.io.TFRecordWriter(TFRECORDS_DIR + "/file_%.2i.tfrec" % num)

def show_google_spec(audio):
    from utils.google_funcs import GoogleMod
    mod = GoogleMod(params)
    spec_func = mod.model.layers[1].call
    pcen_func = mod.model.layers[2].call
    tensor_sig = tf.expand_dims( tf.convert_to_tensor([audio[0]]), -1 )
    spec = pcen_func(spec_func(tensor_sig))
    plt.figure()
    plt.imshow(spec[0].numpy().T, origin='lower')

if __name__ == '__main__':
    write_tfrecords(files)
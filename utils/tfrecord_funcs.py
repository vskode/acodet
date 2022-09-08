import numpy as np
import tensorflow as tf
import pandas as pd
from . import funcs

from pathlib import Path

params = {
    "sr" : 10000,
    "cntxt_wn_sz": 39124,
    "nr_noise_samples": 100,
}

FILE_ARRAY_LIMIT = 600
TFRECORDS_DIR = 'Daten/tfrecords'


def exclude_files_from_dataset(annots):
    exclude_files = [
        '180324160003',
        'PAM_20180323',
        'PAM_20180324',
        'PAM_20180325_0',
        'PAM_20180325_17',
        'PAM_20180326',
        'PAM_20180327',
        'PAM_20180329',
        'PAM_20180318',
        'PAM_20190321',
        '20022315',
        '20022621',
        '210318',
        '210319',
        '210327',
    ]
    get_stems = lambda s: Path(s).stem.split('.')[-1]

    annots['stems'] = list(map(get_stems, annots.filename))

    drop_files = []
    for file in np.unique(annots.stems):
        for exclude in exclude_files:
            if exclude in file:
                drop_files.append(file)
    annots.index = annots.stems

    return annots.drop(annots.loc[drop_files].index)
    
def audio_feature(list_of_floats):
    """
    Returns a list of floats.

    Args:
        list_of_floats (list): list of floats

    Returns:
        tf.train.Feature: tensorflow feature containing a list of floats
    """
    return tf.train.Feature(float_list=tf.train.FloatList(value=list_of_floats))

def int_feature(value):
    """
    Returns a int value.

    Args:
        value (int): label value

    Returns:
        tf.train.Feature: tensorflow feature containing a int
    """
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[int(value)]))
 
def string_feature(value):
    """
    Returns a bytes array.

    Args:
        value (string): path to file

    Returns:
        tf.train.Feature: tensorflow feature containing a bytes array
    """
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[bytes(value, 'utf-8')]))


def create_example(audio, label, file, time):
    """
    Create a tensorflow Example object containing the tf Features that will
    be saved in one file. The file will contain the audio data, corresponding
    label, the file path corresponding to the audio, and the time in samples
    where in the file that audio data begins. The audio data is saved
    according to the frame rate in params, for the Google model 10 kHz.
    The label is binary, either 0 for noise or 1 for call.

    Args:
        audio (list): raw audio data of length params['cntxt_wn_sz']
        label (int): either 1 for call or 0 for noise
        file (string): path of file corresponding to audio data
        time (int): time in samples when audio data begins within file

    Returns:
        tf.train.Example: Example object containing all data
    """
    feature = {
        "audio": audio_feature(audio),
        "label": int_feature(label),
        "file" : string_feature(file),
        "time" : int_feature(time)
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))


def parse_tfrecord_fn(example):
    """
    Parser for tfrecords files.

    Args:
        example (tf.train.Example instance): file containing 4 features

    Returns:
        tf.io object: tensorflow object containg the data
    """
    feature_description = {
        "audio": tf.io.FixedLenFeature([params['cntxt_wn_sz']], tf.float32),
        "label": tf.io.FixedLenFeature([], tf.int64),
        "file" : tf.io.FixedLenFeature([], tf.string),
        "time" : tf.io.FixedLenFeature([], tf.int64)
    }
    example = tf.io.parse_single_example(example, feature_description)
    # audio = example['audio']
    # label = example['label']
    return example

def read_tfrecords(num):
    """
    Read tfrecords file and return the parse_dataset.

    Args:
        num (int): number of tfrecords file

    Returns:
        TFRecordDataset: parsed dataset
    """
    dataset = tf.data.TFRecordDataset(f"{TFRECORDS_DIR}/file_{num:02}.tfrec")
    return dataset.map(parse_tfrecord_fn)

def read_raw_file(file, shift = 0):
    """
    Load annotations for file, correct annotation starting times to make sure
    that the signal is in the window center.

    Args:
        file (string): path to file

    Returns:
        tuple: audio segment arrays, label arrays and time arrays
    """
        
    file_annots = funcs.get_annots_for_file(annots, file)
    file_annots.start -= shift

    x_call, x_noise, times_c, times_n = funcs.return_cntxt_wndw_arr(file_annots, 
                                                              file, 
                                                        return_times =True,
                                                            **params)    
    y_call = np.ones(len(x_call), dtype = 'float32')
    y_noise = np.zeros(len(x_noise), dtype = 'float32')
    
    return (x_call, y_call, times_c), (x_noise, y_noise, times_n)
    
def write_tfrecords(files, shift = 0):
    """
    Write tfrecords files from wav files. 
    First the files are imported and the noise files are generated. After that 
    a loop iterates through the tuples containing the audio arrays, labels, and
    startint times of the audio arrays within the given file. tfrecord files
    contain no more than 600 audio arrays. The respective audio segments, 
    labels, starting times, and file paths are saved in the files.

    Args:
        files (list): list of file paths to the audio files
    """
    tfrec_num, array_per_file = 0, 0
    
    for i, file in enumerate(files):
        print(f'writing tf records files, progress: {i/len(files)*100:.0f} %', end = '\r')
        
        call_tup, noise_tup = read_raw_file(file, shift = shift)
        x_call, y_call, times_c = call_tup
        x_noise, y_noise, times_n = noise_tup
        
        calls = zip(x_call, y_call, [file]*len(x_call), times_c)
        noise = zip(x_noise, y_noise, [file]*len(x_noise), times_n)

        for samples in [calls, noise]:
            for audio, label, file, time in samples:
                if array_per_file > FILE_ARRAY_LIMIT or \
                                        array_per_file == tfrec_num == 0:
                    tfrec_num += 1
                    writer = get_tfrecords_writer(tfrec_num, shift = shift)
                    array_per_file = 0
                    
                example = create_example(audio, label, file, time)
                writer.write(example.SerializeToString())
                array_per_file += 1

def get_tfrecords_writer(num, shift = 0):
    """
    Return TFRecordWriter object to write file.

    Args:
        num (int): file number

    Returns:
        TFRecordWriter object: file handle
    """
    path = TFRECORDS_DIR + f'_{shift}s_shift'
    if not Path(path).exists():
        Path(path).mkdir()
    return tf.io.TFRecordWriter(path + "/file_%.2i.tfrec" % num)

if __name__ == '__main__':

    annots = pd.read_csv('Daten/ket_annot.csv')
    annots = exclude_files_from_dataset(annots) 

    for shift in [0, 1, 2]:
        write_tfrecords(np.unique(annots.filename) , shift = shift)
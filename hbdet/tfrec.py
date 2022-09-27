import numpy as np
import tensorflow as tf
import pandas as pd
from . import funcs
import random
import yaml
import matplotlib.pyplot as plt
from pathlib import Path

with open('hbdet/hbdet/config.yml', 'r') as f:
    config = yaml.safe_load(f)

params = config['preproc']

FILE_ARRAY_LIMIT = 600
TFRECORDS_DIR = 'Daten/tfrecords'

########################################################
#################  WRITING   ###########################
########################################################


def exclude_files_from_dataset(annots):
    """
    Because some of the calls are very faint, a number of files are rexcluded
    from the dataset to make sure that the model performance isn't obscured 
    by poor data. 

    Args:
        annots (pd.DataFrame): annotations

    Returns:
        pd.DataFrame: cleaned annotation dataframe
    """
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
    drop_files = []
    for file in np.unique(annots.filename):
        for exclude in exclude_files:
            if exclude in file:
                drop_files.append(file)
    annots.index = annots.filename

    return annots.drop(annots.loc[drop_files].index), annots.loc[drop_files]
    
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

def read_raw_file(file, annots, shift = 0):
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

    
def write_tfrecords(annots, shift = 0, **kwArgs):
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

    files = np.unique(annots.filename)
    
    random.shuffle(files)

    train_file_index = int(len(files)*config['tfrec']['train_ratio'])
    test_file_index = int(len(files)*config['tfrec']['test_val_ratio'])
    
    for i, file in enumerate(files):
        print('writing tf records files, progress:'
              f'{i/len(files)*100:.0f} %')
        
        if i < train_file_index:
            folder = 'train'
        elif i < train_file_index + test_file_index:
            folder = 'test'
        else:
            folder = 'val'

        call_tup, noise_tup = read_raw_file(file, annots, shift = shift)
        
        calls = randomize_arrays(call_tup, file)
        noise = randomize_arrays(noise_tup, file)
        
        for samples in [calls, noise]:
            for audio, label, file, time in samples:
                if array_per_file > FILE_ARRAY_LIMIT or \
                                        array_per_file == tfrec_num == 0:
                    tfrec_num += 1
                    writer = get_tfrecords_writer(tfrec_num, folder, 
                                                shift = shift, **kwArgs)
                    array_per_file = 0
                    
                example = create_example(audio, label, file, time)
                writer.write(example.SerializeToString())
                array_per_file += 1

def randomize_arrays(tup, file):
    x, y, times = tup
    
    rand = np.arange(len(x))
    random.shuffle(rand)
    
    return zip(x[rand], y[rand], [file]*len(x), np.array(times)[rand])
    

def get_tfrecords_writer(num, fold, shift = 0, alt_subdir = ''):
    """
    Return TFRecordWriter object to write file.

    Args:
        num (int): file number

    Returns:
        TFRecordWriter object: file handle
    """
    path = TFRECORDS_DIR + alt_subdir + \
            f"_{str(shift).replace('.','-')}s_shift"
    Path(path + f'/{fold}').mkdir(parents = True, exist_ok = True)
    return tf.io.TFRecordWriter(path + f"/{fold}/"
                                "file_%.2i.tfrec" % num)


########################################################
#################  READING   ###########################
########################################################

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
    return tf.io.parse_single_example(example, feature_description)


def check_random_spectrogram(filenames, dataset_size = FILE_ARRAY_LIMIT):
    r = np.random.randint(dataset_size)
    dataset = (
        tf.data.TFRecordDataset(filenames)
        .map(parse_tfrecord_fn)
        .skip(r)
        .take(1)
    )

    sample = next(iter(dataset))
    aud, file, lab, time = (sample[k].numpy() for k in list(sample.keys()))
    file = file.decode()

    fig, ax = plt.subplots(ncols = 2, figsize = [12, 8])
    ax[0] = funcs.simple_spec(aud, fft_window_length = 512, sr = 10000, 
                                ax = ax[0], colorbar = False)
    _, ax[1] = funcs.plot_spec_from_file(file, ax = ax[1], start = time, 
                                        fft_window_length = 512, sr = 10000, 
                                        fig = fig)
    ax[0].set_title(f'Spec of audio sample from \ntfrecords array nr. {r}'
                    f' | label: {lab}')
    ax[1].set_title(f'Spec of audio sample from file: \n{Path(file).stem}'
                    f' | time in file: {funcs.get_time(time/10000)}')

    fig.suptitle('Comparison between tfrecord audio and file audio')
    fig.savefig(f'{TFRECORDS_DIR}_check_imgs/comp_{Path(file).stem}.png')


def prepare_sample(features):
    return features["audio"], features["label"]

def get_dataset(filenames, batch_size, AUTOTUNE):
    dataset = (
        tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTOTUNE)
        .map(parse_tfrecord_fn, num_parallel_calls=AUTOTUNE)
        .map(prepare_sample, num_parallel_calls=AUTOTUNE)
        .batch(batch_size)
        .prefetch(AUTOTUNE)
    )
    return dataset

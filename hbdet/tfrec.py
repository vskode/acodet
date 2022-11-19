from tokenize import Intnumber
from functools import partial
import numpy as np
import tensorflow as tf
from . import funcs
import random
from .humpback_model_dir import front_end
from pathlib import Path
import json

config = funcs.load_config()
TFRECORDS_DIR = config.data_dir

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

def read_raw_file(file, annots, **kwargs):
    """
    Load annotations for file, correct annotation starting times to make sure
    that the signal is in the window center.

    Args:
        file (string): path to file

    Returns:
        tuple: audio segment arrays, label arrays and time arrays
    """
        
    file_annots = funcs.get_annots_for_file(annots, file)

    x_call, x_noise, times_c, times_n = funcs.cntxt_wndw_arr(file_annots,
                                                            file, **kwargs)
    y_call = np.ones(len(x_call), dtype = 'float32')
    y_noise = np.zeros(len(x_noise), dtype = 'float32')
    
    return (x_call, y_call, times_c), (x_noise, y_noise, times_n)

def write_tfrecs_for_mixup(file):
    noise, t = funcs.return_windowed_file(file)
    noise_tups = list(zip(noise, [0]*len(t), [file]*len(t), t))
    random.shuffle(noise_tups)
    
    writer = get_tfrecords_writer(1, 'noise')
    for audio, label, file, time in noise_tups:
        examples = create_example(audio, label, file, time)
        writer.write(examples.SerializeToString())
                
def write_tfrecords(annots, all_noise=False, inbetween_noise=True, **kwargs):
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
    files = np.unique(annots.filename)
    
    random.shuffle(files)
    
    split_mode = 'within_file'
    specs = ['size', 'noise', 'calls']
    if all_noise:
        folders = ['noise']
    else:
        folders = ['train', 'test', 'val']
        train_file_index = int(len(files)*config.train_ratio)
        test_file_index = int(len(files)
                            *(1-config.train_ratio)
                            *config.test_val_ratio)
        
    dataset = {k: {k1: 0 for k1 in folders} 
               for k in specs}
    data_meta_dict = dict({'data_split': split_mode})
    files_dict = {}
    tfrec_num = 0
    for i, file in enumerate(files):
        print('writing tf records files, progress:'
              f'{i/len(files)*100:.0f} %')
        
        if all_noise:
            folder = folders[0]
        else:
            if i < train_file_index:
                folder = folders[0]
            elif i < train_file_index + test_file_index:
                folder = folders[1]
            else:
                folder = folders[2]

        call_tup, noise_tup = read_raw_file(file, annots, 
                                            inbetween_noise=inbetween_noise, 
                                            **kwargs)
        
        calls = randomize_arrays(call_tup, file)
        noise = randomize_arrays(noise_tup, file)
        samples = [*calls, *noise]
        random.shuffle(samples)
        data = dict()
        if all_noise:
            data['noise'] = samples
        else:
            end_tr, end_te = map(lambda x: int(x*len(samples)),
                                (config.train_ratio, (1-config.train_ratio)
                                                    *config.test_val_ratio
                                                    +config.train_ratio) )
            
            data['train'] = samples[:end_tr]
            data['test'] = samples[end_tr:end_te]
            data['val'] = samples[end_tr:-1]
        
        
        for folder, samples in data.items():
            split_by_max_length = [samples[j*config.tfrecs_lim:(j+1) * config.tfrecs_lim] \
                                    for j in range(len(samples)//config.tfrecs_lim + 1)]
            for samps in split_by_max_length:
                tfrec_num += 1
                writer = get_tfrecords_writer(tfrec_num, folder, **kwargs)
                files_dict, dataset = update_dict(samps, files_dict,
                                                  dataset, folder, tfrec_num)
                
                for audio, label, file, time in samps:
                    examples = create_example(audio, label, file, time)
                    writer.write(examples.SerializeToString())
                    
    # TODO automatisch die noise sachen miterstellen
    data_meta_dict.update({'dataset': dataset})
    data_meta_dict.update({'files': files_dict})
    with open(Path(TFRECORDS_DIR).joinpath('dataset_meta.json'), 'w') as f:
        json.dump(data_meta_dict, f)
    
def randomize_arrays(tup, file):
    x, y, times = tup
    
    rand = np.arange(len(x))
    random.shuffle(rand)
    
    return zip(x[rand], y[rand], [file]*len(x), np.array(times)[rand])
    
def update_dict(samples, d, dataset_dict, folder, tfrec_num):
    calls = sum(1 for i in samples if i[1] == 1)
    noise = sum(1 for i in samples if i[1] == 0)
    size = noise+calls
    d.update({f"file_%.2i_{folder}" % tfrec_num: 
                        k for k in [size, noise, calls]})
    for l, k in zip(('size', 'calls', 'noise'), (size, calls, noise)):
        dataset_dict[l][folder] += k
    return d, dataset_dict

def get_tfrecords_writer(num, fold, alt_subdir = ''):
    """
    Return TFRecordWriter object to write file.

    Args:
        num (int): file number

    Returns:
        TFRecordWriter object: file handle
    """
    path = TFRECORDS_DIR + alt_subdir
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
        "audio": tf.io.FixedLenFeature([config.context_win], tf.float32),
        "label": tf.io.FixedLenFeature([], tf.int64),
        "file" : tf.io.FixedLenFeature([], tf.string),
        "time" : tf.io.FixedLenFeature([], tf.int64)
    }
    return tf.io.parse_single_example(example, feature_description)
    
def prepare_sample(features, return_meta=False, **kwargs):
    if not return_meta:
        return features["audio"], features["label"]
    else:
        return features["audio"], features["label"], features["file"], features["time"]

def get_dataset(filenames, AUTOTUNE=None, **kwargs):
    dataset = (
        tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTOTUNE)
        .map(parse_tfrecord_fn, num_parallel_calls=AUTOTUNE)
        .map(partial(prepare_sample, **kwargs), num_parallel_calls=AUTOTUNE)
    )
    return dataset

def run_data_pipeline(root_dir, data_dir, AUTOTUNE=None, return_spec=True, **kwargs):
    if not isinstance(root_dir, list):
        root_dir = [root_dir]
    files = []
    for root in root_dir:
        files += tf.io.gfile.glob(f"{root}/{data_dir}/*.tfrec")
    dataset = get_dataset(files, AUTOTUNE = AUTOTUNE, **kwargs)
    if return_spec:
        return make_spec_tensor(dataset)
    else:
        return dataset

def spec():
    return tf.keras.Sequential([
        tf.keras.layers.Input([config.context_win]),
        tf.keras.layers.Lambda(lambda t: tf.expand_dims(t, -1)),
        front_end.MelSpectrogram()])

def prepare(ds, batch_size, shuffle=False, shuffle_buffer=750, 
            augmented_data=None, AUTOTUNE=None):
    if shuffle:
        ds = ds.shuffle(shuffle_buffer)    
    ds = ds.batch(batch_size)    
    return ds.prefetch(buffer_size=AUTOTUNE)

def make_spec_tensor(ds, AUTOTUNE=None):
    ds = ds.batch(1)
    ds = ds.map(lambda x, y: (spec()(x), y), num_parallel_calls=AUTOTUNE)
    return ds.unbatch()
from email import generator
import re
import zipfile
import datetime as dt
import json
import tensorflow as tf
import numpy as np
import librosa as lb
from pathlib import Path
import pandas as pd
from . import global_config as conf

############# ANNOTATION helpers ############################################


def remove_str_flags_from_predictions(df):
    n = df.loc[df[conf.ANNOTATION_COLUMN]=='n'].index
    n_ = df.loc[df[conf.ANNOTATION_COLUMN]=='n '].index
    u = df.loc[df[conf.ANNOTATION_COLUMN]=='u'].index
    u_ = df.loc[df[conf.ANNOTATION_COLUMN]=='u '].index
    c = df.loc[df[conf.ANNOTATION_COLUMN]=='c'].index
    c_ = df.loc[df[conf.ANNOTATION_COLUMN]=='c '].index
    
    clean = df.drop([*n, *u, *c, *n_, *u_, *c_])
    clean.loc[:, conf.ANNOTATION_COLUMN] = (clean[conf.ANNOTATION_COLUMN]
                                            .astype(float))
    return clean


############# TFRECORDS helpers #############################################
def get_annots_for_file(annots: pd.DataFrame, file: str) -> pd.DataFrame:
    """
    Get annotations for a file and sort by the start time of the annotated 
    call. 
    
    Parameters
    ----------
    annots : pd.DataFrame
        global annotations dataframe
    file : str
        file path

    Returns
    -------
    pd.DataFrame
        filtered annotations dataframe
    """
    return annots[annots.filename == file].sort_values('start')


def get_dt_filename(file):
    stem = Path(file).stem
    
    if '_annot_' in stem:
        stem = stem.split('_annot_')[0]
    
    numbs = re.findall('[0-9]+', stem)
    numbs = [n for n in numbs if len(n)%2 == 0] 
    
    i, datetime = 1, ''
    while len(datetime) < 12:
        datetime = ''.join(numbs[-i:])
        i += 1
        
    i = 1
    while 12 <= len(datetime) > 14:
        datetime = datetime[:-i]
        
    for _ in range(2):
        try:
            if len(datetime) == 12:
                file_date = dt.datetime.strptime(datetime, '%y%m%d%H%M%S')
            elif len(datetime) == 14:
                file_date = dt.datetime.strptime(datetime, '%Y%m%d%H%M%S')
        except:
            i = 1
            while  len(datetime) > 12:
                datetime = datetime[:-i]
        
    try:
        # print(file_date)
        return file_date
    except Exception as e:
        print('File date naming not understood.\n', 
              'This will be prevent hourly prediction computation.\n',
              e)
        return 'ERROR'
    
def get_channel(dir):
    if '_CH' in str(dir)[-5:]:
        channel = int(re.findall('[0-9]+', str(dir)[-5:])[0])-1
    else:
        channel = 0
    return channel

def load_audio(file, channel=0, **kwargs) -> np.ndarray:
    """
    Load audio file, print error if file is corrupted. If the sample rate
    specified in the config file is not the same as the downsample sample 
    rate, resample the audio file accordingly. 
    
    Parameters
    ----------
    file : str or pathlib.Path
        file path

    Returns
    -------
    audio_flat: np.ndarray
        audio array
    """       
    try:
        if conf.DOWNSAMPLE_SR and conf.SR != conf.DOWNSAMPLE_SR:
            audio_flat, _ = lb.load(file, sr = conf.DOWNSAMPLE_SR, mono=False,
                                    **kwargs)[channel]
            if len(audio_flat.shape) > 1:
                audio_flat = audio_flat[channel]
            
            audio_flat = lb.resample(audio_flat, orig_sr = conf.DOWNSAMPLE_SR, 
                                    target_sr = conf.SR)
        else:
            audio_flat, _ = lb.load(file, sr = conf.SR, mono=False, 
                                    **kwargs)
            if len(audio_flat.shape) > 1:
                audio_flat = audio_flat[channel]
            
        if len(audio_flat) == 0: return
        return audio_flat
    except:
        print("File is corrputed and can't be loaded.")
        return

def return_windowed_file(file) -> tuple([np.ndarray, np.ndarray]):
    """
    Load audio file and turn the 1D array into a 2D array. The rows length
    corresponds to the window length specified by the config file. The 
    number of columns results from the division of the 1D array length 
    by the context window length. The incomplete remainder is discarded.
    Along with the window, a 1D array of times is returned, corresponding
    to the beginning in seconds of each context window within the original 
    file.

    Parameters
    ----------
    file : str or pathlib.Path
        file path

    Returns
    -------
    audio_arr: np.ndarray
        2D audio array
    times: np.ndarray
        start times of the context windows
    """
    audio = load_audio(file)    
    audio = audio[:len(audio)//conf.CONTEXT_WIN * conf.CONTEXT_WIN]
    audio_arr = audio.reshape([len(audio)//conf.CONTEXT_WIN, 
                               conf.CONTEXT_WIN])
    
    times = np.arange(0, audio_arr.shape[0]*conf.CONTEXT_WIN/conf.SR, 
                      conf.CONTEXT_WIN/conf.SR)
    return audio_arr, times

def cntxt_wndw_arr(annotations: pd.DataFrame, file, 
                   inbetween_noise: bool=True,
                   **kwargs) -> tuple:
    """
    Load an audio file, with the duration given by the time difference
    between the first and last annotation. Iterate through the annotations
    and extract 1D segments from the audio array based on the annotations, 
    the sample rate and the context window length, all specified in the config
    file. The resulting context window is appended to a list, yielding a 2D
    array with context windows corresponding to the annotated sections. 
    
    To be able to find the section in the original audio file, another list
    is filled with all the start times noted in the annotations multiplied
    by the sample rate. The times list contains the beginning time of each
    context window in samples. 

    Finally if the argument 'inbetween_noise' is False, the nosie arrays in between 
    the calls are collected and all 4 arrays are returned.

    Parameters
    ----------
    annotations : pd.DataFrame
        annotations of vocalizations in file
    file : str or pathlib.Path
        file path
    inbetween_noise : bool, defaults to True
        decide if in between noise should be collected as well or, in case
        the argument is True, all the annotations are already all noise, in
        which case no in between noise is returned

    Returns
    -------
    seg_ar: np.ndarray
        segment array containing the 2D audio array
    noise_ar: np.ndarray
        2D audio array of noise
    times_c
        time list for calls
    times_n
        time list for noise
    """
    duration = annotations['end'].iloc[-1] + conf.CONTEXT_WIN/conf.SR
    audio = load_audio(file, duration=duration)
    
    segs, times = [], []
    for _, row in annotations.iterrows(): 
        num_windows = round((row.end-row.start)/(conf.CONTEXT_WIN/conf.SR) - 1)
        num_windows = num_windows or 1
        for i in range(num_windows):# TODO fuer grosse annotationen mehrere fenster erzeugen
            start = row.start + i*(conf.CONTEXT_WIN/conf.SR)
            beg = int(start*conf.SR)
            end = int(start*conf.SR + conf.CONTEXT_WIN)
            
            if len(audio[beg:end]) == conf.CONTEXT_WIN:
                segs.append(audio[beg:end])
                times.append(beg)
            else:
                end = len(audio)
                beg = end - conf.CONTEXT_WIN
                segs.append(audio[beg:end])
                times.append(beg)
                break
        
    segs = np.array(segs, dtype='float32')
    times = np.array(times, dtype='float32')
    
    # TODO docstrings aufraumen
    if len(segs)-len(annotations) < 0:
        annotations = annotations.drop(annotations.index[len(segs)-len(annotations):])
    if not inbetween_noise: # TODO mismatch in lengths, allow longer active learning annots
        seg_ar = np.array(segs[annotations['label']==1], dtype='float32')
        times_c = np.array(times[annotations['label']==1], dtype='float32')
    else:
        seg_ar = segs
        times_c = times
    if inbetween_noise:
        noise_ar, times_n = return_inbetween_noise_arrays(audio, annotations)
    elif len(annotations.loc[annotations['label']==0]) > 0:
        noise_ar = np.array(segs[annotations['label']==0], dtype='float32')
        times_n = np.array(times[annotations['label']==0], dtype='float32')
    else:
        noise_ar, times_n = np.array([]), np.array([])
    
    return seg_ar, noise_ar, times_c, times_n
    
def wins_bet_calls(annotations: pd.DataFrame) -> list:
    """
    Returns a list of ints, corresponding to the number of context windows
    that fit between calls. The indexing is crucial, so that each start time
    is subtracted from the previous end time, thereby yielding the gap length. 

    Parameters
    ----------
    annotations : pd.DataFrame
        annotations

    Returns
    -------
    list
        number of context windows that fit between the start of one and the end of
        the previous annotation
    """
    beg_min_start = annotations.start[1:].values - annotations.end[:-1].values
    return (beg_min_start//(conf.CONTEXT_WIN/conf.SR)).astype(int)

def return_inbetween_noise_arrays(audio: np.ndarray, 
                                  annotations: pd.DataFrame) -> tuple:
    """
    Collect audio arrays based on the gaps between vocalizations. 
    Based on the number of context windows that fit inbetween two
    subsequent annotations, the resulting amount of segments are 
    extracted from the audio file. 
    Again, a list containing the start time of each array is also 
    retrieved. 
    If no entire context window fits between two annotations, no
    noise sample is generated. 
    The resulting 2D noise array is returned along with the times. 

    Parameters
    ----------
    audio : np.ndarray
        flat 1D audio array
    annotations : pd.DataFrame
        annotations of vocalizations

    Returns
    -------
    np.ndarray
        2D audio array of noise 
    times: list
        start times of each context window
    """
    noise_ar, times = list(), list()
    for ind, num_wndws in enumerate(wins_bet_calls(annotations)):
        if num_wndws < 1:
            continue
        
        for window_ind in range(num_wndws):
            beg = int(annotations.end.iloc[ind]*conf.SR) \
                  + conf.CONTEXT_WIN * window_ind
            end = beg + conf.CONTEXT_WIN
            noise_ar.append(audio[beg:end])
            times.append(beg)
    
    return np.array(noise_ar, dtype='float32'), times

def get_train_set_size(tfrec_path):
    if not isinstance(tfrec_path, list):
        tfrec_path = [tfrec_path]
    train_set_size, noise_set_size = 0, 0
    for dataset_dir in tfrec_path:
        try:
            for dic in Path(dataset_dir).glob('**/*dataset*.json'):
                data_dict = json.load(open(dic))
                if 'noise' in str(dic):
                    noise_set_size += data_dict['dataset']['size']['train']
                elif 'train' in data_dict['dataset']['size']:
                    train_set_size += data_dict['dataset']['size']['train']
        except:
            print('No dataset dictionary found, estimating dataset size.'
                'WARNING: This might lead to incorrect learning rates!')
            train_set_size += 5000
            noise_set_size += 100
    return train_set_size, noise_set_size

################ Plotting helpers ###########################################

def get_time(time: float) -> str:
    """
    Return time in readable string format m:s.ms. 

    Parameters
    ----------
    time : float
        time in seconds

    Returns
    -------
    str
        time in minutes:seconds.miliseconds
    """
    return f'{int(time/60)}:{np.mod(time, 60):.1f}s'

################ Model Training helpers #####################################    

def save_model_results(ckpt_dir: str, result: dict):
    """
    Format the results dict so that no error occurrs when saving the json. 

    Parameters
    ----------
    ckpt_dir : str
        checkpoint path
    result : dict
        training results
    """
    result['fbeta'] = [float(n) for n in result['fbeta']]
    result['val_fbeta'] = [float(n) for n in result['val_fbeta']]
    result['fbeta1'] = [float(n) for n in result['fbeta1']]
    result['val_fbeta1'] = [float(n) for n in result['val_fbeta1']]
    with open(f"{ckpt_dir}/results.json", 'w') as f:
        json.dump(result, f)
        
def get_val_labels(val_data: tf.data.Dataset,
                   num_of_samples: int) -> np.ndarray:
    """
    Return all validation set labels. The dataset is batched with the dataset
    size, thus creating one batch from the entire dataset. This batched 
    dataset is then converted to a list and its numpy attribute is returned. 

    Parameters
    ----------
    val_data : tf.data.Dataset
        validation set
    num_of_samples : int
        length of dataset

    Returns
    -------
    np.ndarray
        array of all validation set labels
    """
    return list(val_data.batch(num_of_samples))[0][1].numpy()

############### Model Evaluation helpers ####################################

def print_evaluation(val_data: tf.data.Dataset, 
                     model: tf.keras.Sequential, 
                     batch_size: int):
    """
    Print evaluation results. 

    Parameters
    ----------
    val_data : tf.data.Dataset
        validation data set
    model : tf.keras.Sequential
        keras model
    batch_size : int
        batch size
    """
    model.evaluate(val_data, batch_size = batch_size, verbose =2)

def get_pr_arrays(labels: np.ndarray, preds: np.ndarray, 
                  metric: str, **kwargs) -> np.ndarray:
    """
    Compute Precision or Recall on given set of labels and predictions. 
    Threshold values are created with 0.01 increments. 

    Parameters
    ----------
    labels : np.ndarray
        labels
    preds : np.ndarray
        predictions
    metric : str
        Metric to calculate i.e. Recall or Precision

    Returns
    -------
    np.ndarray
        resulting values
    """
    r = getattr(tf.keras.metrics, metric)(**kwargs)
    r.update_state(labels, preds.reshape(len(preds)))
    return r.result().numpy()

############## Generate Model Annotations helpers ############################

def get_files(*, location: str=f'{conf.GEN_ANNOTS_DIR}', 
              search_str: str='*.wav') -> generator:
    """
    Find all files corresponding to given search string within a specified 
    location. 

    Parameters
    ----------
    location : str, optional
        root directory of files, by default 'generated_annotations/src'
    search_str : str, optional
        search string containing search pattern, for example '*.wav', 
        by default '*.wav'

    Returns
    -------
    generator
        generator object containing pathlib.Path objects of all files fitting 
        the pattern
    """
    folder = Path(location)
    return list(folder.glob(search_str))

def window_data_for_prediction(audio: np.ndarray) -> tf.Tensor:
    """
    Compute predictions based on spectrograms. First the number of context
    windows that fit into the audio array are calculated. The result is an 
    integer unless the last section is reached, in that case the audio is
    zero padded to fit the length of a multiple of the context window length. 
    The array is then zero padded to fit a integer multiple of the context 
    window.

    Parameters
    ----------
    audio : np.ndarray
        1D audio array

    Returns
    -------
    tf.Tensor
        2D audio tensor with shape [context window length, number of windows]
    """
    num = np.ceil(len(audio) / conf.CONTEXT_WIN)
    # zero pad in case the end is reached
    audio = [*audio, *np.zeros([int(num*conf.CONTEXT_WIN - len(audio))])]
    wins = np.array(audio).reshape([int(num), conf.CONTEXT_WIN])
    
    return tf.convert_to_tensor(wins)

def create_Raven_annotation_df(preds: np.ndarray, ind: int) -> pd.DataFrame:
    """
    Create a DataFrame with column names according to the Raven annotation
    format. The DataFrame is then filled with the corresponding values. 
    Beginning and end times for each context window, high and low frequency
    (from config), and the prediction values. Based on the predicted values,
    the sections with predicted labels of less than the threshold are 
    discarded.

    Parameters
    ----------
    preds : np.ndarray
        predictions
    ind : int
        batch of current predictions (in case predictions are more than 
        the specified limitation for predictions)

    Returns
    -------
    pd.DataFrame
        annotation dataframe for current batch, filtered by threshold
    """
    df = pd.DataFrame(columns = ['Begin Time (s)', 'End Time (s)',
                                 'High Freq (Hz)', 'Low Freq (Hz)'])

    df['Begin Time (s)'] = (np.arange(0, len(preds)) * conf.CONTEXT_WIN) \
                            / conf.SR
    df['End Time (s)'] = df['Begin Time (s)'] \
                            + conf.CONTEXT_WIN/conf.SR
                                
    df['Begin Time (s)'] += (ind*conf.PRED_BATCH_SIZE)/conf.SR
    df['End Time (s)'] += (ind*conf.PRED_BATCH_SIZE)/conf.SR
    
    df['High Freq (Hz)'] = conf.ANNOTATION_DF_FMAX
    df['Low Freq (Hz)'] = conf.ANNOTATION_DF_FMIN
    df[conf.ANNOTATION_COLUMN] = preds

    return df.iloc[preds.reshape([len(preds)]) > conf.DEFAULT_THRESH]
    
def create_annotation_df(audio_batches: np.ndarray, 
                         model: tf.keras.Sequential) -> pd.DataFrame:
    """
    Create a annotation dataframe containing all necessary information to
    be imported into a annotation program. The loaded audio batches are 
    iterated over and used to predict labels. All information is then used
    to fill a DataFrame. After having gone through all batches, the index
    column is set to a increasing integers named 'Selection' (convention). 

    Parameters
    ----------
    audio_batches : np.ndarray
        audio batches
    model : tf.keras.Sequential
        model instance to predict values

    Returns
    -------
    pd.DataFrame
        annotation dataframe
    """
    annots = pd.DataFrame()
    for ind, audio in enumerate(audio_batches):
        preds = model.predict(window_data_for_prediction(audio))
        df = create_Raven_annotation_df(preds, ind)
        annots = pd.concat([annots, df], ignore_index=True)
        
    annots.index  = np.arange(1, len(annots)+1)
    annots.index.name = 'Selection'
    return annots

def batch_audio(audio_flat: np.ndarray) -> np.ndarray:
    """
    Divide 1D audio array into batches depending on the config parameter
    pred_batch_size (predictions batch size) i.e. the number of windows
    that are being simultaneously predicted. 

    Parameters
    ----------
    audio_flat : np.ndarray
        1D audio array

    Returns
    -------
    np.ndarray
        batched audio array
    """
    if len(audio_flat) < conf.PRED_BATCH_SIZE:
        audio_batches = [audio_flat]
    else:
        n = conf.PRED_BATCH_SIZE
        audio_batches = [audio_flat[i:i+n] for i in \
                    range(0, len(audio_flat), conf.PRED_BATCH_SIZE)]
    return audio_batches

def get_directory_structure_relative_to_config_path(file):
    return file.relative_to(conf.SOUND_FILES_SOURCE).parent

def get_top_dir_name_if_only_one_parent_dir(file, parent_dirs):
    if str(parent_dirs) == '.':
        parent_dirs = file.parent.stem
    return parent_dirs

def check_top_dir_crit(parent_dirs):
    return Path(parent_dirs).parts[0] != Path(conf.SOUND_FILES_SOURCE).stem

def check_no_subdir_crit(parent_dirs):
    return len(list(Path(parent_dirs).parents)) == 1

def check_top_dir_is_conf_top_dir():
    return not Path(conf.SOUND_FILES_SOURCE).stem == conf.TOP_DIR_NAME

def manage_dir_structure(file):
    parent_dirs = get_directory_structure_relative_to_config_path(file)
    parent_dirs = get_top_dir_name_if_only_one_parent_dir(file, parent_dirs)
    
    bool_top_dir_crit = check_top_dir_crit(parent_dirs)
    bool_no_subdir = check_no_subdir_crit(parent_dirs)
    bool_top_dir_is_conf = check_top_dir_is_conf_top_dir()
    
    if (bool_top_dir_crit and bool_no_subdir) and bool_top_dir_is_conf:
        parent_dirs = (Path(Path(conf.SOUND_FILES_SOURCE).stem)
                        .joinpath(parent_dirs))
    return parent_dirs

def get_top_dir(parent_dirs):
    return str(parent_dirs).split('/')[0]

def gen_annotations(file, model: tf.keras.Model,
                    mod_label: str, time_start: str):
    """
    Load audio file, instantiate model, use it to predict labels, fill a 
    dataframe with the predicted labels as well as necessary information to
    import the annotations in a annotation program (like Raven). Finally the 
    annotations are saved as a single text file in directories corresponding
    to the model checkpoint name within the generated annotations directory. 

    Parameters
    ----------
    file : str or pathlib.Path object
        file path
    model : tf.keras.Model
        tensorflow model
    mod_label : str
        label to clarify which model was used
    time_start : str
        date time string corresponding to the time the annotations were 
        computed
    """
    parent_dirs = manage_dir_structure(file)
            
    channel = get_channel(get_top_dir(parent_dirs))
    
    audio_batches = batch_audio(load_audio(file, channel))
    
    annotation_df = create_annotation_df(audio_batches, model)
    
    save_path = (Path(conf.GEN_ANNOTS_DIR).joinpath(time_start)
                 .joinpath('thresh_0.5')
                 .joinpath(parent_dirs))
    save_path.mkdir(exist_ok=True, parents=True)
    annotation_df.to_csv(save_path
                         .joinpath(f'{file.stem}_annot_{mod_label}.txt'),
                         sep='\t')
    
    return annotation_df

from email import generator
import json
import tensorflow as tf
import numpy as np
import librosa as lb
from pathlib import Path
import collections
import yaml
import pandas as pd

def load_config() -> collections.namedtuple:
    """
    Load configuration file and return config with attributes. 

    Returns
    -------
    collections.namedtuple
        config object
    """
    with open('hbdet/hbdet/config.yml', 'r') as f:
        config = yaml.safe_load(f)
    
    fft_hop = (config['context_win'] - config['stft_frame_len']) \
                // (config['n_freq_bins'] - 1)

    pred_batch_size = config['pred_win_lim'] * config['context_win']                
    config.update({'fft_hop': fft_hop})
    config.update({'pred_batch_size': pred_batch_size})
    Config = collections.namedtuple("Config", list(config.keys()))
    Config.__new__.__defaults__ = (tuple(config.values()))
    return Config()

############# Define Config #################################################

config = load_config()

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

def load_audio(file, **kwargs) -> np.ndarray:
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
        if config.sr == config.downsample_sr:
            audio_flat, _ = lb.load(file, sr = config.downsample_sr, **kwargs)
        else:
            audio_flat, _ = lb.load(file, sr = config.downsample_sr, **kwargs)
            audio_flat = lb.resample(audio_flat, orig_sr = config.downsample_sr, 
                                    target_sr = config.sr)
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
    audio = audio[:len(audio)//config.context_win * config.context_win]
    audio_arr = audio.reshape([len(audio)//config.context_win, 
                               config.context_win])
    
    times = np.arange(0, audio_arr.shape[0]*config.context_win/config.sr, 
                      config.context_win/config.sr)
    return audio_arr, times

def cntxt_wndw_arr(annotations: pd.DataFrame, file, 
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

    Finally the nosie arrays in between the calls are collected and all
    4 arrays are returned.

    Parameters
    ----------
    annotations : pd.DataFrame
        annotations of vocalizations in file
    file : str or pathlib.Path
        file path

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
    duration = annotations['start'].iloc[-1] + config.context_win/config.sr
    audio = load_audio(file, duration=duration)
    
    seg_ar, times_c = [], []
    for _, row in annotations.iterrows():
        beg = int((row.start)*config.sr)
        end = int((row.start)*config.sr + config.context_win)
        
        if len(audio[beg:end]) == config.context_win:
            seg_ar.append(audio[beg:end])
            times_c.append(beg)
        else:
            end = len(audio)
            beg = end - config.context_win
            seg_ar.append(audio[beg:end])
            times_c.append(beg)
            break
        
    seg_ar = np.array(seg_ar, dtype='float32')
    noise_ar, times_n = return_inbetween_noise_arrays(audio, annotations)
    
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
    return (beg_min_start//(config.context_win/config.sr)).astype(int)

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
            beg = int(annotations.end.iloc[ind]*config.sr) \
                  + config.context_win * window_ind
            end = beg + config.context_win
            noise_ar.append(audio[beg:end])
            times.append(beg)
    
    return np.array(noise_ar, dtype='float32'), times

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

def init_model(model_instance: type, 
               checkpoint_dir: str, **kwargs) -> tf.keras.Sequential:
    """
    Initiate model instance, load weights. As the model is trained on 
    spectrogram tensors but will now be used for inference on audio files
    containing continuous audio arrays, the input shape of the model is 
    changed after the model weights have been loaded. 

    Parameters
    ----------
    model_instance : type
        callable class to create model object
    checkpoint_dir : str
        checkpoint path

    Returns
    -------
    tf.keras.Sequential
        the sequential model with pretrained weights
    """
    mod_obj = model_instance(**kwargs)
    mod_obj.load_ckpt(checkpoint_dir)
    mod_obj.change_input_to_array()
    return mod_obj.model

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
                  metric: str) -> np.ndarray:
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
    threshs=np.linspace(0, 1, num=100)[:-1]
    r = getattr(tf.keras.metrics, metric)(thresholds = list(threshs))
    r.update_state(labels, preds.reshape(len(preds)))
    return r.result().numpy()

def get_labels_and_preds(model_instance: type, 
                         training_path: str, 
                         val_data: tf.data.Dataset, 
                         **kwArgs) -> tuple:
    """
    Retrieve labels and predictions of validation set and given model
    checkpoint. 

    Parameters
    ----------
    model_instance : type
        model class
    training_path : str
        path to checkpoint
    val_data : tf.data.Dataset
        validation dataset

    Returns
    -------
    labels: np.ndarray
        labels
    preds: mp.ndarray
        predictions
    """
    model = init_model(model_instance, training_path, **kwArgs)
    preds = model.predict(x = val_data.batch(batch_size=32))
    labels = get_val_labels(val_data, len(preds))
    return labels, preds

############## Generate Model Annotations helpers ############################

def get_files(*, location: str='generated_annotations/src', 
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
    return folder.glob(search_str)

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
    num = np.ceil(len(audio) / config.context_win)
    # zero pad in case the end is reached
    audio = [*audio, *np.zeros([int(num*config.context_win - len(audio))])]
    wins = np.array(audio).reshape([int(num), config.context_win])
    
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

    df['Begin Time (s)'] = (np.arange(0, len(preds)) * config.context_win) \
                            / config.sr
    df['End Time (s)'] = df['Begin Time (s)'] \
                            + config.context_win/config.sr
                                
    df['Begin Time (s)'] += (ind*config.pred_batch_size)/config.sr
    df['End Time (s)'] += (ind*config.pred_batch_size)/config.sr
    
    df['High Freq (Hz)'] = config.fmax
    df['Low Freq (Hz)'] = config.fmin
    df['Prediction/Comments'] = preds

    return df.iloc[preds.reshape([len(preds)]) > config.thresh]
    
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
    if len(audio_flat) < config.pred_batch_size:
        audio_batches = [audio_flat]
    else:
        n = config.pred_batch_size
        audio_batches = [audio_flat[i:i+n] for i in \
                    range(0, len(audio_flat), config.pred_batch_size)]
    return audio_batches

def gen_annotations(file, model_instance: type, training_path: str, 
                         mod_label: str, time_start: str, **kwargs):
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
    model_instance : type
        callable class to retrieve model
    training_path : str
        path to model checkpoint
    mod_label : str
        label to clarify which model was used
    time_start : str
        date time string corresponding to the time the annotations were 
        computed
    """
    audio_batches = batch_audio(load_audio(file))
            
    model = init_model(model_instance, 
                       f'{training_path}/{mod_label}/unfreeze_no-TF', **kwargs)
    annotation_df = create_annotation_df(audio_batches, model)
    
    save_path = Path(f'generated_annotations/{time_start}')
    save_path.mkdir(exist_ok=True, parents=True)

    annotation_df.to_csv(save_path.joinpath(f'{file.stem}_annot_{mod_label}.txt'),
                sep='\t')

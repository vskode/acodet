import os.path
import glob
import re
import datetime as dt
import json
import numpy as np
import librosa as lb
from pathlib import Path
import pandas as pd
from . import global_config as conf

############# ANNOTATION helpers ############################################


def remove_str_flags_from_predictions(df):
    """
    Remove all annotations that were flagged as noise ('n'), undefined ('u')
    or call ('c') and return the remaining annotations with numeric
    prediction values. Every value is checked individually, so the flags are
    found independently of the type of the first value in the column.
    Values that are neither a flag nor a number are converted to NaN.
    """
    if len(df) == 0:
        return df
    flags = df[conf.ANNOTATION_COLUMN].astype(str).str.strip().str.lower()
    clean = df.loc[~flags.isin(["n", "u", "c"])].copy()
    clean[conf.ANNOTATION_COLUMN] = pd.to_numeric(
        clean[conf.ANNOTATION_COLUMN], errors="coerce"
    )
    return clean


def get_bacpipe_model_constants(model_name):
    """
    Return the sample rate and the segment length (in samples) of a bacpipe
    model without loading the model weights.

    Parameters
    ----------
    model_name : str
        name of a model supported by bacpipe

    Returns
    -------
    tuple
        (sample rate, segment length in samples)
    """
    try:
        # available from bacpipe 1.3.5, also resolves models that inherit
        # their constants from the module of another model
        from bacpipe.core.audio_processor import _get_model_constants
    except ImportError:
        _get_model_constants = None

    if _get_model_constants is not None:
        _, sr, segment_length = _get_model_constants(model_name)
        return sr, segment_length

    import importlib

    model_module = importlib.import_module(
        f"bacpipe.model_pipelines.feature_extractors.{model_name}"
    )
    return model_module.SAMPLE_RATE, model_module.LENGTH_IN_SAMPLES


def get_dt_filename(file):
    if isinstance(file, Path):
        stem = file.stem
    else:
        stem = file

    if "_annot_" in stem:
        stem = stem.split("_annot_")[0]

    numbs = re.findall("[0-9]+", stem)
    numbs = [n for n in numbs if len(n) % 2 == 0]

    i, datetime = 1, ""
    while len(datetime) < 12:
        if i > 1000:
            raise NameError(
                """Time stamp Error: time stamp in filename 
                doesn't fit any known pattern. If you would
                like to predict this file anyway, please
                choose predefined settings option 1.
                """
            )
        datetime = "".join(numbs[-i:])
        i += 1

    i = 1
    while 12 <= len(datetime) > 14:
        datetime = datetime[:-i]

    for _ in range(2):
        try:
            if len(datetime) == 12:
                file_date = dt.datetime.strptime(datetime, "%y%m%d%H%M%S")
            elif len(datetime) == 14:
                file_date = dt.datetime.strptime(datetime, "%Y%m%d%H%M%S")
        except:
            i = 1
            while len(datetime) > 12:
                datetime = datetime[:-i]

    try:
        # print(file_date)
        return file_date
    except Exception as e:
        print(
            "File date naming not understood.\n",
            "This will prevent hourly prediction computation.\n",
            e,
        )
        return "ERROR"


def get_channel(dir):
    if "_CH" in str(dir)[-5:]:
        channel = int(re.findall("[0-9]+", str(dir)[-5:])[0]) - 1
    else:
        channel = 0
    return channel


def load_audio(file, 
               channel=0, 
               save_downsampled_files=conf.SAVE_DOWNSAMPLED_FILES, 
               **kwargs) -> np.ndarray:
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
        if conf.MODELCLASSNAME == 'BacpipeModel':
            sr, conf.CONTEXT_WIN = get_bacpipe_model_constants(conf.MODEL_NAME)
            conf.SR = sr
            with open(file, 'rb') as f:
                audio_flat, _ = lb.load(
                    f, sr=sr, mono=True, **kwargs
                )
                
        elif conf.DOWNSAMPLE_SR and conf.SR != conf.DOWNSAMPLE_SR:
            with open(file, "rb") as f:
                audio_flat, _ = lb.load(
                    f, sr=conf.DOWNSAMPLE_SR, mono=False, **kwargs
                )[channel]
            if len(audio_flat.shape) > 1:
                audio_flat = audio_flat[channel]

            audio_flat = lb.resample(
                audio_flat, orig_sr=conf.DOWNSAMPLE_SR, target_sr=conf.SR
            )
        else:
            with open(file, "rb") as f:
                audio_flat, _ = lb.load(f, sr=conf.SR, mono=False, **kwargs)
            if len(audio_flat.shape) > 1:
                audio_flat = audio_flat[channel]
        if save_downsampled_files:
            from  scipy.io import wavfile 
            path = (
                Path(conf.SOUND_FILES_SOURCE)
                .parent
                .joinpath(f'resampled_{Path(conf.SOUND_FILES_SOURCE).stem}_{conf.SR}Hz')
                .joinpath(Path(file).relative_to(conf.SOUND_FILES_SOURCE))
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w") as f:
                wavfile.write(path, conf.SR, audio_flat)
        if len(audio_flat) == 0:
            return
        return audio_flat
    except Exception as e:
        print("File is corrputed and can't be loaded.", e)
        return


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
    return f"{int(time/60)}:{np.mod(time, 60):.1f}s"


################ Model Training helpers #####################################

def extract_value(supp):
    if isinstance(supp, float):
        return supp
    if isinstance(supp, np.ndarray):
        # I would just say supp.item() here but this is legacy code and I'm unsure if it's always a single value.
        return float(supp[0])
    else:
        # Assume it's a torch.Tensor here. Raises exception if not.
        return float(supp.numpy()[0])


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
    # Some metrics might be numpy or torch arrays. Need to fix that before serializing.
    # First extract items into list for safe modification of map while iterating.
    for k, v in list(result.items()):
        if "support" in k or "f_1" in k:
            result[k] = [extract_value(n) for n in v]

    with open(f"{ckpt_dir}/results.json", "w") as f:
        json.dump(result, f)


############## Generate Model Annotations helpers ############################


def get_files(
        location: str = f"{conf.GEN_ANNOTS_DIR}", search_str: str = "**/*.wav", follow_symlinks=True, regex=False
) -> list:
    """
    Find all files corresponding to given search string within a specified location.
    Searches subfolders recursively if the search string includes '**'.

    Parameters
    ----------
    location : str, optional
        root directory of files, by default the configured 'generated_annotations_folder'
    search_str : str, optional
        search string containing search pattern, for example '*.wav', by default '**/*.wav'
    follow_symlinks : bool, optional
        whether to follow symlinks, default True. If True, then `regex` must be False.
    regex : bool, optional
        whether to use a regex pattern for matching. If True, then `follow_symlinks` must be False.

    Returns
    -------
    list
        list containing pathlib.Path objects of all files fitting the pattern
    """
    if follow_symlinks:
        # Note: We don't use Path.rglob here because it can't follow symlinks for Python <3.13. This is why we can't
        #       support regext. The user promises not to have a symlink loop in this directory.
        if regex:
            raise NotImplementedError(f"Cannot use regex patterns when searching recursively through symlinks.")
        return [Path(path_str) for path_str in glob.glob(os.path.join(location, search_str), recursive=True)]
    else:
        return list(Path(location).rglob(search_str))


def window_data_for_prediction(audio: np.ndarray):
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
    import tensorflow as tf
    num = np.ceil(len(audio) / conf.CONTEXT_WIN)
    # zero pad in case the end is reached
    audio = [*audio, *np.zeros([int(num * conf.CONTEXT_WIN - len(audio))])]
    wins = np.array(audio).reshape([int(num), conf.CONTEXT_WIN])

    return tf.convert_to_tensor(wins)


def create_Raven_annotation_df(preds: np.ndarray) -> pd.DataFrame:
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

    Returns
    -------
    pd.DataFrame
        annotation dataframe for current batch, filtered by threshold
    """
    df = pd.DataFrame(
        columns=[
            "Begin Time (s)",
            "End Time (s)",
            "High Freq (Hz)",
            "Low Freq (Hz)",
        ]
    )
    df['Selection'] = range(1, len(preds)+1)
    df['View'] = 'Spectrogram 1'
    df['Channel'] = '1'
    df = df.set_index('Selection')
    df["Begin Time (s)"] = (
        np.arange(0, len(preds)) * conf.CONTEXT_WIN
    ) / conf.SR
    df["End Time (s)"] = df["Begin Time (s)"] + conf.CONTEXT_WIN / conf.SR

    if conf.SCALE_FMAX_WITH_PREDICTIONS:
        df["High Freq (Hz)"] = (
            (preds-conf.DEFAULT_THRESH) / (1-conf.DEFAULT_THRESH)*(conf.SR / 2)
            )
    else:
        df["High Freq (Hz)"] = (conf.SR / 2)
    df["Low Freq (Hz)"] = conf.ANNOTATION_DF_FMIN
    df[conf.ANNOTATION_COLUMN] = preds

    return df.iloc[preds.reshape([len(preds)]) > conf.DEFAULT_THRESH]


def run_inference(
    file, 
    model,
    channel=0,
    callbacks: None = None,
    **kwargs,
) -> pd.DataFrame:
    predictions = []
    if 'predict' in dir(model):
        audio = load_audio(file, channel)
        if audio is None:
            raise ImportError(
                f"The audio file `{str(file)}` cannot be loaded. Check if file has "
                "one of the supported endings "
                "(wav, mp3, flac, etc.)) and is not empty."
            )
        audio_batches = batch_audio(audio)
        for ind, batch in enumerate(audio_batches):
            if callbacks is not None and ind == 0:
                callbacks = callbacks(**kwargs)
            preds = model(
                window_data_for_prediction(batch),
                training=False,
                callbacks=callbacks
            )
            # collect the predictions of every batch, not only the last one
            predictions.extend(preds)
    elif conf.MODELCLASSNAME == 'BacpipeModel':
        if conf.BOOL_LIN_CLFIER:
            import torch
            with torch.no_grad():
                predictions = torch.sigmoid(model(file)).squeeze()
            try:
                predictions = np.array(predictions)
            except:
                predictions = np.array(predictions.cpu().detach())
            if len(predictions.shape) == 0:
                predictions = np.array([predictions])
                
        elif conf.MODEL_NAME == 'google_whale':
            import tensorflow as tf
            predictions = model(file, **kwargs)
            predictions = tf.sigmoid(predictions)
        else:
            predictions = model(file, **kwargs)
    elif conf.MODELCLASSNAME == 'TorchModel':
        import torch
        audio = load_audio(file, channel)
        audio_batches = batch_audio(audio)
        torch_batches = []
        for batch in audio_batches:
            torch_batches.append(torch.tensor(window_data_for_prediction(batch).numpy(), dtype=torch.float32))
        torch_batches = torch.vstack(torch_batches).unsqueeze(1)
        
        with torch.inference_mode():
            predictions = torch.sigmoid(model(torch_batches))
            # one row per window, also if the file only has one window
            predictions = predictions.reshape(len(torch_batches), -1).squeeze(-1)

    return np.array(predictions)


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
        audio_batches = [
            audio_flat[i : i + n]
            for i in range(0, len(audio_flat), conf.PRED_BATCH_SIZE)
        ]
    return audio_batches


def get_directory_structure_relative_to_config_path(file):
    return file.relative_to(conf.SOUND_FILES_SOURCE).parent


def get_top_dir_name_if_only_one_parent_dir(file, parent_dirs):
    if str(parent_dirs) == ".":
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
        parent_dirs = Path(Path(conf.SOUND_FILES_SOURCE).stem).joinpath(
            parent_dirs
        )
    return parent_dirs


def get_top_dir(parent_dirs):
    return Path(parent_dirs).parts[0]


def gen_annotations(
    file,
    model,
    mod_label: str,
    timestamp_foldername: str,
    **kwargs,
):
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
    timestamp_foldername : str
        date time string foldername corresponding to the time the annotations were
        computed
    """
    parent_dirs = manage_dir_structure(file)

    channel = get_channel(get_top_dir(parent_dirs))
    predictions = run_inference(file, model, channel, **kwargs)

    save_path_func = lambda x: (
        Path(conf.GEN_ANNOTS_DIR)
        .joinpath(timestamp_foldername)
        .joinpath(conf.THRESH_LABEL)
        .joinpath(x)
        .joinpath(parent_dirs)
    )
    if 'multi_datasets_annot' in kwargs and kwargs['multi_datasets_annot']:
        dataset_dir = Path(parent_dirs).parts[0]
        save_path_func = lambda x: (
            Path(conf.GEN_ANNOTS_DIR)
            .joinpath(timestamp_foldername)
            .joinpath(conf.THRESH_LABEL)
            .joinpath(dataset_dir)
            .joinpath(x)
            .joinpath(parent_dirs.relative_to(dataset_dir))
            )

    if len(predictions.squeeze().shape) > 1:
        
        annotation_df = save_multiclass_dfs(file, model, save_path_func, 
                                            mod_label, predictions)
        
    else:
        annotation_df = create_Raven_annotation_df(predictions)
        
        save_path = save_path_func('.')
        
        save_path.mkdir(exist_ok=True, parents=True)
        annotation_df.to_csv(
            save_path.joinpath(f"{file.stem}_annot_{mod_label}.selection.table.txt"), sep="\t"
        )

    return annotation_df


def save_multiclass_dfs(file, model, save_path_func, mod_label, predictions):
    # since bacpipe 1.3.5 make_classification_dict is a static method of the
    # Classifier class and the classes are an attribute of the bacpipe model
    from bacpipe.model_pipelines.runner import Classifier

    df_preds = Classifier.make_classification_dict(
        np.asarray(predictions),
        model.embedder.model.classes,
        conf.DEFAULT_THRESH,
        )
    head = df_preds.pop('head')
    filtered_labels = list(df_preds.keys())
    pred_arr = np.zeros([len(filtered_labels), head['Time bins in this file']])
    for row, label in enumerate(filtered_labels):
        pred_arr[row, df_preds[label]['time_bins_exceeding_threshold']] = df_preds[label]['classifier_predictions']
    
    
    if len(filtered_labels) == 0:
        save_path = save_path_func('.')
        save_path.parent.mkdir(exist_ok=True, parents=True)
        return create_Raven_annotation_df(np.array([]))
    
    elif len(filtered_labels) > 1:
        save_combined_and_multiclass_dfs(pred_arr, 
                                         filtered_labels, 
                                         save_path_func, 
                                         file, mod_label)

        
    for preds, label in zip(pred_arr, filtered_labels):
        
        annotation_df = create_Raven_annotation_df(preds)
        
        save_path = save_path_func(label.replace('/', '_'))
        
        save_path.mkdir(exist_ok=True, parents=True)
        annotation_df.to_csv(
            save_path.joinpath(f"{file.stem}_annot_{mod_label}_{label.replace('/', '_')}.txt"), sep="\t"
        )
    
    return annotation_df

def save_combined_and_multiclass_dfs(predictions, 
                                     labels, 
                                     save_path_func, 
                                     file, mod_label):
    # predictions has the shape (number of classes, number of time bins)
    max_preds = np.max(predictions, axis=0)
    cls_idxes = np.argmax(predictions, axis=0)
    _, tmp_idxes = np.where(predictions > conf.DEFAULT_THRESH)
    single_maxes = np.unique(tmp_idxes)
    
    save_paths = dict()
    save_paths['multiclass'] = save_path_func('multiclass')
    
    multiclass_df = create_Raven_annotation_df(max_preds)
    multiclass_df[conf.ANNOTATION_COLUMN] = [
        f"{predictions[cls_idx, tmp_idx]:.4f}__{labels[cls_idx]}"
        for cls_idx, tmp_idx in 
        zip(cls_idxes[single_maxes], single_maxes)
        ]
    save_paths['multiclass'].mkdir(exist_ok=True, parents=True)
    multiclass_df.to_csv(
        save_paths['multiclass'].joinpath(f"{file.stem}_annot_{mod_label}_multiclass.txt"), sep="\t"
    )
    
    save_paths['All_Combined'] = save_path_func('All_Combined')
    
    all_combined_df = create_Raven_annotation_df(max_preds)
    all_combined_df.pop(conf.ANNOTATION_COLUMN)
    
    # The index of the annotation dataframe is the 1-based Raven selection
    # number, i.e. the time bin index + 1. Select the per class predictions
    # of exactly these time bins, so every row gets its own predictions.
    time_bin_idxs = all_combined_df.index.values - 1
    results_df = pd.DataFrame(
        predictions.T[time_bin_idxs],
        index=all_combined_df.index,
        columns=labels,
    )
    all_combined_df = pd.concat([all_combined_df, results_df], axis=1)
        
    save_paths['All_Combined'].mkdir(exist_ok=True, parents=True)
    all_combined_df.to_csv(
        save_paths['All_Combined'].joinpath(f"{file.stem}_annot_{mod_label}_combined.txt"), sep="\t"
    )

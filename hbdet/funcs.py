import json
import tensorflow as tf
import numpy as np
import librosa as lb
from pathlib import Path
import collections
import yaml
import pandas as pd

############# Define Config #################################################

with open('hbdet/hbdet/config.yml', 'r') as f:
  config = yaml.safe_load(f)

Config = collections.namedtuple("Config", [
    "sr",
    "downsample_sr",
    "context_win",
    "fmin",
    "fmax",
    "pred_win_lim",
    "thresh"
])
Config.__new__.__defaults__ = (config['sr'],
                               config['downsample_sr'],
                               config['cntxt_wn_sz'],
                               config['fmin'],
                               config['fmax'],
                               config['pred_win_lim'],
                               config['thresh'])
config = Config()

############# TFRECORDS helpers #############################################
def get_annots_for_file(annots, file):
    return annots[annots.filename == file].sort_values('start')

def load_audio(file, **kwargs):
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

def return_windowed_file(file):
    audio = load_audio(file)    
    audio = audio[:len(audio)//config.context_win * config.context_win]
    audio_arr = audio.reshape([len(audio)//config.context_win, 
                               config.context_win])
    
    times = np.arange(0, audio_arr.shape[0]*config.context_win/config.sr, 
                      config.context_win/config.sr)
    return audio_arr, times

def cntxt_wndw_arr(annotations, file, **kwargs):
    duration = annotations['start'].iloc[-1] + config.context_win/config.sr
    audio = load_audio(file, duration=duration)
    
    seg_ar, times_c = list(), list()
    for index, row in annotations.iterrows():
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
            
    noise_ar, times_n = return_inbetween_noise_arrays(audio, annotations,
                                                      config.sr, 
                                                      config.context_win)
    return seg_ar, noise_ar, times_c, times_n
    
def return_inbetween_noise_arrays(audio, annotations):
    num_wndws_btw_end_start = ( (
        annotations.start[1:].values-annotations.end[:-1].values
        ) // (config.context_win/config.sr) ).astype(int)
    noise_ar, times = list(), list()
    for ind, num_wndws in enumerate(num_wndws_btw_end_start):
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

def get_time(time):
    return f'{int(time/60)}:{np.mod(time, 60):.1f}s'

################ Model Training helpers #####################################    

def save_model_results(ckpt_dir, result):
    result['fbeta'] = [float(n) for n in result['fbeta']]
    result['val_fbeta'] = [float(n) for n in result['val_fbeta']]
    result['fbeta1'] = [float(n) for n in result['fbeta1']]
    result['val_fbeta1'] = [float(n) for n in result['val_fbeta1']]
    with open(f"{ckpt_dir}/results.json", 'w') as f:
        json.dump(result, f)
        
def get_val_labels(val_data, num_of_samples):
    return list(val_data.batch(num_of_samples))[0][1].numpy()

############### Model Evaluation helpers ####################################

def init_model(model_instance, checkpoint_dir, **kwargs):
    mod_obj = model_instance(**kwargs)
    mod_obj.load_ckpt(checkpoint_dir)
    mod_obj.change_input_to_array()
    return mod_obj.model

def print_evaluation(val_data, model, batch_size):
    return model.evaluate(val_data, batch_size = batch_size, verbose =2)
    
def predict_values(val_data, model):
    return model.predict(x = val_data.batch(batch_size=32))

def get_pr_arrays(labels, preds, metric):
    threshs=np.linspace(0, 1, num=100)[:-1]
    r = getattr(tf.keras.metrics, metric)(thresholds = list(threshs))
    r.update_state(labels, preds.reshape(len(preds)))
    result = r.result().numpy()    
    return result

def get_labels_and_preds(model_instance, training_path, val_data, **kwArgs):
    model = init_model(model_instance, training_path, **kwArgs)
    preds = predict_values(val_data, model)
    labels = get_val_labels(val_data, len(preds))
    return labels, preds

############## Generate Model Annotations helpers ############################

def get_files(search_str):
    folder = Path('generated_annotations/src')
    fold_glob = folder.glob(search_str)
    return fold_glob

def split_audio_into_sections(audio, pred_len_samps):
    if len(audio) > pred_len_samps:
        n = pred_len_samps
        audio_secs = [audio[i:i+n] for i in range(0, len(audio), n)]
    else:
        audio_secs = [audio]
    return audio_secs

def compute_predictions(audio, model):
    num = np.ceil(len(audio) / config.context_win)
    # zero pad in case the end is reached
    audio = [*audio, *np.zeros([int(num*config.context_win - len(audio))])]

    wins = np.array(audio).reshape([int(num), config.context_win])

    return model.predict(x = tf.convert_to_tensor(wins))

def create_Raven_annotation_df(preds, ind, pred_len_samps):
    df = pd.DataFrame(columns = ['Begin Time (s)', 'End Time (s)',
                                 'High Freq (Hz)', 'Low Freq (Hz)'])

    df['Begin Time (s)'] = (np.arange(0, len(preds)) 
                                    * config.context_win) \
                                    / config.sr
    df['End Time (s)'] = df['Begin Time (s)'] + \
                                config.context_win/config.sr
                                
    df['Begin Time (s)'] += (ind*pred_len_samps)/config.sr
    df['End Time (s)'] += (ind*pred_len_samps)/config.sr
    
    df['High Freq (Hz)'] = config.fmax
    df['Low Freq (Hz)'] = config.fmin
    df['Prediction/Comments'] = preds
    return df
    
def create_annotation_df(audio_secs, model, pred_len_samps):
    annots = pd.DataFrame()
    for ind, audio in enumerate(audio_secs):
        preds = compute_predictions(audio, model)        
        df = create_Raven_annotation_df(preds, ind, pred_len_samps)
        df = df.iloc[preds.reshape([len(preds)]) > config.thresh]

        annots = pd.concat([annots, df], ignore_index=True)
    annots.index  = np.arange(1, len(annots)+1)
    annots.index.name = 'Selection'
    return annots

def gen_raven_annotation(file, model, mod_label, time_start):
    audio_flat = load_audio(file)
    pred_len_samps = config.pred_win_lim * config.context_win
    
    audio_secs = split_audio_into_sections(audio_flat, pred_len_samps)
        
    annotation_df = create_annotation_df(audio_secs, model, pred_len_samps)
    
    save_path = Path(f'generated_annotations/{time_start}')
    save_path.mkdir(exist_ok=True, parents=True)

    annotation_df.to_csv(save_path.joinpath(f'{file.stem}_annot_{mod_label}.txt'),
                sep='\t')

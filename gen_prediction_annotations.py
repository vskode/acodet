import yaml
import numpy as np
import pandas as pd
import librosa as lb
import tensorflow as tf
from pathlib import Path
from hbdet.google_funcs import GoogleMod

with open('hbdet/hbdet/config.yml', 'r') as f:
    config = yaml.safe_load(f)

params = config['preproc']

# file = Path('Daten/OneDrive_1_1-24-2022/channelA_2021-03-18_01-00-05.wav')
# file = Path('/media/vincent/Expansion/Tolsta/2020/D8_Tolsta_wavs/335564853.200222210624.wav')
file = Path('/home/vincent/Code/MA/generated_annotations/335564853.200222210624.wav')
# file = Path('/home/vincent/Code/MA/Daten/googles_train_data/Hawaii_K_02_080506_183545.d20.x.wav')
# file = Path('/media/vincent/Expansion/NOAA/NRS08_20162018_new20220711/NRS08_20160414_233332.wav')

# Create a new model instance
G = GoogleMod(config['model'])
model = G.model

# model.load_weights('trainings/2022-09-24_19/unfreeze_15/cp-0092.ckpt')
model.load_weights('models/google_humpback_model')


def gen_raven_annotation(file, model, resample = True):
    if not resample:
        audio_flat, _ = lb.load(file, sr = params['sr'])
    else:
        audio_flat, _ = lb.load(file, sr = 2000)
        audio_flat = lb.resample(audio_flat, orig_sr = 2000, 
                                 target_sr = params['sr'])

    pred_len_samps = config['model']['pred_win_lim'] * params['cntxt_wn_sz']
    if len(audio_flat) > pred_len_samps:
        n = pred_len_samps
        audio_secs = [audio_flat[i:i+n] for i in range(0, len(audio_flat), n)]
    else:
        audio_secs = [audio_flat]
        
    annots = pd.DataFrame()
    
    for ind, audio in enumerate(audio_secs):
        num = np.ceil(len(audio) / params['cntxt_wn_sz'])
        audio = [*audio, 
                 *np.zeros([int(num*params['cntxt_wn_sz'] - len(audio))]) ]

        wins = np.array(audio).reshape([int(num), params['cntxt_wn_sz']])

        preds = model.predict(x = tf.convert_to_tensor(wins))

        annots_sec = pd.DataFrame(columns = ['Begin Time (s)', 'End Time (s)',
                                        'High Freq (Hz)', 'Low Freq (Hz)'])

        annots_sec['Begin Time (s)'] = (np.arange(0, len(preds)) * 
                                    params['cntxt_wn_sz'])/params['sr']
        annots_sec['End Time (s)'] = annots_sec['Begin Time (s)'] + \
                                    params['cntxt_wn_sz']/params['sr']
                                    
        annots_sec['Begin Time (s)'] += (ind*pred_len_samps)/params['sr']
        annots_sec['End Time (s)'] += (ind*pred_len_samps)/params['sr']
        
        annots_sec['High Freq (Hz)'] = params['fmax']
        annots_sec['Low Freq (Hz)'] = params['fmin']
        annots_sec['Prediction value'] = preds

        annots_sec = annots_sec.iloc[
            preds.reshape([len(preds)])>config['model']['thresh']
            ]

        annots = pd.concat([annots, annots_sec], ignore_index=True)


    annots.index  = np.arange(1, len(annots)+1)
    annots.index.name = 'Selection'

    annots.to_csv(f'generated_annotations/{file.stem}_annot_untrained.txt', 
                  sep='\t')
    
    
if __name__ == '__main__':
    gen_raven_annotation(file, model, resample=True)
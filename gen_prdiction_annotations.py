from utils.google_funcs import GoogleMod
import librosa as lb
import numpy as np
import pandas as pd
import tensorflow as tf
import yaml
from pathlib import Path

with open('humpzam/config.yml', 'r') as f:
    config = yaml.safe_load(f)

params = config['preproc']

# file = Path('Daten/OneDrive_1_1-24-2022/channelA_2021-03-18_01-00-05.wav')
file = Path('/media/vincent/Expansion/Tolsta/2020/D8_Tolsta_wavs/335564853.200222210624.wav')

# Create a new model instance
G = GoogleMod(config['model'])
model = G.model

# model.load_weights('trainings/unfreeze_25_lr_exp/cp-0035.ckpt')
model.load_weights('models/google_humpback_model')


def gen_raven_annotation(file, model, resample = True):
    if not resample:
        audio, _ = lb.load(file, sr = params['sr'])
    else:
        audio, _ = lb.load(file, sr = 2000)
        audio = lb.resample(audio, orig_sr = 2000, target_sr = params['sr'])

    num = np.ceil(len(audio) / params['cntxt_wn_hop'])
    audio = [*audio, *np.zeros([int(num*params['cntxt_wn_hop'] - len(audio))]) ]

    wins = np.array(audio).reshape([int(num), params['cntxt_wn_hop']])

    preds = model.predict(x = tf.convert_to_tensor(wins))

    annots = pd.DataFrame(columns = ['Begin Time (s)', 'End Time (s)', 
                                    'High Freq (Hz)', 'Low Freq (Hz)'])

    annots['Begin Time (s)'] = (np.arange(0, len(preds)) * 
                                params['cntxt_wn_hop'])/params['sr']
    annots['End Time (s)'] = annots['Begin Time (s)'] + \
                                params['cntxt_wn_hop']/params['sr']
    annots['High Freq (Hz)'] = params['fmax']
    annots['Low Freq (Hz)'] = params['fmin']

    annots = annots.iloc[preds.reshape([len(preds)])>config['model']['thresh']]

    annots.index  = np.arange(1, len(annots)+1)
    annots.index.name = 'Selection'

    annots.to_csv(f'generated_annotations/{file.stem}_annot_untrained.txt', sep='\t')
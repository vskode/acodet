import yaml
import time
import numpy as np
import pandas as pd
import librosa as lb
import tensorflow as tf
from pathlib import Path
from hbdet.google_funcs import GoogleMod

with open('hbdet/hbdet/config.yml', 'r') as f:
    config = yaml.safe_load(f)

def get_files():
    # file = Path('Daten/OneDrive_1_1-24-2022/channelA_2021-03-18_01-00-05.wav')
    # file = Path('/media/vincent/Expansion/Tolsta/2020/D8_Tolsta_wavs/335564853.200222210624.wav')
    # file = Path('/home/vincent/Code/MA/generated_annotations/335564853.200222210624.wav')
    # file = Path('/mnt/c/Documents and Settings/sa01vk/Documents/NOAA/MELLINGER_NOVA-SCOTIA_200508_EmrldN/EmrldN-00000005-050821-153834.wav')
    # file = Path('/home/vincent/Code/MA/Daten/googles_train_data/Hawaii_K_02_080506_183545.d20.x.wav')
    # file = Path('/media/vincent/Expansion/NOAA/NRS08_20162018_new20220711/NRS08_20160414_233332.wav')
    # folder = Path('/mnt/e/MA/SAMOSAS_VINCENT')
    folder = Path('generated_annotations/src')
    # fold_glob = folder.glob('*_2kHz_March-April/2021-03-13_*/**/*.wav')
    fold_glob = folder.glob('*2009*')
    return fold_glob

# Create a new model instance
def get_model(model_checkpoint, untrained = False):
    G = GoogleMod()
    model = G.model

    if untrained:
        model.load_weights('models/google_humpback_model')
        mod_label = 'untrained'
    else:
        model_checkpoint = model_checkpoint
        mod_label = str(Path(model_checkpoint).parent.parent.stem
                        + Path(model_checkpoint).parent.stem.replace('unfreeze_', '_u')
                        + Path(model_checkpoint).stem.replace('cp-0', '_cp'))
        model.load_weights(f'trainings/{model_checkpoint}')
    return model, mod_label

def gen_raven_annotation(file, model, mod_label, time_start, resample = True):
    try:
        if not resample:
            audio_flat, _ = lb.load(file, sr = config['sr'])
        else:
            audio_flat, _ = lb.load(file, sr = 2000)
            audio_flat = lb.resample(audio_flat, orig_sr = 2000, 
                                    target_sr = config['sr'])
        if len(audio_flat) == 0: return
    except:
        print("File is corrputed and can't be loaded.")
        return

    pred_len_samps = config['pred_win_lim'] * config['cntxt_wn_sz']
    if len(audio_flat) > pred_len_samps:
        n = pred_len_samps
        audio_secs = [audio_flat[i:i+n] for i in range(0, len(audio_flat), n)]
    else:
        audio_secs = [audio_flat]
        
    annots = pd.DataFrame()
    
    for ind, audio in enumerate(audio_secs):
        num = np.ceil(len(audio) / config['cntxt_wn_sz'])
        audio = [*audio, 
                 *np.zeros([int(num*config['cntxt_wn_sz'] - len(audio))]) ]

        wins = np.array(audio).reshape([int(num), config['cntxt_wn_sz']])

        preds = model.predict(x = tf.convert_to_tensor(wins))

        annots_sec = pd.DataFrame(columns = ['Begin Time (s)', 'End Time (s)',
                                        'High Freq (Hz)', 'Low Freq (Hz)'])

        annots_sec['Begin Time (s)'] = (np.arange(0, len(preds)) * 
                                    config['cntxt_wn_sz'])/config['sr']
        annots_sec['End Time (s)'] = annots_sec['Begin Time (s)'] + \
                                    config['cntxt_wn_sz']/config['sr']
                                    
        annots_sec['Begin Time (s)'] += (ind*pred_len_samps)/config['sr']
        annots_sec['End Time (s)'] += (ind*pred_len_samps)/config['sr']
        
        annots_sec['High Freq (Hz)'] = config['fmax']
        annots_sec['Low Freq (Hz)'] = config['fmin']
        annots_sec['Prediction value'] = preds

        annots_sec = annots_sec.iloc[
            preds.reshape([len(preds)])>config['thresh']
            ]

        annots = pd.concat([annots, annots_sec], ignore_index=True)


    annots.index  = np.arange(1, len(annots)+1)
    annots.index.name = 'Selection'

    save_path = Path(f'generated_annotations/{time_start}/{file.parent.parent.stem}/'
        f'{file.parent.stem}')
    save_path.mkdir(exist_ok=True, parents=True)

    annots.to_csv(save_path.joinpath(f'{file.stem}_annot_{mod_label}.txt'),
                sep='\t')
    
    
if __name__ == '__main__':
    time_start = time.strftime('%Y-%m-%d_%H', time.gmtime())
    files = get_files()
    for file in files:
        model, label = get_model('2022-09-26_16/unfreeze_15/cp-0094.ckpt', 
                        untrained = False)
        gen_raven_annotation(file, model, label, time_start, resample=True)
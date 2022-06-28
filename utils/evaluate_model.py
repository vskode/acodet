#%%
import pandas as pd
import numpy as np
from pathlib import Path
import os

from funcs import *
from google_funcs import *
from hump_spot_funcs import *


if not 'Daten' in os.listdir():
    os.chdir('../..')

params = {
    "sr" : 10000,
    "cntxt_wn_hop": 39124,
    "cntxt_wn_sz": 39124,
    "fft_window_length" : 2**11,
    "n_freq_bins": 2**8,
    "freq_cmpr": 'linear',
    "fmin":  50,
    "fmax":  1000,
    "nr_noise_samples": 100,
    "sequence_len" : 39124,
    "fft_hop": 300,
}
mtrxs = {'mse':0, 'rmse':0, 'mae':0}
mtrxs.update({f'{m}_t': 0 for m in mtrxs})
mtrxs.update({f'{m}_n': 0 for m in mtrxs})
mtrxs.update({'bin_cross_entr': 0, 'bin_cross_entr_n': 0})

preds = {'signal':[], 'thresh': [], 'noise': [], 'thresh_noise': []}



# durations = pd.read_csv('Daten/file_durations.csv')

thresh = .25
annots = pd.read_csv('Daten/ket_annot_file_exists.csv')
files = np.unique(annots.filename)
df_mse = pd.DataFrame()
Gog = GooglePreds(params)
Hum = HumpSpot(params)


for mod, model_name in zip((Gog, Hum), ('google', 'benoit')):
    for file in files[7:]:
        
        file_annots = get_annots_for_file(annots, file)
        
        mod.load_data(file, file_annots)
        
        preds['signal'] = mod.pred()
        preds['thresh'] = (preds['signal'] > thresh).astype(int)
        
        mtrxs['bin_cross_entr'] = mod.eval()
        if model_name == 'google':
            y_test, y_noise = mod.return_test_data()
        
        # if len(y_noise) > 0:
        #     preds['noise'] = mod.pred(noise=True)
        #     mtrxs['bin_cross_entr_n'] = mod.eval(noise=True)
        #     preds['thresh_noise'] = (preds['noise'] > thresh).astype(int)
        # else:
        #     preds['noise'] = 0
        #     preds['thresh_noise'] = 0
        

        mtrxs['mse'], mtrxs['rmse'], mtrxs['mae'] = get_metrics(preds['signal'], 
                                                                y_test)
        # mtrxs['mse_n'], mtrxs['rmse_n'], mtrxs['mae_n'] = get_metrics(preds['noise'], 
        #                                                             y_noise)
        mtrxs['mse_t'], mtrxs['rmse_t'], mtrxs['mae_t'] = get_metrics(preds['thresh'], 
                                                                    y_test)
        # mtrxs['mse_t_n'], mtrxs['rmse_t_n'], mtrxs['mae_t_n'] = get_metrics(preds['thresh_noise'],
        #                                                                     y_noise)
        
        # np.random.seed(33)
        # rndm = np.random.randint(20)
        # plot_and_save_spectrogram(y_test[rndm], Path(file).stem, 
        #                         preds['signal'][rndm], 
        #                         start = file_annots['start'].iloc[rndm],
        #                         **params)
        # if len(y_noise) > rndm:
        #     plot_and_save_spectrogram(y_noise[rndm], Path(file).stem, 
        #                             preds['noise'][rndm], 
        #                             start = 0, noise=True, **params)
        
        print(f'{model_name} model mse: {mtrxs["mse"]*100:.2f}%')
        print(f'{model_name} model mse_noise: {mtrxs["mse_n"]*100:.2f}%')
        # df_mse = df_mse.append({"file": file, 
        #                 "number_of_annots": len(y_test),
        #                 "quality_of_recording": get_quality_of_recording(file),
        #                 "number_of_annots_noise": len(y_noise),
        #                 **{f"{metric}({model_name})" : \
        #                     mtrxs[metric] for metric in mtrxs}},
        #                     ignore_index = True)
        mtrx_dict = {f"{metric}({model_name})" : mtrxs[metric] for metric in mtrxs}
        
        df_mse = df_mse.append({"file": file, 
                            "number_of_annots": len(y_test),
                            "quality_of_recording": get_quality_of_recording(file),
                            "number_of_annots_noise": len(y_noise), 
                            **mtrx_dict}, 
                            ignore_index=True)
    
    df_mse.to_csv(f'{model_name}_model_evaluation.csv')

# %%

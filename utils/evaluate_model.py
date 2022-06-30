#%%
import pandas as pd
import numpy as np
from pathlib import Path
import os

from funcs import *
from google_funcs import *
from hump_spot_funcs import *
from ketos_narw_funcs import *


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
preds, mtrxs = get_dicts()


thresh = .25
specs = True
annots = pd.read_csv('Daten/ket_annot_file_exists.csv')
files = np.unique(annots.filename)
df_mse = pd.DataFrame()

available_models = (NarwMod, GoogleMod, BenoitMod)


for mod_iter, model in enumerate(available_models[:2]):
    np.random.seed(33)
    model = model(params)
    model_name = type(model).__name__
    
    for i, file in enumerate(files):
        
        file_annots = get_annots_for_file(annots, file)
        file_annots.start -= params['cntxt_wn_sz'] / params['sr'] / 2
        
        x_test, x_noise = return_cntxt_wndw_arr(file_annots, file, **params)
        y_test = np.ones(len(x_test), dtype = 'float32')
        y_noise = np.zeros(len(x_noise), dtype = 'float32')
        
        model.load_data(file, file_annots, y_test, y_noise, x_test, x_noise)
        
        preds['call'] = model.pred()
        preds['thresh'] = (preds['call'] > thresh).astype(int)
        mtrxs['bin_cross_entr'] = model.eval()
        
        if len(y_noise) > 0:
            preds['noise'] = model.pred(noise=True)
            preds['thresh_noise'] = (preds['noise'] > thresh).astype(int)
            mtrxs['bin_cross_entr_n'] = model.eval(noise=True)
        else:
            preds['noise'] = 0
            preds['thresh_noise'] = 0
        
        mtrxs = collect_all_metrics(mtrxs, preds, y_test, y_noise)
        
        if specs:
            generate_spectrograms(x_test, x_noise, y_test, y_noise, model, 
                                  file, file_annots, mod_iter, **params)

        print(f'file_{i}({model_name})_mse: {mtrxs["mse"]*100:.2f}%')
        print(f'file_{i}({model_name})_mse_noise: {mtrxs["mse_n"]*100:.2f}%')

        df = pd.DataFrame({"file": file, 
                            "number_of_annots": len(y_test),
                        "quality_of_recording": get_quality_of_recording(file),
                            "number_of_annots_noise": len(y_noise), 
            **{f"{metric}({model_name})" : mtrxs[metric] for metric in mtrxs}},
                          index = [0])
        
        df_mse = pd.concat([df_mse, df], axis = 0)
        df = None
    
    df_mse.to_csv(f'{model_name}_model_evaluation.csv')
    model = None

# %%

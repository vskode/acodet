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
}

durations = pd.read_csv('Daten/file_durations.csv')
params["fft_hop"] = params['fft_window_length']//2

thresh = .25

annots = pd.read_csv('Daten/ket_annot_file_exists.csv')
files = np.unique(annots.filename)
model = load_google_sequential()#load_google_hub()

df_mse = pd.DataFrame()

#%%
model = load_HUMP_SPOT()
params["sequence_len"] = durations.duration[0]*params['sr'] #cntxt_wn_sz*20 hat auch 20 ergebnisse geliefert ... ?
data_loader = create_data_loader_benoit(files[0], 
                                        offset = 3, 
                                        **params)
preds = get_HOMP_SPOT_preds(model, data_loader)

#%%

for file in files:
    annotations = annots[annots.filename == file].sort_values('start')
    
    seg_ar, noise_ar = return_cntxt_wndw_arr(annotations, file, **params)

    model_name = 'google'
    
    y_test = return_labels(annotations, file)[:len(seg_ar)]
    y_noise = np.zeros([len(noise_ar)])
    
    predictions = model.predict(seg_ar).T[0]
    if len(noise_ar) > 0:
        predictions_noise = model.predict(noise_ar).T[0]

    mse, rmse, mae = get_metrics(predictions, y_test)
    mse_noise, rmse_noise, mae_noise = get_metrics(predictions_noise, y_noise)
    
    predictions_thresh = (predictions > thresh).astype(int)
    mse_thresh, rmse_thresh, mae_thresh = get_metrics(predictions_thresh, y_test)
    
    np.random.seed(33)
    rndm = np.random.randint(20)
    plot_and_save_spectrogram(seg_ar[rndm], Path(file).stem, 
                              predictions[rndm], 
                              start = annotations['start'].iloc[rndm],
                              **params)
    if len(noise_ar) > rndm:
        plot_and_save_spectrogram(noise_ar[rndm], Path(file).stem, 
                                predictions_noise[rndm], 
                                start = 0, noise=True, **params)
    
    print(f'{model_name} model mse: {mse*100:.2f}%')
    print(f'{model_name} model mse_noise: {mse_noise*100:.2f}%')
    df_mse = df_mse.append({"file": file, 
                    "number_of_annots": len(y_test),
                    "quality_of_recording": get_quality_of_recording(file),
                    f"mse({model_name})" : mse, 
                    f"rmse({model_name})" : rmse, 
                    f"mae({model_name})" : mae, 
                    f"mse_thresh({model_name})" : mse_thresh, 
                    f"rmse_thresh({model_name})" : rmse_thresh, 
                    f"mae_thresh({model_name})" : mae_thresh, 
                "number_of_annots_noise": len(y_noise),
                f"mse_noise({model_name})" : mse_noise, 
                f"rmse_noise({model_name})" : rmse_noise,
                f"mae_noise({model_name})" : mae_noise}, 
                           ignore_index = True)
    
    df_mse.to_csv('google_model_evaluation.csv')

# %%

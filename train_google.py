#%%
import os
import pandas as pd
import numpy as np
from pathlib import Path
import tensorflow as tf
import time
import json

from utils.tfrec import get_dataset, check_random_spectrogram
from utils.google_funcs import GoogleMod

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
    "lr": 1e-3,
}

TFRECORDS_DIR = 'Daten/tfrecords*s_shift'
AUTOTUNE = tf.data.AUTOTUNE


#%% init
# time.sleep(2000)
batch_size = 32
epochs = 50
# steps_per_epoch = 400
rep = 1


train_files = tf.io.gfile.glob(f"{TFRECORDS_DIR}/train/*.tfrec")
train_data = get_dataset(train_files, batch_size, AUTOTUNE = AUTOTUNE)

test_files = tf.io.gfile.glob(f"{TFRECORDS_DIR}/test/*.tfrec")
test_data = get_dataset(test_files, batch_size, AUTOTUNE = AUTOTUNE)

train_data = train_data.shuffle(50)
check_random_spectrogram(train_files, dataset_size = 20000)

unfreezes = [2, 9, 15, 25]
# lrs = np.linspace(1.4e-3, 2.5e-3, 5)
lr = tf.keras.optimizers.schedules.ExponentialDecay(1e-2,
                                        decay_steps = 1000,
                                        decay_rate = 0.7,
                                        staircase = True)
for unfreeze in unfreezes:
    # for lr in lrs:
        
    params['lr'] = lr

    G = GoogleMod(params)
    model = G.model
    for layer in model.layers[:-unfreeze]:
        layer.trainable = False

    # model.load_weights(f'trainings/unfreeze_{unfreeze}_lr_exp/cp-0035.ckpt')

    #%% define training
    checkpoint_path = f"trainings/unfreeze_{unfreeze}_" + \
                        f"lr_exp_all_data" + \
                        "/cp-{epoch:04d}.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    # Create a callback that saves the model's weights every 5 epochs
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path, 
        verbose=1, 
        save_weights_only=True,
        save_freq=epochs*300)

    # Save the weights using the `checkpoint_path` format
    model.save_weights(checkpoint_path.format(epoch=0))

    # Train the model with the new callback
    hist = model.fit(train_data, 
            epochs = epochs, 
            # steps_per_epoch=steps_per_epoch, 
            validation_data = test_data,
            callbacks=[cp_callback]
            )

    result = hist.history
    avg_val_loss = np.mean(result['val_loss'][-int(len(result)/3):])

    pd.DataFrame().to_csv(f"{checkpoint_dir}/res_2nd_run_{avg_val_loss:.2f}.csv")

    with open(f"{checkpoint_dir}/results_2nd_run.json", 'w') as f:
        json.dump(result, f)


#%%


    


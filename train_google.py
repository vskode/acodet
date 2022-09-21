import os
import yaml
import time
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from hbdet.utils.google_funcs import GoogleMod
from keras.utils.layer_utils import count_params
from hbdet.utils.model_funcs import plot_model_results
from hbdet.utils.tfrec import get_dataset, check_random_spectrogram

with open('humpzam/config.yml', 'r') as f:
    config = yaml.safe_load(f)



TFRECORDS_DIR = 'Daten/tfrecords*s_shift'
AUTOTUNE = tf.data.AUTOTUNE


batch_size = 32
epochs = 20

load_weights = False
steps_per_epoch = False
rep = 1
good_file_size = 432
poor_file_size = 236
num_of_shifts = 5
init_lr = 1e-2
final_lr = 1e-5

unfreezes = [11, 15, 18, 5]


# for TFRECORDS_DIR, poor_file_size in (('Daten/tfrecords*s_shift', 236), ('Daten/tfrecords_*s_shift', 0)):
info_text = f"""Model run INFO:

model: untrained model 
dataset: good and poor data, 5 shifts from 0s - 2s
lr: new lr settings
comments:

VARS:
batch_size      = {batch_size}
epochs          = {epochs}
load_weights    = {load_weights}
steps_per_epoch = {steps_per_epoch}
rep             = {rep}
good_file_size  = {good_file_size}
poor_file_size  = {poor_file_size}
num_of_shifts   = {num_of_shifts}
init_lr         = {init_lr}
final_lr        = {final_lr}
unfreezes       = {unfreezes}
"""




#############################################################################
#############################  RUN  #########################################
#############################################################################



time_start = time.strftime('%Y-%m-%d_%H', time.gmtime())
dataset_size = (good_file_size + poor_file_size)*num_of_shifts

train_files = tf.io.gfile.glob(f"{TFRECORDS_DIR}/train/*.tfrec")
train_data = get_dataset(train_files, batch_size, AUTOTUNE = AUTOTUNE)

test_files = tf.io.gfile.glob(f"{TFRECORDS_DIR}/test/*.tfrec")
test_data = get_dataset(test_files, batch_size, AUTOTUNE = AUTOTUNE)

train_data = train_data.shuffle(50)
check_random_spectrogram(train_files, dataset_size = dataset_size*batch_size)

lr = tf.keras.optimizers.schedules.ExponentialDecay(init_lr,
                                decay_steps = dataset_size,
                                decay_rate = (final_lr/init_lr)**(1/epochs),
                                staircase = True)

for ind, unfreeze in enumerate(unfreezes):
        
    config['model']['lr'] = lr

    G = GoogleMod(config['model'])
    model = G.model
    for layer in model.layers[:-unfreeze]:
        layer.trainable = False

    if load_weights:
        model.load_weights(
            f'trainings/2022-09-16_10/unfreeze_{unfreeze}/cp-0032.ckpt')

    checkpoint_path = f"trainings/{time_start}/unfreeze_{unfreeze}" + \
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
    
    if ind == 0:
        with open(f'trainings/{time_start}/training_info.txt', 'w') as f:
            f.write(info_text)

    # Train the model with the new callback
    hist = model.fit(train_data, 
            epochs = epochs, 
            # steps_per_epoch=steps_per_epoch, 
            validation_data = test_data,
            callbacks=[cp_callback]
            )

    result = hist.history

    pd.DataFrame().to_csv(f"{checkpoint_dir}/trainable_"
                        f"{count_params(model.trainable_weights):.0f}.csv")

    with open(f"{checkpoint_dir}/results.json", 'w') as f:
        json.dump(result, f)

plot_model_results(f'{time_start}')


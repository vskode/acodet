import os
import yaml
import time
# time.sleep(3600*26)
import json
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
from hbdet.humpback_model_dir import front_end
from hbdet.google_funcs import GoogleMod
from hbdet.plotting import plot_model_results
from keras.utils.layer_utils import count_params
from evaluate_Gmodel import create_and_save_figure
from hbdet.tfrec import get_dataset
from hbdet.funcs import save_rndm_spectrogram

from hbdet.augmentation import CropAndFill

with open('hbdet/hbdet/config.yml', 'r') as f:
    config = yaml.safe_load(f)

TFRECORDS_DIR = 'Daten/Datasets/ScotWest_v1/tfrecords_0s*'
AUTOTUNE = tf.data.AUTOTUNE

batch_size = 32
epochs = 50

load_weights = False
steps_per_epoch = False
rep = 1
good_file_size = 370
poor_file_size = 0
num_of_shifts = 7
data_description = '{}, {} x 0.5s shifts, pretrained'
init_lr = 1e-3
final_lr = 1e-6
pre_blocks = 9

unfreezes = ['no-TF', 15, 5, 19]
data_description = data_description.format(Path(TFRECORDS_DIR).parent.stem, 
                                           num_of_shifts)

info_text = f"""Model run INFO:

model: untrained model 
dataset: {data_description}
lr: new lr settings
comments:

VARS:
data_path       = {TFRECORDS_DIR}
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
preproc blocks  = {pre_blocks}
"""


#############################################################################
#############################  RUN  #########################################
#############################################################################
h, w = 64, 128

crop = tf.keras.Sequential([
    CropAndFill(h, w, seed = 100)
    # tf.keras.layers.RandomRotation(0.2)
])

spec = tf.keras.Sequential([
    tf.keras.layers.Input([39124]),
    tf.keras.layers.Lambda(lambda t: tf.expand_dims(t, -1)),
    front_end.MelSpectrogram()
])

def prepare(ds, augments=3, shuffle=False, time_aug=False):
    ds = ds.batch(batch_size)
    # create specs from audio arrays
    ds = ds.map(lambda x, y: (spec(x), y), num_parallel_calls=AUTOTUNE)
    
    if time_aug:
        ds_augs = []
        for i in range(augments):
            ds_augs.append(ds.map(lambda x, y: (crop(x, training=True), y), 
                    num_parallel_calls=AUTOTUNE))
        for a in ds_augs:
            ds = ds.concatenate(a)
    
    if shuffle:
        ds = ds.shuffle(1000)
        
    return ds.prefetch(buffer_size=AUTOTUNE)


time_start = time.strftime('%Y-%m-%d_%H', time.gmtime())
dataset_size = (good_file_size + poor_file_size)*num_of_shifts

train_files = tf.io.gfile.glob(f"{TFRECORDS_DIR}/train/*.tfrec")
train_data = get_dataset(train_files, batch_size, AUTOTUNE = AUTOTUNE)
train_data = prepare(train_data, shuffle=True, time_aug=True)

test_files = tf.io.gfile.glob(f"{TFRECORDS_DIR}/test/*.tfrec")
test_data = get_dataset(test_files, batch_size, AUTOTUNE = AUTOTUNE)
test_data = prepare(test_data)

save_rndm_spectrogram(train_data)
save_rndm_spectrogram(test_data)

lr = tf.keras.optimizers.schedules.ExponentialDecay(init_lr,
                                decay_steps = dataset_size,
                                decay_rate = (final_lr/init_lr)**(1/epochs),
                                staircase = True)

for ind, unfreeze in enumerate(unfreezes):
        
    config['model']['lr'] = lr
    if unfreeze == 'no-TF':
        config['model']['load_g_ckpt'] = False

    G = GoogleMod(config['model'])
    model = G.model
    if not unfreeze == 'no-TF':
        for layer in model.layers[pre_blocks:-unfreeze]:
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
        monitor = 'val_loss',
        mode = 'min',
        save_best_only = True, 
        verbose=1, 
        save_weights_only=True,
        save_freq='epoch')

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

plot_model_results(time_start, data = data_description, lr_begin = init_lr,
                    lr_end = final_lr)
create_and_save_figure(TFRECORDS_DIR, batch_size, time_start,
                        data = data_description)

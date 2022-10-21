import os
import time
time.sleep(3600*18)
from pathlib import Path
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from functools import partial

from hbdet.humpback_model_dir import front_end
from hbdet.funcs import save_model_results
from hbdet.google_funcs import GoogleMod
from hbdet.plot_utils import plot_model_results
from keras.utils.layer_utils import count_params
from evaluate_Gmodel import create_and_save_figure
from hbdet.tfrec import get_dataset
from hbdet.plot_utils import plot_sample_spectrograms

import hbdet.augmentation as aug


TFRECORDS_DIR = 'Daten/Datasets/ScotWest_v1/tfrecords_0s_shift'
AUTOTUNE = tf.data.AUTOTUNE

batch_size = 32
epochs = 60

load_weights = False
load_g_weights = False
steps_per_epoch = False
rep = 1
good_file_size = 370
poor_file_size = 0
num_of_shifts = 5
data_description = '{}, {} x time shifts'
init_lr = 1e-3
final_lr = 1e-6
pre_blocks = 9
f_score_beta = 0.5
f_score_thresh = 0.5

unfreezes = ['no-TF', 15, 5, 19]
data_description = data_description.format(Path(TFRECORDS_DIR).parent.stem, 
                                           num_of_shifts)

info_text = f"""Model run INFO:

model: untrained model 
dataset: {data_description}
lr: new lr settings
comments: 10 khz; time shift and first mixup implementation included

VARS:
data_path       = {TFRECORDS_DIR}
batch_size      = {batch_size}
epochs          = {epochs}
load_weights    = {load_weights}
steps_per_epoch = {steps_per_epoch}
f_score_beta    = {f_score_beta}
f_score_thresh  = {f_score_thresh}
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

time_start = time.strftime('%Y-%m-%d_%H', time.gmtime())
Path(f'trainings/{time_start}').mkdir(exist_ok=True)
dataset_size = (good_file_size + poor_file_size)*num_of_shifts
seed = np.random.randint(100)

noise_files = tf.io.gfile.glob(f"{TFRECORDS_DIR}/noise/*.tfrec")
noise_data = get_dataset(noise_files, batch_size, AUTOTUNE = AUTOTUNE)
noise_data = aug.make_spec_tensor(noise_data)
mix_up = tf.keras.Sequential([aug.MixCallAndNoise(noise_data=noise_data)])

train_files = tf.io.gfile.glob(f"{TFRECORDS_DIR}/train/*.tfrec")
train_data = get_dataset(train_files, batch_size, AUTOTUNE = AUTOTUNE)
train_data = aug.make_spec_tensor(train_data)
time_aug_data = list(zip(aug.augment(train_data, augments = num_of_shifts, 
                        aug_func=aug.time_shift),
                         ['time_shift']*num_of_shifts ))

mixup_aug_data = list(zip(aug.augment(train_data, augments = 1, 
                        aug_func=mix_up),
                          ['mix_up']*1 ))

mixup_aug_data += list(zip(aug.augment(time_aug_data[0][0], augments = 1, 
                        aug_func=mix_up),
                           ['mix_up']*1 ))
augmented_data = [*time_aug_data, *mixup_aug_data, (noise_data, 'noise')]

plot_train_aug_spec = partial(plot_sample_spectrograms, ds_size = good_file_size)
plot_train_aug_spec(train_data, dir = time_start, name = 'train', 
                         seed=seed)
for i, (augmentation, aug_name) in enumerate(augmented_data):
    plot_train_aug_spec(augmentation, dir = time_start, 
                            name=f'augment_{i}-{aug_name}', seed=seed)
    
train_data = aug.prepare(train_data, batch_size, shuffle=True, 
                     shuffle_buffer=dataset_size//2, 
                     augmented_data=np.array(augmented_data)[:,0])

test_files = tf.io.gfile.glob(f"{TFRECORDS_DIR}/test/*.tfrec")
test_data = get_dataset(test_files, batch_size, AUTOTUNE = AUTOTUNE)
test_data = aug.make_spec_tensor(test_data)
plot_sample_spectrograms(test_data, dir = time_start, name = 'test')
test_data = aug.prepare(test_data, batch_size)

open(f'trainings/{time_start}/training_info.txt', 'w').write(info_text)
lr = tf.keras.optimizers.schedules.ExponentialDecay(init_lr,
                                decay_steps = dataset_size,
                                decay_rate = (final_lr/init_lr)**(1/epochs),
                                staircase = True)

for ind, unfreeze in enumerate(unfreezes):
    
    if unfreeze == 'no-TF':
        load_g_ckpt = False
    else:
        load_g_ckpt = True

    model = GoogleMod(load_g_ckpt=load_g_ckpt).model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate = lr),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics = [tf.keras.metrics.BinaryAccuracy(),
                    tf.keras.metrics.Precision(),
                    tf.keras.metrics.Recall(),
                    tfa.metrics.FBetaScore(num_classes=1,
                                            beta=f_score_beta,
                                            threshold=f_score_thresh,
                                            name='fbeta'),           
        ]
    )
        
    if not unfreeze == 'no-TF':
        for layer in model.layers[pre_blocks:-unfreeze]:
            layer.trainable = False
            
    if load_weights:
        model.load_weights(
            f'trainings/2022-10-20_13/unfreeze_{unfreeze}/cp-last.ckpt')

    checkpoint_path = f"trainings/{time_start}/unfreeze_{unfreeze}" + \
                        "/cp-last.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    
    # Create a callback that saves the model's weights every 5 epochs
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path, 
        mode = 'min',
        verbose=1, 
        save_weights_only=True,
        save_freq='epoch')

    # Save the weights using the `checkpoint_path` format
    model.save_weights(checkpoint_path.format(epoch=0))

    # Train the model with the new callback
    hist = model.fit(train_data, 
            epochs = epochs, 
            # steps_per_epoch=steps_per_epoch, 
            validation_data = test_data,
            callbacks=[cp_callback])
    result = hist.history
    save_model_results(checkpoint_dir, result)

plot_model_results(time_start, data = data_description, lr_begin = init_lr,
                    lr_end = final_lr)
create_and_save_figure(TFRECORDS_DIR, batch_size, time_start,
                        data = data_description)

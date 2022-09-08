#%%
import pandas as pd
import numpy as np
from pathlib import Path
import os

from utils.funcs import *
from utils.tfrecord_funcs import *
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

TFRECORDS_DIR = 'Daten/tfrecords_0s_shift'


def prepare_sample(features):
    return features["audio"], features["label"]

def get_dataset(filenames, batch_size):
    dataset = (
        tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTOTUNE)
        .map(parse_tfrecord_fn, num_parallel_calls=AUTOTUNE)
        .map(prepare_sample, num_parallel_calls=AUTOTUNE)
        # .shuffle(batch_size)
        # .batch(batch_size)
    )
    return dataset

    

#%% init

filenames = tf.io.gfile.glob(f"{TFRECORDS_DIR}/*.tfrec")
filenames_trial_run = filenames
batch_size = 80
epochs = 10
steps_per_epoch = 50
AUTOTUNE = tf.data.AUTOTUNE

data = get_dataset(filenames_trial_run, batch_size)
num_arrays = len(list(data))
rep = 1

data = data.shuffle(int(num_arrays//2))
data = data.repeat(rep)

train_data = data.take(int (num_arrays*rep * 0.7) )
test_data = data.skip(int (num_arrays*rep * 0.7) )

val_data = test_data.skip(int (num_arrays*rep * 0.15) )
test_data = test_data.take(int (num_arrays*rep * 0.15) )

# train_data = train_data.shuffle(int (num_arrays * 0.7) )
train_data = train_data.batch(batch_size)
train_data = train_data.prefetch(AUTOTUNE)

test_data = test_data.batch(batch_size)
test_data = test_data.prefetch(AUTOTUNE)
# train_data = train_data.shuffle(int (num_arrays * 0.7))

unfreezes = np.arange(5, 20)
lrs = np.linspace(1.4e-4, 1.8e-4, 15)

for unfreeze in unfreezes:
    for lr in [0.0014]:
        
        # unfreeze = 4
        params['lr'] = lr

        #%% freeze layers
        G = GoogleMod(params)
        model = G.model
        for layer in model.layers[:-unfreeze]:
            layer.trainable = False

        #%% define training

        # Include the epoch in the file name (uses `str.format`)
        checkpoint_path = f"trainings/unfreeze_{unfreeze}_" + \
                            f"lr_{params['lr']:.3f}" + \
                            "/cp-{epoch:04d}.ckpt"
        checkpoint_dir = os.path.dirname(checkpoint_path)


        # Create a callback that saves the model's weights every 5 epochs
        cp_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path, 
            verbose=1, 
            save_weights_only=True,
            save_freq=epochs*100)

        # Save the weights using the `checkpoint_path` format
        model.save_weights(checkpoint_path.format(epoch=0))

        # Train the model with the new callback
        hist = model.fit(train_data, 
                epochs = epochs, 
                # steps_per_epoch=steps_per_epoch, 
                validation_data = test_data,
                callbacks=[cp_callback],
                )

        result = hist.history['val_loss']
        avg_val_loss = np.mean(result[-int(len(result)/3):])

        pd.DataFrame().to_csv(f"{checkpoint_dir}/res_{avg_val_loss:.2f}.csv")

#%% Evaluate the model


# Create a new model instance
# model = GoogleMod(params).model


# data = get_dataset(filenames, batch_size)

# dset = tf.data.TFRecordDataset(f"{TFRECORDS_DIR}/file_01.tfrec").map(parse_tfrecord_fn)
# ll = list()
# for features in dset:
#     ll.append(list(features["audio"].numpy()))
# e = model.predict(np.array(ll, dtype='float32'))


# model.evaluate(data, batch_size = batch_size, verbose =2)
# model.predict(x = get_dataset(filenames, batch_size))

# trainingsverlauf

# 149/149 [==============================] - 74s 445ms/step - loss: 0.3708 - val_loss: 2.1994
# Epoch 2/10
# 149/149 [==============================] - 63s 421ms/step - loss: 0.2869 - val_loss: 0.6403
# Epoch 3/10
# 149/149 [==============================] - 63s 420ms/step - loss: 0.2330 - val_loss: 1.7759
# Epoch 4/10
# 149/149 [==============================] - 63s 421ms/step - loss: 0.1891 - val_loss: 3.6567
# Epoch 5/10
# 149/149 [==============================] - 63s 422ms/step - loss: 0.1667 - val_loss: 5.8410
# Epoch 6/10
# 149/149 [==============================] - 63s 421ms/step - loss: 0.1362 - val_loss: 0.9721
# Epoch 7/10
# 105/149 [====================>.........] - ETA: 17s - loss: 0.1036 
# Epoch 7: saving model to trainings/unfreeze_5_lr_0.001/cp-0007.ckpt
# 149/149 [==============================] - 63s 425ms/step - loss: 0.1072 - val_loss: 7.3036
# Epoch 8/10
# 149/149 [==============================] - 63s 421ms/step - loss: 0.0778 - val_loss: 13.6656
# Epoch 9/10
# 149/149 [==============================] - 63s 421ms/step - loss: 0.0603 - val_loss: 12.7539
# Epoch 10/10
# 149/149 [==============================] - 63s 421ms/step - loss: 0.0553 - val_loss: 12.4619


# 2022-09-08 11:20:19.137210: W tensorflow/core/util/tensor_slice_reader.cc:96] Could not open 
# models/google_humpback_model: FAILED_PRECONDITION: models/google_humpback_model; Is a 
# directory: perhaps your file is in a different file format and you need to use a different 
# restore operator?

# Epoch 1/10
# 149/149 [==============================] - 66s 424ms/step - loss: 0.3600 - val_loss: 3.3810
# Epoch 2/10
# 149/149 [==============================] - 63s 421ms/step - loss: 0.3021 - val_loss: 1.4117
# Epoch 3/10
# 149/149 [==============================] - 63s 421ms/step - loss: 0.2573 - val_loss: 2.5419
# Epoch 4/10
# 149/149 [==============================] - 63s 421ms/step - loss: 0.1970 - val_loss: 0.5688
# Epoch 5/10
# 149/149 [==============================] - 63s 421ms/step - loss: 0.1758 - val_loss: 3.8134
# Epoch 6/10
# 149/149 [==============================] - 63s 420ms/step - loss: 0.1313 - val_loss: 0.7013
# Epoch 7/10
# 105/149 [====================>.........] - ETA: 17s - loss: 0.0965 
# Epoch 7: saving model to trainings/unfreeze_6_lr_0.001/cp-0007.ckpt
# 149/149 [==============================] - 63s 422ms/step - loss: 0.1105 - val_loss: 0.9925
# Epoch 8/10
# 149/149 [==============================] - 63s 421ms/step - loss: 0.0892 - val_loss: 5.8304
# Epoch 9/10
#  16/149 [==>...........................] - ETA: 53s - loss: 0.0957
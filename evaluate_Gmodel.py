import numpy as np
import tensorflow as tf
from hbdet.tfrec import get_dataset
from hbdet.google_funcs import GoogleMod
import matplotlib.pyplot as plt

#%% Evaluate the model

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
AUTOTUNE = tf.data.AUTOTUNE

#%% init
# time.sleep(2000)
batch_size = 32
epochs = 50
steps_per_epoch = 200
rep = 1
num_of_samples = 400
unfreeze = 15
train_date = '2022-09-21_08'


train_files = tf.io.gfile.glob(f"{TFRECORDS_DIR}/train/*.tfrec")
train_data = get_dataset(train_files, batch_size, AUTOTUNE = AUTOTUNE)
# num_arrays = len(list(train_data))
# check_random_spectrogram(train_files, dataset_size = 13000)

test_files = tf.io.gfile.glob(f"{TFRECORDS_DIR}/test/*.tfrec")
test_data = get_dataset(test_files, batch_size, AUTOTUNE = AUTOTUNE)

val_data = test_data.unbatch().take(num_of_samples)
labels = list(val_data.batch(num_of_samples))[0][1].numpy()
# train_data = train_data.shuffle(50)

# Create a new model instance
G = GoogleMod(params)
model = G.model

model.load_weights(f'trainings/{train_date}/unfreeze_{unfreeze}/cp-0020.ckpt')
# model.load_weights('models/google_humpback_model')

print(model.evaluate(train_data.take(1), batch_size = batch_size, verbose =2))
preds = model.predict(x = val_data.batch(32))


threshs = np.linspace(0, 1, 100)
r = tf.keras.metrics.Recall(thresholds = list(threshs))
r.update_state(labels, preds.reshape(len(preds)))
recall_res = r.result().numpy()

p = tf.keras.metrics.Precision(thresholds = list(threshs))
p.update_state(labels, preds.reshape(len(preds)))
precision_res = p.result().numpy()

plt.figure()
plt.title('Precision and Recall Curve')
plt.plot(threshs, recall_res, label = 'recall')
plt.plot(threshs, precision_res, label = 'precision')
plt.xlabel('threshold')
plt.legend()
plt.savefig(f'trainings/{train_date}/unfreeze_{unfreeze}/pr_curve.png')
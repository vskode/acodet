from utils.google_funcs import GoogleMod
from utils.tfrec import get_dataset
import tensorflow as tf

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
TFRECORDS_DIR = 'Daten/tfrecords_2s_shift'
AUTOTUNE = tf.data.AUTOTUNE


#%% init
# time.sleep(2000)
batch_size = 32
epochs = 50
steps_per_epoch = 200
rep = 1


train_files = tf.io.gfile.glob(f"{TFRECORDS_DIR}/train/*.tfrec")
train_data = get_dataset(train_files, batch_size, AUTOTUNE = AUTOTUNE)
# num_arrays = len(list(train_data))
# check_random_spectrogram(train_files, dataset_size = 13000)

test_files = tf.io.gfile.glob(f"{TFRECORDS_DIR}/test/*.tfrec")
test_data = get_dataset(test_files, batch_size, AUTOTUNE = AUTOTUNE)

# train_data = train_data.shuffle(50)

# Create a new model instance
G = GoogleMod(params)
model = G.model

model.load_weights('trainings/unfreeze_25_lr_exp/cp-0035.ckpt')
# model.load_weights('models/google_humpback_model')

print(model.evaluate(train_data, batch_size = batch_size, verbose =2))
# print(model.predict(x = test_data))


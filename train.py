import os
import time
from pathlib import Path
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

from hbdet.funcs import save_model_results, load_config, get_train_set_size
from hbdet.google_funcs import GoogleMod
from hbdet.plot_utils import plot_model_results, create_and_save_figure
from hbdet.tfrec import TFRECORDS_DIR, run_data_pipeline, prepare
from hbdet.plot_utils import plot_pre_training_spectrograms
from hbdet.augmentation import run_augment_pipeline

NUMBER_OF_TRAININGS = 4
config = load_config()
TFRECORDS_DIR = ['Daten/Datasets/ScotWest_v5_2khz', 
                 'Daten/Datasets/ScotWest_v4_2khz']
AUTOTUNE = tf.data.AUTOTUNE

batch_size = 32
epochs = 25

load_weights = False
load_g_weights = False
steps_per_epoch = False
n_time_augs = [0, 0, 3, 0]
n_mixup_augs = [0, 3, 0, 0]
data_description = TFRECORDS_DIR
init_lr = [1e-3, 1e-3, 1e-3, 1e-6]
final_lr = [1e-6, 1e-6, 1e-6, 1e-9]
pre_blocks = 9
f_score_beta = 0.5
f_score_thresh = 0.5

unfreezes = ['no-TF']
# data_description = data_description.format(Path(TFRECORDS_DIR).parent.stem)


def run_training(config=config, 
                 TFRECORDS_DIR=TFRECORDS_DIR, 
                 AUTOTUNE=AUTOTUNE, 
                 batch_size=batch_size, 
                 epochs=epochs, 
                 load_weights=load_weights, 
                 load_g_weights=load_g_weights, 
                 steps_per_epoch=steps_per_epoch, 
                 n_time_augs=n_time_augs, 
                 n_mixup_augs=n_mixup_augs, 
                 data_description=data_description, 
                 init_lr=init_lr, 
                 final_lr=final_lr, 
                 pre_blocks=pre_blocks, 
                 f_score_beta=f_score_beta, 
                 f_score_thresh=f_score_thresh, 
                 unfreezes=unfreezes):
    
    info_text = f"""Model run INFO:

    model: untrained model 
    dataset: {data_description}
    lr: new lr settings
    comments: 2 khz; iterating through different augmentations to see effect on overfit

    VARS:
    data_path       = {TFRECORDS_DIR}
    batch_size      = {batch_size}
    epochs          = {epochs}
    load_weights    = {load_weights}
    steps_per_epoch = {steps_per_epoch}
    f_score_beta    = {f_score_beta}
    f_score_thresh  = {f_score_thresh}
    num_of_shifts   = {n_time_augs}
    num_of_MixUps   = {n_mixup_augs}
    init_lr         = {init_lr}
    final_lr        = {final_lr}
    unfreezes       = {unfreezes}
    preproc blocks  = {pre_blocks}
    """


    #############################################################################
    #############################  RUN  #########################################
    #############################################################################
    
    
    ########### INIT TRAINING RUN AND DIRECTORIES ###############################
    time_start = time.strftime('%Y-%m-%d_%H', time.gmtime())
    Path(f'trainings/{time_start}').mkdir(exist_ok=True)

    n_train, n_noise = get_train_set_size(TFRECORDS_DIR)
    n_train_set = (n_train*(1+n_time_augs+2*n_mixup_augs) + n_noise) // batch_size
    print('Train set size = {}. Epoch should correspond to this amount of steps.'
        .format(n_train_set), '\n')

    seed = np.random.randint(100)
    open(f'trainings/{time_start}/training_info.txt', 'w').write(info_text)

    ###################### DATA PREPROC PIPELINE ################################

    train_data = run_data_pipeline(TFRECORDS_DIR, data_dir='train', 
                                AUTOTUNE=AUTOTUNE)
    test_data = run_data_pipeline(TFRECORDS_DIR, data_dir='test', 
                                AUTOTUNE=AUTOTUNE)
    noise_data = run_data_pipeline(TFRECORDS_DIR, data_dir='noise', 
                                AUTOTUNE=AUTOTUNE)

    augmented_data = run_augment_pipeline(train_data, noise_data,
                                        n_noise, n_time_augs, 
                                        n_mixup_augs,
                                        seed)

    plot_pre_training_spectrograms(train_data, test_data, augmented_data,
                                time_start, seed)

    train_data = prepare(train_data, batch_size, shuffle=True, 
                        shuffle_buffer=n_train_set//2, 
                        augmented_data=np.array(augmented_data)[:,0])

    test_data = prepare(test_data, batch_size)


    #############################################################################
    ######################### TRAINING ##########################################
    #############################################################################

    lr = tf.keras.optimizers.schedules.ExponentialDecay(init_lr,
                                    decay_steps = n_train_set,
                                    decay_rate = (final_lr/init_lr)**(1/epochs),
                                    staircase = True)
    for ind, unfreeze in enumerate(unfreezes):
        # continue
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
                        tfa.metrics.FBetaScore(num_classes=1,
                                                beta=1.,
                                                threshold=f_score_thresh,
                                                name='fbeta1'),       
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
        
        cp_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path, 
            mode = 'min',
            verbose=1, 
            save_weights_only=True,
            save_freq='epoch')

        model.save_weights(checkpoint_path)

        hist = model.fit(train_data, 
                epochs = epochs, 
                validation_data = test_data,
                callbacks=[cp_callback])
        result = hist.history
        save_model_results(checkpoint_dir, result)


    ############## PLOT TRAINING PROGRESS & MODEL EVALUTAIONS ###################

    plot_model_results(time_start, data = data_description, init_lr = init_lr,
                        final_lr = final_lr)
    create_and_save_figure(GoogleMod, TFRECORDS_DIR, batch_size, time_start,
                            plot_cm=True, data = data_description)

if __name__ == '__main__':
    for i in range(NUMBER_OF_TRAININGS):
        run_training(n_time_augs=n_time_augs[i], 
                     n_mixup_augs=n_mixup_augs[i], 
                     init_lr=init_lr[i], 
                     final_lr=final_lr[i])
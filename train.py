import os
import time
from pathlib import Path
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

from hbdet.funcs import save_model_results, get_train_set_size, init_model
from hbdet import models
from hbdet.plot_utils import plot_model_results, create_and_save_figure
from hbdet.tfrec import run_data_pipeline, prepare
from hbdet.augmentation import run_augment_pipeline

TFRECORDS_DIR = ['../Data/Datasets/ScotWest_v4_2khz', 
                #  '../Daten/Datasets/ScotWest_v4_2khz',
                #  '../Daten/Datasets/Mixed_v1_2khz',
                #  '../Daten/Datasets/Mixed_v2_2khz',
                 '../Data/Datasets/SABA01_201511_201604_SN275',
                 '../Data/Datasets/SABA01_201604_201608_SN276',
                 '../Data/Datasets/BERCHOK_SAMANA_200901_4',
                 '../Data/Datasets/CHALLENGER_AMAR123.1',
                 '../Data/Datasets/MELLINGER_NOVA-SCOTIA_200508_EmrldN',
                 '../Data/Datasets/NJDEP_NJ_200903_PU182',
                 '../Data/Datasets/SALLY_TUCKERS_AMAR088.1.16000',
                 '../Data/Datasets/SAMOSAS_EL1_2021',
                 '../Data/Datasets/SAMOSAS_N1_2021',
                 '../Data/Datasets/SAMOSAS_S1_2021',
                 '../Data/Datasets/Tolsta_2kHz_D2_2018'
                ]
AUTOTUNE = tf.data.AUTOTUNE

epochs = [40, 100]

batch_size = [32]*2
time_augs =  [True]*2
mixup_augs = [True]*2
spec_aug =   [True]*2
init_lr = [8e-5, 3e-4] 
final_lr = [6e-6, 1e-5]
weight_clip = [1]*2
ModelClassName = ['GoogleMod', 'EffNet']
keras_mod_name = [False, 'EfficientNetB5']

load_ckpt_path = ['2022-11-17_17', False]
load_g_weights = False
steps_per_epoch = False
data_description = TFRECORDS_DIR
pre_blocks = 9
f_score_beta = 0.5
f_score_thresh = 0.5

unfreezes = ['no-TF']
# data_description = data_description.format(Path(TFRECORDS_DIR).parent.stem)


def run_training(ModelClassName=ModelClassName,
                 TFRECORDS_DIR=TFRECORDS_DIR, 
                 AUTOTUNE=AUTOTUNE, 
                 batch_size=batch_size, 
                 epochs=epochs, 
                 load_ckpt_path=load_ckpt_path, 
                 keras_mod_name=keras_mod_name,
                 load_g_weights=load_g_weights, 
                 steps_per_epoch=steps_per_epoch, 
                 time_augs=time_augs, 
                 mixup_augs=mixup_augs, 
                 spec_aug=spec_aug, 
                 data_description=data_description, 
                 init_lr=init_lr, 
                 final_lr=final_lr, 
                 pre_blocks=pre_blocks, 
                 f_score_beta=f_score_beta, 
                 f_score_thresh=f_score_thresh, 
                 unfreezes=unfreezes,
                 weight_clip=weight_clip):
    
    info_text = f"""Model run INFO:

    model: untrained model 
    dataset: {data_description}
    comments: implemented proper on-the-fly augmentation

    VARS:
    data_path       = {TFRECORDS_DIR}
    batch_size      = {batch_size}
    Model           = {ModelClassName}
    keras_mod_name  = {keras_mod_name}
    epochs          = {epochs}
    load_ckpt_path  = {load_ckpt_path}
    steps_per_epoch = {steps_per_epoch}
    f_score_beta    = {f_score_beta}
    f_score_thresh  = {f_score_thresh}
    bool_time_shift = {time_augs}
    bool_MixUps     = {mixup_augs}
    bool_SpecAug    = {spec_aug}
    weight_clipping = {weight_clip}
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
    Path(f'../trainings/{time_start}').mkdir(exist_ok=True)

    n_train, n_noise = get_train_set_size(TFRECORDS_DIR)
    n_train_set = n_train*(1+time_augs + mixup_augs+spec_aug*2) #// batch_size
    print('Train set size = {}. Epoch should correspond to this amount of steps.'
        .format(n_train_set), '\n')

    seed = np.random.randint(100)
    open(f'../trainings/{time_start}/training_info.txt', 'w').write(info_text)

    ###################### DATA PREPROC PIPELINE ################################

    train_data = run_data_pipeline(TFRECORDS_DIR, data_dir='train', 
                                AUTOTUNE=AUTOTUNE)
    test_data = run_data_pipeline(TFRECORDS_DIR, data_dir='test', 
                                AUTOTUNE=AUTOTUNE)
    noise_data = run_data_pipeline(TFRECORDS_DIR, data_dir='noise', 
                                AUTOTUNE=AUTOTUNE)

    train_data = run_augment_pipeline(train_data, noise_data,
                                        n_noise, n_train, time_augs, 
                                        mixup_augs, seed, spec_aug=spec_aug,
                                        time_start=time_start, plot=True,
                                        random=False)
    
    train_data = prepare(train_data, batch_size, shuffle=True, 
                        shuffle_buffer=n_train_set*3)
    if steps_per_epoch and n_train_set // batch_size < epochs*steps_per_epoch:
        train_data = train_data.repeat(epochs*steps_per_epoch//n_train_set+1)
    

    test_data = prepare(test_data, batch_size)


    #############################################################################
    ######################### TRAINING ##########################################
    #############################################################################

    lr = tf.keras.optimizers.schedules.ExponentialDecay(init_lr,
                                    decay_steps = n_train_set//batch_size,
                                    decay_rate = (final_lr/init_lr)**(1/epochs),
                                    staircase = True)
    for ind, unfreeze in enumerate(unfreezes):
        # if ModelClassName=='GoogleMod':
        #     continue
        if unfreeze == 'no-TF':
            load_g_ckpt = False
        else:
            load_g_ckpt = True
            
        model_class = getattr(models, ModelClassName)#(load_g_ckpt=load_g_ckpt,
                                            #keras_mod_name=keras_mod_name).model
        model = init_model(model_class, 
                    f'../trainings/{load_ckpt_path}/unfreeze_no-TF', 
                    keras_mod_name=keras_mod_name, input_specs=True,
                    load_g_ckpt=load_g_ckpt)
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate = lr,
                                               clipvalue = weight_clip),
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
        # continue
        if not unfreeze == 'no-TF':
            for layer in model.layers[pre_blocks:-unfreeze]:
                layer.trainable = False

        checkpoint_path = f"../trainings/{time_start}/unfreeze_{unfreeze}" + \
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
                # steps_per_epoch=steps_per_epoch,
                validation_data = test_data,
                callbacks=[cp_callback])
        result = hist.history
        save_model_results(checkpoint_dir, result)


    ############## PLOT TRAINING PROGRESS & MODEL EVALUTAIONS ###################

    plot_model_results(time_start, data = data_description, init_lr = init_lr,
                        final_lr = final_lr)
    ModelClass = getattr(models, ModelClassName)
    create_and_save_figure(ModelClass, TFRECORDS_DIR, batch_size, time_start,
                            plot_cm=True, data = data_description, 
                            keras_mod_name=keras_mod_name)

if __name__ == '__main__':
    for i in range(len(time_augs)):
        run_training(batch_size=batch_size[i],
                     epochs=epochs[i],
                     keras_mod_name=keras_mod_name[i],
                     time_augs=time_augs[i], 
                     mixup_augs=mixup_augs[i], 
                     spec_aug=spec_aug[i],
                     init_lr=init_lr[i], 
                     final_lr=final_lr[i],
                     weight_clip=weight_clip[i],
                     ModelClassName=ModelClassName[i],
                     load_ckpt_path=load_ckpt_path[i])
import os
import time
from pathlib import Path
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

from acodet.funcs import save_model_results, get_train_set_size
from acodet import models
from acodet.plot_utils import plot_model_results, create_and_save_figure
from acodet.tfrec import run_data_pipeline, prepare
from acodet.augmentation import run_augment_pipeline
from acodet import global_config as conf

AUTOTUNE = tf.data.AUTOTUNE


def run_training(
    ModelClassName=conf.MODELCLASSNAME,
    data_dir=conf.TFREC_DESTINATION,
    # TODO trennen destination und standardpfad - oder doch nicht?
    batch_size=conf.BATCH_SIZE,
    epochs=conf.EPOCHS,
    load_ckpt_path=conf.LOAD_CKPT_PATH,
    load_g_ckpt=conf.LOAD_G_CKPT,
    keras_mod_name=conf.KERAS_MOD_NAME,
    steps_per_epoch=conf.STEPS_PER_EPOCH,
    time_augs=conf.TIME_AUGS,
    mixup_augs=conf.MIXUP_AUGS,
    spec_aug=conf.SPEC_AUG,
    data_description=conf.DATA_DESCRIPTION,
    init_lr=conf.INIT_LR,
    final_lr=conf.FINAL_LR,
    pre_blocks=conf.PRE_BLOCKS,
    f_score_beta=conf.F_SCORE_BETA,
    f_score_thresh=conf.F_SCORE_THRESH,
    unfreeze=conf.UNFREEZE,
    **kwargs
):
    info_text = f"""Model run INFO:

    model: untrained model 
    dataset: {data_description}
    comments: implemented proper on-the-fly augmentation

    VARS:
    data_path       = {data_dir}
    batch_size      = {batch_size}
    Model           = {ModelClassName}
    keras_mod_name  = {keras_mod_name}
    epochs          = {epochs}
    load_ckpt_path  = {load_ckpt_path}
    load_g_ckpt     = {load_g_ckpt}
    steps_per_epoch = {steps_per_epoch}
    f_score_beta    = {f_score_beta}
    f_score_thresh  = {f_score_thresh}
    bool_time_shift = {time_augs}
    bool_MixUps     = {mixup_augs}
    bool_SpecAug    = {spec_aug}
    init_lr         = {init_lr}
    final_lr        = {final_lr}
    unfreeze       = {unfreeze}
    preproc blocks  = {pre_blocks}
    """

    #############################################################################
    #############################  RUN  #########################################
    #############################################################################
    data_dir = list(Path(data_dir).iterdir())

    ########### INIT TRAINING RUN AND DIRECTORIES ###############################
    time_start = time.strftime("%Y-%m-%d_%H", time.gmtime())
    Path(f"../trainings/{time_start}").mkdir(exist_ok=True, parents=True)

    n_train, n_noise = get_train_set_size(data_dir)
    n_train_set = n_train * (
        1 + time_augs + mixup_augs + spec_aug * 2
    )  # // batch_size
    print(
        "Train set size = {}. Epoch should correspond to this amount of steps.".format(
            n_train_set
        ),
        "\n",
    )

    seed = np.random.randint(100)
    open(f"../trainings/{time_start}/training_info.txt", "w").write(info_text)

    ###################### DATA PREPROC PIPELINE ################################

    train_data = run_data_pipeline(
        data_dir, data_dir="train", AUTOTUNE=AUTOTUNE
    )
    test_data = run_data_pipeline(data_dir, data_dir="test", AUTOTUNE=AUTOTUNE)
    noise_data, n_noise = run_data_pipeline(
        data_dir, data_dir="noise", AUTOTUNE=AUTOTUNE
    )

    train_data = run_augment_pipeline(
        train_data,
        noise_data,
        n_noise,
        n_train,
        time_augs,
        mixup_augs,
        seed,
        spec_aug=spec_aug,
        time_start=time_start,
        plot=False,
        random=False,
    )
    train_data = prepare(
        train_data, batch_size, shuffle=True, shuffle_buffer=n_train_set * 3
    )
    if (
        steps_per_epoch
        and n_train_set // batch_size < epochs * steps_per_epoch
    ):
        train_data = train_data.repeat(
            epochs * steps_per_epoch // (n_train_set // batch_size) + 1
        )

    test_data = prepare(test_data, batch_size)

    #############################################################################
    ######################### TRAINING ##########################################
    #############################################################################

    lr = tf.keras.optimizers.schedules.ExponentialDecay(
        init_lr,
        decay_steps=steps_per_epoch or n_train_set // batch_size,
        decay_rate=(final_lr / init_lr) ** (1 / epochs),
        staircase=True,
    )

    model = models.init_model(
        model_instance=ModelClassName,
        checkpoint_dir=f"../trainings/{load_ckpt_path}/unfreeze_no-TF",
        keras_mod_name=keras_mod_name,
        input_specs=True,
    )

    model.compile(
        optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=lr),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(),
            tf.keras.metrics.Precision(),
            tf.keras.metrics.Recall(),
            tfa.metrics.FBetaScore(
                num_classes=1,
                beta=f_score_beta,
                threshold=f_score_thresh,
                name="fbeta",
            ),
            tfa.metrics.FBetaScore(
                num_classes=1,
                beta=1.0,
                threshold=f_score_thresh,
                name="fbeta1",
            ),
        ],
    )

    if unfreeze:
        for layer in model.layers[pre_blocks:-unfreeze]:
            layer.trainable = False

    checkpoint_path = (
        f"../trainings/{time_start}/unfreeze_{unfreeze}" + "/cp-last.ckpt"
    )
    checkpoint_dir = os.path.dirname(checkpoint_path)

    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        mode="min",
        verbose=1,
        save_weights_only=True,
        save_freq="epoch",
    )

    model.save_weights(checkpoint_path)
    hist = model.fit(
        train_data,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=test_data,
        callbacks=[cp_callback],
    )
    result = hist.history
    save_model_results(checkpoint_dir, result)

    ############## PLOT TRAINING PROGRESS & MODEL EVALUTAIONS ###################

    plot_model_results(
        time_start, data=data_description, init_lr=init_lr, final_lr=final_lr
    )
    ModelClass = getattr(models, ModelClassName)
    create_and_save_figure(
        ModelClass,
        data_dir,
        batch_size,
        time_start,
        plot_cm=True,
        data=data_description,
        keras_mod_name=keras_mod_name,
    )


def save_model(
    string,
    model,
    lr=5e-4,
    weight_clip=None,
    f_score_beta=0.5,
    f_score_thresh=0.5,
):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(),
            tf.keras.metrics.Precision(),
            tf.keras.metrics.Recall(),
            tfa.metrics.FBetaScore(
                num_classes=1,
                beta=f_score_beta,
                threshold=f_score_thresh,
                name="fbeta",
            ),
            tfa.metrics.FBetaScore(
                num_classes=1,
                beta=1.0,
                threshold=f_score_thresh,
                name="fbeta1",
            ),
        ],
    )
    model.save(f"acodet/src/models/{string}")


##############################################################################
##############################################################################
####################### CONFIGURE TRAINING ###################################
##############################################################################
##############################################################################

if __name__ == "__main__":
    data_dir = list(Path(conf.TFREC_DESTINATION).iterdir())

    epochs = [*[43] * 5, 100]
    batch_size = [32] * 6
    time_augs = [True]
    mixup_augs = [True]
    spec_aug = [True]
    init_lr = [*[4e-4] * 5, 1e-5]
    final_lr = [3e-6] * 6
    weight_clip = [None] * 6
    ModelClassName = ["GoogleMod"] * 6
    keras_mod_name = [None] * 6
    load_ckpt_path = [*[False] * 5, "2022-11-30_01"]
    load_g_weights = [False]
    steps_per_epoch = [1000]
    data_description = [data_dir]
    pre_blocks = [9]
    f_score_beta = [0.5]
    f_score_thresh = [0.5]
    unfreeze = ["no-TF"]

    for i in range(len(time_augs)):
        run_training(
            data_dir=data_dir,
            epochs=epochs[i],
            batch_size=batch_size[i],
            time_augs=time_augs[i],
            mixup_augs=mixup_augs[i],
            spec_aug=spec_aug[i],
            init_lr=init_lr[i],
            final_lr=final_lr[i],
            #  weight_clip=weight_clip[i],
            ModelClassName=ModelClassName[i],
            keras_mod_name=keras_mod_name[i],
            load_ckpt_path=load_ckpt_path[i],
            #  load_g_weights=load_g_weights[i],
            steps_per_epoch=steps_per_epoch[i],
            data_description=data_description[i],
            pre_blocks=pre_blocks[i],
            f_score_beta=f_score_beta[i],
            f_score_thresh=f_score_thresh[i],
            unfreeze=unfreeze[i],
        )

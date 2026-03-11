import os
from datetime import datetime as dt
from pathlib import Path
import numpy as np
# import tensorflow_addons as tfa


def set_seed(seed):
    import torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

from acodet import global_config as conf

# AUTOTUNE = tf.data.AUTOTUNE


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
    from acodet import models
    if conf.MODELCLASSNAME == 'TorchModel':
        torch_model = True 
    else:
        torch_model = False
        
    #############################################################################
    #############################  RUN  #########################################
    #############################################################################
    # data_dir = list(Path(data_dir).iterdir())
    # if 'dataset_meta_train' in [d.stem for d in data_dir]:
    #     data_dir = [data_dir[0].parent]

    ########### INIT TRAINING RUN AND DIRECTORIES ###############################
    time_start = dt.strftime(dt.now(), "%Y-%m-%d_%H-%M-%S")
    if load_ckpt_path:
        time_start = load_ckpt_path
    Path(f"../trainings/{time_start}").mkdir(exist_ok=True, parents=True)
    
    model = models.init_model(
        model_instance=ModelClassName,
        checkpoint_dir=f"../trainings/{load_ckpt_path}/unfreeze_False",
        keras_mod_name=keras_mod_name,
        input_specs=True,
    )
    
    if not torch_model:
        

        from acodet.funcs import save_model_results, get_train_set_size
        from acodet.plot_utils import plot_model_results, create_and_save_figure
        from acodet.tfrec import run_data_pipeline, prepare, make_spec_tensor
        from acodet.augmentation import run_augment_pipeline
        from acodet.humpback_model_dir.leaf_pcen import FBetaScore
        import tensorflow as tf
        
        from .tf_dataloader import TFLoader
        
        tfl_obj = TFLoader(conf.ANNOT_DEST)
        # n_train, n_noise = get_train_set_size(data_dir)
        n_train_set = tfl_obj.n_train * (
            1 + time_augs + mixup_augs + spec_aug * 2
        )  # // batch_size
        
        shuffle_buffer_size = conf.STEPS_PER_EPOCH * conf.BATCH_SIZE
        
        print(
            f"\nTrain set size = {n_train_set}. "
            f"\nShuffle buffer size = {shuffle_buffer_size} meaning how much data is loaded "
            "to ensure a thorough shuffle. adjust if this causes memory issues. "
            f"\nYou are running training for {conf.EPOCHS} with {conf.STEPS_PER_EPOCH} steps per epoch. "
            "\n\n\n",
        )

        seed = np.random.randint(100)
        info_text += f'\ntrain_set_size = {tfl_obj.n_train}'
        # info_text += f'\nnoise_set_size = {n_noise}'
        open(f"../trainings/{time_start}/training_info.txt", "w").write(info_text)

        ###################### DATA PREPROC PIPELINE ################################

        # train_data = run_data_pipeline(
        #     data_dir, data_dir="train"
        # )
        # test_data = run_data_pipeline(data_dir, data_dir="test")
        # noise_data = run_data_pipeline(
        #     data_dir, data_dir="noise"
        # )
        train_data = make_spec_tensor(tfl_obj.train_loader())
        noise_data = make_spec_tensor(tfl_obj.noise_loader())
        val_data = make_spec_tensor(tfl_obj.val_loader())

        train_data = run_augment_pipeline(
            train_data,
            noise_data,
            tfl_obj.n_noise,
            tfl_obj.n_train,
            time_augs,
            mixup_augs,
            seed,
            spec_aug=spec_aug,
            time_start=time_start,
            plot=False,
            random=False,
        )
        train_data = prepare(
            train_data, 
            batch_size, 
            shuffle=True, 
            shuffle_buffer=shuffle_buffer_size, 
            AUTOTUNE=tf.data.AUTOTUNE
        )
        if (
            steps_per_epoch
            and n_train_set // batch_size < epochs * steps_per_epoch
        ):
            train_data = train_data.repeat(
                epochs * steps_per_epoch // (n_train_set // batch_size) + 1
            )

        val_data = prepare(
            val_data, 
            batch_size, 
            shuffle=True, 
            seed=42, 
            shuffle_buffer=shuffle_buffer_size, 
            AUTOTUNE=tf.data.AUTOTUNE
            )

        #############################################################################
        ######################### TRAINING ##########################################
        #############################################################################

        lr = tf.keras.optimizers.schedules.ExponentialDecay(
            init_lr,
            decay_steps=steps_per_epoch or n_train_set // batch_size,
            decay_rate=(final_lr / init_lr) ** (1 / epochs),
            staircase=True,
        )
        
        # lr = tf.keras.optimizers.schedules.CosineDecay(
        #     initial_learning_rate=init_lr,
        #     # decay_steps=steps_per_epoch,
        #     first_decay_steps=steps_per_epoch*5,
        #     # t_mul=1.5,
        #     # m_mul=2.0,
        #     alpha=final_lr,
        #     name='CosineDecay',
        #     warmup_target=init_lr,
        #     warmup_steps=2000
        # )
        # lr = tf.keras.optimizers.schedules.CosineDecayRestarts(
        #     initial_learning_rate=5e-4,
        #     first_decay_steps=steps_per_epoch * 5,
        #     t_mul=1.5,
        #     m_mul=1.0,
        #     alpha=5e-6,
        # )

        # def warmup_schedule(step):
        #     if step < 2000:
        #         return (step / 2000) * 5e-4  # Linear warmup
        #     return lr(step - 2000)  # Then use CosineDecay

        # final_lr_schedule = tf.keras.optimizers.schedules.LearningRateSchedule(warmup_schedule)
        if int(tf.__version__.split('.')[1]) == 15:
            optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=lr)
            checkpoint_path = (
                f"../trainings/{time_start}/unfreeze_{unfreeze}" + "/cp-last.ckpt"
            )
        else:
            optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
            checkpoint_path = (
                f"../trainings/{time_start}/unfreeze_{unfreeze}" + "/cp-last.weights.h5"
            )
        model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
            metrics=[
                tf.keras.metrics.BinaryAccuracy(),
                tf.keras.metrics.Precision(),
                tf.keras.metrics.Recall(),
                FBetaScore(
                    num_classes=1,
                    beta=f_score_beta,
                    threshold=f_score_thresh,
                    name="fbeta",
                ),
                FBetaScore(
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


        checkpoint_dir = Path(checkpoint_path).parent
        checkpoint_dir.mkdir(exist_ok=True, parents=True)
        
        earlystopping_callback = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            min_delta=0.01,
            patience=3,
            verbose=1,
            mode='auto',
            baseline=0.15,
            start_from_epoch=65
        )


        modelsaving_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            mode="min",
            verbose=1,
            save_weights_only=True,
            save_freq="epoch",
        )

        model.save_weights(checkpoint_path)

        if load_ckpt_path:
            st = load_ckpt_path
        else:
            st = time_start
        if int(tf.__version__.split('.')[1]) > 15:
            st += '.h5'
        save_model(st, model)

        hist = model.fit(
            train_data,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            validation_data=val_data,
            validation_steps=steps_per_epoch // 5,  
            callbacks=[earlystopping_callback, modelsaving_callback],
        )
        result = hist.history      
        save_model_results(checkpoint_dir, result)
        
        ############## PLOT TRAINING PROGRESS & MODEL EVALUTAIONS ###################

        plot_model_results(
            time_start, data=data_description, init_lr=init_lr, final_lr=final_lr
        )
        
        # create_and_save_figure(
        #     ModelClassName,
        #     data_dir,
        #     batch_size,
        #     time_start,
        #     plot_cm=True,
        #     data=data_description,
        #     keras_mod_name=keras_mod_name,
        # )
        
    else:
        set_seed(42)
        from .torch_train import train, test
        from .torch_data import Loader

        annotations = conf.ANNOT_DEST
        
        data_loaders = Loader(
            annotations,
        )
        
        model = train(model, data_loaders, device=conf.DEVICE)
        
        import torch
        torch.save(model.state_dict, Path(conf.MODEL_DIR).joinpath('torchmodel_v1.pt'))


def save_model(
    string,
    model,
    lr=5e-4,
    weight_clip=None,
    f_score_beta=0.5,
    f_score_thresh=0.5,
):
    
    import tensorflow as tf
    from acodet.humpback_model_dir.leaf_pcen import FBetaScore
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(),
            tf.keras.metrics.Precision(),
            tf.keras.metrics.Recall(),
            FBetaScore(
                num_classes=1,
                beta=f_score_beta,
                threshold=f_score_thresh,
                name="fbeta",
            ),
            FBetaScore(
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

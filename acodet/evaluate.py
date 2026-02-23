import os
from datetime import datetime as dt
from pathlib import Path
import numpy as np

import torch
import torchaudio as ta
from acodet import models
from acodet import global_config as conf

from .torch_data import Loader

def evaluate(train_date=False, **kwargs):
    
    timestamp_foldername = dt.strftime(dt.now(), "%Y-%m-%d_%H-%M-%S")
    timestamp_foldername += conf.ANNOTS_TIMESTAMP_FOLDER


    if not train_date:
        model = models.init_model(timestamp_foldername=timestamp_foldername)
    else:
        model = models.init_model(
            checkpoint_dir=f"../trainings/{train_date}/unfreeze_no-TF",
        )
        
        
    data_loader = Loader(conf.ANNOT_DEST)
    test_data = data_loader.test_loader()

    
    for idx, tuple in enumerate(test_data):
        audio, new_labels, paths, timestamps = tuple
        
        
        if conf.MODELCLASSNAME == 'BacpipeModel':
            re_audio = ta.functional.resample(
                audio, 
                conf.SR, 
                model.model.model.sr
                )
            preprocessed_frames = model.model.model.preprocess(re_audio)
            new_predictions = model.classify(preprocessed_frames, **kwargs)
        elif conf.MODELCLASSNAME == 'TorchModel':
            new_predictions = model(audio).detach().cpu().squeeze().numpy()
        else:
            new_predictions = model.model.predict(
                    tf.convert_to_tensor(audio)
                ).squeeze()
        if idx == 0:
            predictions = new_predictions
            labels = new_labels
        else:
            predictions = torch.vstack([
                predictions, 
                torch.tensor(new_predictions)
                ])
            labels = torch.vstack([labels, new_labels])
            
    if conf.MODEL_NAME == 'perch_v2':
        class_labels = model.model.model.classes
        humpback_label_idx = np.where(np.array(class_labels)=='Megaptera novaeangliae')[0][0]
        predictions = predictions[:, humpback_label_idx]
    elif conf.MODEL_NAME == 'google_whale':
        class_labels = model.model.model.classes
        humpback_label_idx = np.where(np.array(class_labels)=='Humpback')[0][0]
        predictions = predictions[:, humpback_label_idx]
        

    ### now you have predictions and labels vectors that can be compared
    
def get_tensorflow_preds():
    import tensorflow as tf
    import librosa as lb

    from acodet.funcs import get_files, run_inference
    from acodet.annotate import MetaData
    from acodet import tfrec
    tfrec_path = conf.TFREC_DESTINATION
    model_name = conf.MODEL_NAME
    
    val_data = tfrec.run_data_pipeline(tfrec_path, "test", return_spec=False)
    
    model = models.init_model(
        load_from_ckpt=True,
        model_name=model_name,
        training_path=conf.LOAD_CKPT_PATH
    )
    preds = model.predict(x=models.prep_ds_4_preds(val_data))
    labels = models.get_val_labels(val_data, len(preds))
    
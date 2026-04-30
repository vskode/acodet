import os
from datetime import datetime as dt
from pathlib import Path
import logging
import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics as metrics

import torch
import torchaudio as ta
from acodet import models
from acodet import global_config as conf

from .torch_data import Loader

def evaluate(train_date=False, **kwargs):
    logging.basicConfig(level='INFO', format='%(asctime)s %(levelname)s %(message)s')
    logger = logging.getLogger(__name__)

    model_file = conf.EVAL_MODEL_FILE
    
    if not conf.MODELCLASSNAME in ('TorchModel', 'HumpBackNorthAtlantic', 'BacpipeModel'):
        logger.error(f"Evaluation step not yet implemented for {conf.MODELCLASSNAME}. Aborting.")
        return 1

    if conf.DEVICE != 'cpu':
        logger.warning(f"This script runs on CPU. Current device {conf.DEVICE} may not be used.")

    # don't import tensorflow if it's not needed
    if not conf.MODELCLASSNAME in ('TorchModel', 'BacpipeModel'):
        import tensorflow as tf

    timestamp_foldername = dt.strftime(dt.now(), "%Y-%m-%d_%H-%M-%S")
    timestamp_foldername += conf.ANNOTS_TIMESTAMP_FOLDER

    logger.info(f"Initializing model {conf.MODELCLASSNAME}")

    if conf.MODELCLASSNAME == 'TorchModel':
        # if using TorchModel, load from the appropriate path
        model = models.init_model()
        model_path = Path(model_file)
        if not Path.exists(model_path):
            logger.error(f"Model file {model_file} not found. Please check path.")
            return 1
        checkpoint = torch.load(model_file, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint)
        figure_dir = model_path.parent.joinpath('evaluation/')
        figure_dir.mkdir(exist_ok=True)
        print(figure_dir)
    elif not train_date:
        # allow user to evaluate a model that they have not trained yet
        model = models.init_model(timestamp_foldername=timestamp_foldername)
        figure_dir = f"../trainings/{timestamp_foldername}/figures/"
    else:
        # load specified training
        if not Path(f"../trainings/{train_date}/").exists():
            logger.error("Advanced config setting `load_ckpt_path` not found")
            return 1
        logger.info("initializing model")
        model = models.init_model(
            checkpoint_dir=f"../trainings/{train_date}/unfreeze_no-TF",
        )

        figure_dir = f"../trainings/{train_date}/figures/"

    logger.info(f"Loading test data from {conf.ANNOT_DEST}")

    # load test data from advanced config ANNOTATION_DESTINATION
    data_loader = Loader(conf.ANNOT_DEST)
    test_data = data_loader.test_loader()

    # create two vectors: one for true labels, and one for predicted labels

    for idx, tuple in enumerate(test_data):
        audio, new_labels, paths, timestamps = tuple

        if conf.MODELCLASSNAME == 'BacpipeModel':
            re_audio = ta.functional.resample(
                audio, 
                conf.SR, 
                model.model.model.sr
                )
            preprocessed_frames = model.model.model.preprocess(re_audio)
            new_predictions = torch.tensor(model.classify(preprocessed_frames, **kwargs))
        elif conf.MODELCLASSNAME == 'TorchModel':
            new_predictions = model(audio).detach().cpu().squeeze()
        else:
            new_predictions = torch.tensor(model.predict(
                    tf.convert_to_tensor(audio)
                ).squeeze())

        if idx == 0:
            predictions = new_predictions
            class_labels = new_labels
        else:
            predictions = torch.hstack([
                predictions, 
                new_predictions
                ])
            class_labels = torch.hstack([class_labels, new_labels])
            
    if conf.MODEL_NAME == 'perch_v2':
        class_labels = model.model.model.classes
        humpback_label_idx = np.where(np.array(class_labels)=='Megaptera novaeangliae')[0][0]
        predictions = predictions[:, humpback_label_idx]
    elif conf.MODEL_NAME == 'google_whale':
        class_labels = model.model.model.classes
        humpback_label_idx = np.where(np.array(class_labels)=='Humpback')[0][0]
        predictions = predictions[:, humpback_label_idx]

    logger.info("All predictions collected; flattening")

    class_labels = class_labels.flatten()
    predictions = predictions.flatten()

    # create path to save the figures in
    Path(figure_dir).mkdir(exist_ok=True, parents=True)

    ####################################
    ### Precision, recall, and f1 score 
    ####################################
    logger.info("Calculating precision, recall, and f1 scores")

    # calculate precision and recall
    precision, recall, thresholds = metrics.precision_recall_curve(class_labels, predictions)
    fig_filepath = Path(figure_dir).joinpath('precision_recall_stats.txt')

    # iterate through thresholds
    # and write precision, recall, and f1 score to a text file
    with open(fig_filepath, 'w') as file:
        file.write("precision,recall,threshold,f1_score\n")
        for i, t in enumerate(thresholds):
            f1_score = 2 * (precision[i] * recall[i]) / (precision[i] + recall[i])
            line = f"{precision[i]},{recall[i]},{t},{f1_score}\n"
            file.write(line)

    # create precision-recall curve plot
    fig, ax = plt.subplots()
    ax.plot(recall, precision, color='tab:blue')
    ax.set_title('Precision-Recall Curve')
    ax.set_ylabel('Precision')
    ax.set_xlabel('Recall')

    # save plot
    fig_filepath = Path(figure_dir).joinpath('precision_recall_curve.png')
    fig.savefig(fig_filepath)

    ###################################
    # Confusion matrix
    ###################################
    logger.info("Creating confusion matrix")

    # a confusion matrix needs binary classification
    # so use the different thresholds calculated above 
    # to mask the continuous values into class predictions

    for threshold in thresholds:
        # if the predicted value is greater than the threshold,
        # give it a value of 1.0, otherwise it's 0.0
        threshold_labels = (predictions > threshold).to(torch.float)

        # calculate confusion matrix
        confusion_matrix = metrics.confusion_matrix(class_labels, threshold_labels)

        # create interpretable display
        cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix)

        # save plot
        threshold_pretty = (threshold * 100).astype('int')
        fig_filepath = Path(figure_dir).joinpath(f'confusion_matrix_threshold_{threshold_pretty}.png')
        cm_display.plot().figure_.savefig(fig_filepath)
        plt.close()

    ###################################
    # ROC Curve
    ###################################
    logger.info("Creating ROC curve")

    # calculate roc curve
    false_positive_rate, true_positive_rate, thresholds = metrics.roc_curve(class_labels, predictions)
    roc_auc = metrics.auc(false_positive_rate, true_positive_rate)

    # create figure
    fig, ax = plt.subplots()
    ax.plot(false_positive_rate, true_positive_rate, color='tab:blue', label='ROC curve (area = %0.2f)' % roc_auc)
    ax.plot([0, 1], [0, 1], 'k--') # plot straight x/y ("no skill") line for comparison
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title("ROC Curve")
    plt.legend()

    # save figure
    fig_filepath = Path(figure_dir).joinpath('roc_curve.png')
    fig.savefig(fig_filepath)

    return


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
    

import os
from datetime import datetime as dt
from pathlib import Path
import logging
import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics as metrics
from tqdm import tqdm

import torch
import torchaudio as ta
from acodet import models
from acodet import global_config as conf

from .torch_data import Loader

def evaluate(train_date=False, **kwargs):
    logging.basicConfig(level='INFO', format='%(asctime)s %(levelname)s %(message)s')
    logger = logging.getLogger(__name__)

    model_file = conf.LOAD_CKPT_PATH
    if not conf.LOAD_CKPT_PATH:
        model_file = conf.MODEL_NAME
    figure_dir = Path(f'../trainings/{model_file}') / 'evaluation'
    figure_dir.mkdir(exist_ok=True, parents=True)
    
    if not conf.MODELCLASSNAME in ('TorchModel', 'HumpBackNorthAtlantic', 'BacpipeModel'):
        logger.error(f"Evaluation step not yet implemented for {conf.MODELCLASSNAME}. Aborting.")
        return 1

    # don't import tensorflow if it's not needed
    if not conf.MODELCLASSNAME in ('TorchModel', 'BacpipeModel'):
        import tensorflow as tf

    timestamp_foldername = dt.strftime(dt.now(), "%Y-%m-%d_%H-%M-%S")
    timestamp_foldername += conf.ANNOTS_TIMESTAMP_FOLDER

    logger.info(f"Initializing model {conf.MODELCLASSNAME}")

    if conf.MODELCLASSNAME == 'TorchModel':
        # if using TorchModel, load from the appropriate path
        model = models.init_model()
    elif conf.MODELCLASSNAME == 'BacpipeModel':
        model = models.init_model()
        conf.SR = model.embedder.model.sr
        
    # elif not train_date:
    #     # allow user to evaluate a model that they have not trained yet
    #     model = models.init_model(timestamp_foldername=timestamp_foldername)
    else:
        logger.info("initializing model")
        model = models.init_model()


    logger.info(f"Loading test data from {conf.ANNOT_DEST}")

    # load test data from advanced config ANNOTATION_DESTINATION
    data_loader = Loader(conf.ANNOT_DEST)
    test_data = data_loader.test_loader()

    # create two vectors: one for true labels, and one for predicted labels

    predictions = []
    class_labels = []
    for idx, tuple in tqdm(
        enumerate(test_data), 
        'running inference on test data', 
        total=len(data_loader.test) // conf.BATCH_SIZE
        ):
        audio, new_labels, paths, timestamps = tuple

        if conf.MODELCLASSNAME == 'BacpipeModel':
            
            # this is the case for using their own classifiers
            if not conf.BOOL_LIN_CLFIER and conf.MODEL_NAME in ['perch_v2', 'google_whale']:
                import tensorflow as tf
                audio = tf.convert_to_tensor(audio, dtype=tf.float32)
                embedings = model.embedder.model(audio).squeeze()
                new_predictions = model.embedder.model.classifier_predictions(embedings)
                if conf.MODEL_NAME == 'google_whale':
                    new_predictions = tf.sigmoid(new_predictions)
                    
            else:    
                audio = torch.tensor(audio)
                with torch.inference_mode():
                    new_predictions = torch.sigmoid(model(audio.to(conf.DEVICE))).squeeze()
        elif conf.MODELCLASSNAME == 'TorchModel':
            with torch.inference_mode():
                new_predictions = torch.sigmoid(model(audio)).squeeze()#.detach().cpu().squeeze()
        else:
            new_predictions = torch.tensor(model.predict(
                    tf.convert_to_tensor(audio)
                ).squeeze())
        predictions.extend(new_predictions)
        class_labels.extend(new_labels)
        
        # Uncomment this if you just want to try running it for a little data to 
        # make sure the code runs.
        if idx > 100:
            break
    
    if not isinstance(predictions[0], torch.Tensor):
        predictions = torch.tensor(np.array(predictions))
        class_labels = np.array(class_labels)
    else:
        predictions = torch.hstack(predictions).to('cpu')
        class_labels = torch.hstack(class_labels).to('cpu')

    # I commented these two out cause at the moment, we are using the model's only as 
    # feature extractors, we would need to add another option 
    
    if (
        conf.MODEL_NAME == 'perch_v2' 
        and conf.MODELCLASSNAME == 'BacpipeModel'
        and not conf.BOOL_LIN_CLFIER
        ):
        model_labels = np.array(model.embedder.model.classes)
        humpback_label_idx = np.where(model_labels=='Megaptera novaeangliae')[0][0]
        predictions = predictions[:, humpback_label_idx]
    elif (
        conf.MODEL_NAME == 'google_whale' 
        and conf.MODELCLASSNAME == 'BacpipeModel'
        and not conf.BOOL_LIN_CLFIER
        ):
        model_labels = np.array(model.embedder.model.classes)
        humpback_label_idx = np.where(model_labels=='Humpback')[0][0]
        predictions = predictions[:, humpback_label_idx]

    logger.info("All predictions collected; flattening")

    class_labels = class_labels.flatten()
    predictions = predictions.flatten()

    ####################################
    ### Precision, recall, and f1 score 
    ####################################
    logger.info("Calculating precision, recall, and f1 scores")

    # calculate precision and recall
    precision, recall, thresholds = metrics.precision_recall_curve(class_labels, predictions)
    fig_filepath = Path(figure_dir).joinpath('precision_recall_stats.csv')

    fn, true, fp = np.unique([np.round(jj)-ii for jj, ii in zip(predictions, class_labels)], return_counts=True)[-1]
    logger.info(f"{fn=}, {true=}, {fp=}")
    
    d = metrics.classification_report(
        class_labels,
        [np.round(ii) for ii in predictions],
        output_dict=True
    )
    logger.info(d)
    # iterate through thresholds
    # and write precision, recall, and f1 score to a text file
    f1_scores = []

    with open(fig_filepath, 'w') as file:
        file.write("precision,recall,threshold,f1_score\n")
        for i, t in enumerate(thresholds):
            f1_score = 2 * (precision[i] * recall[i]) / (precision[i] + recall[i])
            f1_scores.append(f1_score)
            line = f"{precision[i]},{recall[i]},{t},{f1_score}\n"
            file.write(line)
    auc_pr = metrics.auc(recall, precision)

    # create threshold vs f1 score plot
    fig, ax = plt.subplots()
    ax.plot(thresholds, f1_scores)
    ax.set_title(f'F1 Score by threshold: {model_file}')
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel('F1 Score')
    ax.set_xlabel('Threshold')
    fig_filepath = Path(figure_dir).joinpath('f1_threshold_curve.png')
    fig.savefig(fig_filepath)

    # create precision-recall curve plot
    fig, ax = plt.subplots()
    ax.plot(recall, precision, color='tab:blue', label='PR curve (area = %0.2f)' % auc_pr)
    ax.set_title(f'Precision-Recall Curve: {model_file}')
    ax.set_ylabel('Precision')
    ax.set_xlabel('Recall')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    plt.legend()

    # save plot
    fig_filepath = Path(figure_dir).joinpath('precision_recall_curve.png')
    fig.savefig(fig_filepath)

    #####################################
    # Confusion matrix for threshold of 0.5
    #####################################
    logger.info("Creating confusion matrix for threshold 0.5")

    # a confusion needs binary classification
    # use 0.5 first, which is what we use in 
    # training logging
    # so it's a clean comparison

    # first find the threshold closest to 0.5
    # create a matrix of the absolute difference between threshold and 0.5
    threshold_diff = np.absolute(thresholds - 0.5)
    # then find the index of the smallest element of that array
    closest_threshold_index = threshold_diff.argmin() 
    closest_threshold = thresholds[closest_threshold_index]

    # now use that cutoff to sort predictions into noise/ call
    prediction_classes = (predictions > closest_threshold).to(torch.float)

    # calculate confusion matrix
    confusion_matrix = metrics.confusion_matrix(class_labels, prediction_classes)
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix)
    title = f"{model_file} threshold {closest_threshold:.2f}"
    cm_display.plot()
    cm_display.ax_.set_title(title)

    # save plot
    fig_filepath = Path(figure_dir).joinpath(f'confusion_matrix_05.png')
    cm_display.figure_.savefig(fig_filepath)
    plt.close()

    ###################################
    # Confusion matrix for best f1 score`
    ###################################
    logger.info("Creating confusion matrix for best threshold")

    # a confusion matrix needs binary classification
    # so use the best f1 score calculated above
    # to mask the continuous values into class predictions

    best_f1_index = np.argmax(f1_scores)
    best_threshold = thresholds[best_f1_index]

    # if the predicted value is greater than the threshold,
    # give it a value of 1.0, otherwise it's 0.0
    threshold_labels = (predictions > best_threshold).to(torch.float)

    # calculate confusion matrix
    confusion_matrix = metrics.confusion_matrix(class_labels, threshold_labels)

    # create interpretable display
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix)
    cm_display.plot()
    cm_display.ax_.set_title(f"{model_file} threshold {best_threshold:.2f}")

    # save plot
    fig_filepath = Path(figure_dir).joinpath(f'confusion_matrix_best_threshold.png')
    cm_display.figure_.savefig(fig_filepath)
    plt.close()
    
    ###################################################
    # write the stats of the best f1 score to file
    ###################################################

    best_stats_filepath = Path(figure_dir).joinpath('best_f1_stats.csv')
    with open(best_stats_filepath, 'w') as file:
        file.write("precision,recall,threshold,f1_score\n")
        file.write(str(precision[best_f1_index]) + ',' + \
                str(recall[best_f1_index]) + ',' + \
                str(thresholds[best_f1_index]) + ',' + \
                str(f1_scores[best_f1_index]) + '\n')

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
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f"ROC Curve: {model_file}")
    plt.legend()

    # save figure
    fig_filepath = Path(figure_dir).joinpath('roc_curve.png')
    fig.savefig(fig_filepath)
    
    
    print(f'All plots saved to {figure_dir=}')

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
    

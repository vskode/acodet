import ntpath
from glob import glob

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn import metrics
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from .utils import get_state_dict, batch_to_device
from .data import AudioDataset


def inference_k_random(net, state_dict_path, test_dl, test_metadata_df, k=1):
    """
    Inference helper function based on selecting a random window from a single
    test file for scoring. Results can be improved by increasing k, leading to
    selecting k random windows for scoring. Final predictions are averaged over
    all k runs.

    Parameters
    ----------
    net: Pytorch network class, e.g. SpectrogramCNN() defined in net.py.
    state_dict_path: str. Path to a Pytorch state dictionary containing the model weights.
    test_dl: Pytorch Dataloader
    test_metadata_df: Pandas dataframe containing the metadata of all test instances
    k: int (default=1). Number of random drawing iterations.

    Returns
    -------
    test_metadata_df: Pandas dataframe containing the final predictions
    preds: np.array containing all individual predictions

    """
    torch.cuda.empty_cache()
    device = "cuda" if torch.cuda.is_available() else 'cpu'
    sd = get_state_dict(state_dict_path)
    model = net.eval().to(device)
    model.load_state_dict(sd)

    preds = []
    for _ in range(k):
        with torch.no_grad():
            _preds = []
            for batch in tqdm(test_dl):
                batch = batch_to_device(batch, device)
                out = model(batch['wave']).cpu().numpy()
                if np.isnan(out).any():
                    raise
                _preds += [out]
            preds.append(np.vstack(_preds))

    avg_preds = np.mean(preds, axis=0)
    test_metadata_df['predicted_class_id'] = avg_preds.argmax(axis=-1)
    torch.cuda.empty_cache()
    return test_metadata_df, np.array(preds)


def inference_all(net, state_dict_path, test_metadata_df, cfg, data_path):
    """
    Inference helper function that scores all preprocessed snippets of a
    long file and averages over all predictions. In order to use all information
    available in a long audio file, e.g. 60 seconds, when a model is trained on shorter
    fragments, e.g. 5 seconds, the audio file can be preprocessed into smaller snippets,
    e.g. 12 5 second snippets. This function loops over all these snippets for inference
    and averages the final result per test file.

    Parameters
    ----------
    net: Pytorch network class, e.g. SpectrogramCNN() defined in net.py.
    state_dict_path: str. Path to a Pytorch state dictionary containing the model weights.
    test_metadata_df: Pandas dataframe containing the metadata of all test instances
    cfg: SimpleNameSpace containing all configurations
    data_path: str. Path to the test files.

    Returns
    -------
    test_metadata_df: Pandas dataframe containing the final predictions
    preds: np.array containing all individual predictions
    """
    torch.cuda.empty_cache()
    device = "cuda" if torch.cuda.is_available() else 'cpu'
    sd = get_state_dict(state_dict_path)
    net = net.eval().to(device)
    net.load_state_dict(sd)

    preds = []
    # Loop over all individual test files and
    # initialize a Dataloader.
    # Performance could be increased here when
    # more files are batched together instead of
    # initializing a Dataloader for single audio files.
    for i in tqdm(range(len(test_metadata_df))):
        name = ntpath.basename(test_metadata_df.iloc[i]['path'][:-4])
        data = glob(f'{data_path}/{name}_*.wav')
        df = pd.DataFrame(data, columns=['path'])
        pred_ds = AudioDataset(df, mode='test', cfg=cfg)
        pred_dl = DataLoader(pred_ds, shuffle=False, batch_size=cfg.batch_size, num_workers=cfg.num_workers)

        with torch.no_grad():
            _preds = []
            # Predict all batches
            for batch in pred_dl:
                batch = batch_to_device(batch, device)
                out = net(batch['wave'])
                _preds += [out.cpu().numpy()]
            _preds = np.vstack(_preds)
            preds.append(_preds)
    torch.cuda.empty_cache()
    avg_preds = np.array([p.mean(axis=0) for p in preds])
    test_metadata_df['predicted_class_id'] = avg_preds.argmax(axis=-1)
    return test_metadata_df, preds


def error_analysis(exp_path, dset, n_classes, filename=None, tag='', classlabels=None):
    """
    Helper function to automatize the error analysis.
    Computes and plots a confusion matrix, as well as
    metrics like f1 score, precision/recall etc.
    Final results are saved in a .csv file.

    Parameters
    ----------
    exp_path: str. Path for saving the file.
    filename: str. Custom filename of .csv file containing model predictions.
    tag: str. Custom tag to be appended for saving files.

    """
    if filename:
        df = pd.read_csv(f'{exp_path}/{filename}')
    else:
        df = pd.read_csv(f'{exp_path}/{dset}_predictions_k-random.csv')

    df_eval = df[['label', 'predicted_class_id']]
    y_pred = df_eval['predicted_class_id']
    y_true = df_eval['label']

    cm = metrics.confusion_matrix(y_true, y_pred)
    np.save(f"{exp_path}/{dset}_cm{tag}.npy", cm)

    plot_confusion_matrix(cm, exp_path, dset)

    report = metrics.classification_report(y_true, y_pred, digits=3, output_dict=True)
    evaluation = pd.DataFrame(report).transpose()

    if classlabels:
        evaluation["class_label"] = ""
        for i in range(n_classes):
            evaluation['class_label'][i] = classlabels[i]

    pd.options.display.float_format = "{:,.2f}".format
    evaluation.to_csv(f'{exp_path}/{dset}_evaluation{tag}.csv')


def plot_confusion_matrix(cm, exp_path, dset):
    """
    Plotting function for confusion matrices.
    A confusion matrix (cm) can be passed and
    automatically be plotted.

    Parameters
    ----------
    cm: np.array. Confusion matrix, e.g. obtained from sklearn.
    exp_path: str. Path for saving the file.

    """
    fig = plt.figure(figsize=(16, 14))
    ax = plt.subplot()
    sns.heatmap(cm, annot=True, ax=ax, fmt='g', cmap="magma", mask=cm == 0, vmax=10)
    ax.xaxis.set_label_position('bottom')
    plt.xticks(rotation=90)
    ax.set_ylabel('True', fontsize=20)
    plt.yticks(rotation=0)
    plt.title(f'Confusion Matrix: {dset} Set', fontsize=20)
    plt.savefig(f'{exp_path}/{dset}_conf_matrix_best_model.png')
    plt.close()
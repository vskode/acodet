import numpy as np
import matplotlib.pyplot as plt
import time
from pathlib import Path
import seaborn as sns
import json
from . import global_config as conf
from . import funcs
import tensorflow as tf

drop_keyz = {"fbeta", "val_fbeta"}
sns.set_theme()
sns.set_style("white")


def plot_model_results(
    datetimes, labels=None, fig=None, legend=True, **kwargs
):
    plt.rc("axes", labelsize=20)
    plt.rc("axes", titlesize=20)
    plt.rc("xtick", labelsize=14)
    plt.rc("ytick", labelsize=14)
    if fig == None:
        fig = plt.figure(figsize=(15, 8))
        savefig = True
    else:
        savefig = False

    if not isinstance(datetimes, list):
        datetimes = [datetimes]

    checkpoint_paths = []
    for datetime in datetimes:
        checkpoint_paths += list(
            Path(f"../trainings/{datetime}").glob("unfreeze_*")
        )

    r, c = 2, 5
    for j, checkpoint_path in enumerate(checkpoint_paths):
        unfreeze = checkpoint_path.stem.split("_")[-1]

        if not Path(f"{checkpoint_path}/results.json").exists():
            if j == 0:
                ax = fig.subplots(ncols=c, nrows=r)
            continue
        with open(f"{checkpoint_path}/results.json", "r") as f:
            results = json.load(f)
            for k in drop_keyz:
                try:
                    del results[k]
                except KeyError:
                    continue

        if j == 0:
            c = len(list(results.keys())) // 2
            ax = fig.subplots(ncols=c, nrows=r)

        if not labels is None:
            label = labels[j]
        else:
            label = f"{checkpoint_path.parent.stem}_{unfreeze}"

        for i, key in enumerate(results.keys()):
            if "val_" in key:
                row = 1
            else:
                row = 0
            if "loss" in key or i == 0:
                col = 0
            else:
                col += 1

            ax[row, col].plot(results[key], label=label)
            if not col == 0:
                ax[row, col].set_ylim([0.7, 1.01])

            # axis handling depending on subplot index
            if row == 1 and col == 0:
                ax[row, col].set_ylim([0, 1.01])
            if row == 0 and j == 0:
                ax[row, col].set_title(f"{key}")
            if row == col == 0:
                ax[row, col].set_ylim([0, 0.5])
                ax[row, col].set_ylabel("training")
            elif row == 1 and col == 0:
                ax[row, col].set_ylabel("val")
            if legend and row == 1 and col == c - 1:
                ax[row, col].legend(loc="center left", bbox_to_anchor=(1, 0.5))

    info_string = ""
    for key, val in kwargs.items():
        info_string += f" | {key}: {val}"

    ref_time = time.strftime("%Y%m%d", time.gmtime())
    if savefig:
        fig.tight_layout()
        fig.savefig(f"../trainings/{datetime}/model_results_{ref_time}.png")
    else:
        return fig


def plot_sample_spectrograms(
    dataset,
    *,
    dir,
    name,
    ds_size=None,
    random=True,
    seed=None,
    sr=conf.SR,
    rows=4,
    cols=4,
    plot_meta=False,
    **kwargs,
):
    r, c = rows, cols
    if isinstance(dataset, tf.data.Dataset):
        if random:
            if ds_size is None:
                ds_size = sum(1 for _ in dataset)
            np.random.seed(seed)
            rand_skip = np.random.randint(ds_size)
            sample = dataset.skip(rand_skip).take(r * c)
        else:
            sample = dataset.take(r * c)
    elif isinstance(dataset, list):
        sample = dataset

    max_freq_bin = 128 // (conf.SR // 2000)

    fmin = sr / 2 / next(iter(sample))[0].numpy().shape[0]
    fmax = sr / 2 / next(iter(sample))[0].numpy().shape[0] * max_freq_bin
    fig, axes = plt.subplots(nrows=r, ncols=c, figsize=[12, 10])

    for i, (aud, *lab) in enumerate(sample):
        if i == r * c:
            break
        ar = aud.numpy()[:, 1:max_freq_bin].T
        axes[i // r][i % c].imshow(
            ar, origin="lower", interpolation="nearest", aspect="auto"
        )
        if len(lab) == 1:
            axes[i // r][i % c].set_title(f"label: {lab[0]}")
        elif len(lab) == 3:
            label, file, t = (v.numpy() for v in lab)
            axes[i // r][i % c].set_title(
                f"label: {label}; t in f: {funcs.get_time(t)}\n"
                f"file: {Path(file.decode()).stem}"
            )
        if i // r == r - 1 and i % c == 0:
            axes[i // r][i % c].set_xticks(np.linspace(0, ar.shape[1], 5))
            xlabs = np.linspace(0, 3.9, 5).astype(str)
            axes[i // r][i % c].set_xticklabels(xlabs)
            axes[i // r][i % c].set_xlabel("time in s")
            axes[i // r][i % c].set_yticks(np.linspace(0, ar.shape[0] - 1, 7))
            ylabs = np.linspace(fmin, fmax, 7).astype(int).astype(str)
            axes[i // r][i % c].set_yticklabels(ylabs)
            axes[i // r][i % c].set_ylabel("freq in Hz")
        else:
            axes[i // r][i % c].set_xticks([])
            axes[i // r][i % c].set_xticklabels([])
            axes[i // r][i % c].set_yticks([])
            axes[i // r][i % c].set_yticklabels([])

    fig.suptitle(f"{name} sample of 16 spectrograms. random={random}")
    fig.savefig(f"../trainings/{dir}/{name}_sample.png")

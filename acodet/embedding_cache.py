"""
Cache of frozen bacpipe model embeddings for training a linear classifier.

When a BacpipeModel trains only its linear classifier (bool_lin_clfier), the
bacpipe model is frozen and no augmentation is applied, so every clip has the
same embedding in every epoch. Computing the embeddings once and saving them
to disk makes each epoch take seconds instead of running the bacpipe model
over the whole dataset again.

Each cache file is keyed by the model name, its sample rate and input length,
and the exact list of clips (files, start times, durations and labels), so
changing any of these creates a new file. A new checkpoint saved under the
same model name is NOT detected: delete the cache directory if the model's
weights change.
"""
import hashlib
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from acodet import global_config as conf
from acodet.progress import progress
from acodet.torch_data import collate_fn, NUM_CORES


def cache_path(model, dataset, split):
    """
    Path of the cache file for the embeddings of a dataset.

    Parameters
    ----------
    model : acodet.models.BacpipeModel
        model whose embeddings are cached
    dataset : acodet.torch_data.AudioDataset
        clips to embed
    split : str
        name of the split, e.g. 'train' or 'val' (only used in the file name)

    Returns
    -------
    pathlib.Path
        path of the cache file
    """
    bacpipe_model = model.embedder.model
    key = hashlib.sha1()
    key.update(f"{conf.MODEL_NAME}|{bacpipe_model.sr}|{bacpipe_model.segment_length}".encode())
    key.update("\n".join(map(str, dataset.filepaths)).encode())
    for values in (dataset.starts, dataset.durations, dataset.labels.numpy()):
        key.update(np.ascontiguousarray(values).tobytes())
    file_name = (
        f"{conf.MODEL_NAME}_{bacpipe_model.segment_length}_{split}_{key.hexdigest()[:12]}.pt"
    )
    return Path(conf.EMBEDDING_CACHE_DIR) / file_name


def load_or_compute(model, dataset, split, device):
    """
    Load the embeddings of a dataset from the cache, or compute and cache them.

    Parameters
    ----------
    model : acodet.models.BacpipeModel
        model whose embeddings are cached
    dataset : acodet.torch_data.AudioDataset
        clips to embed
    split : str
        name of the split, e.g. 'train' or 'val' (only used in the file name)
    device : str
        device to send the audio to

    Returns
    -------
    embeddings : torch.Tensor
        float32 embeddings, shape (clips, embedding_dim)
    labels : torch.Tensor
        label of each clip
    """
    path = cache_path(model, dataset, split)
    if path.exists():
        print(f"Loading cached {split} embeddings from {path}", flush=True)
        cached = torch.load(path)
        return cached["embeddings"], cached["labels"]

    loader = DataLoader(
        dataset,
        batch_size=conf.BATCH_SIZE,
        shuffle=False,
        pin_memory=True,
        num_workers=NUM_CORES,
        prefetch_factor=4,
        collate_fn=collate_fn,
    )
    # Embeddings go into one preallocated tensor rather than a list of
    # per-batch tensors. Small tensors kept from every batch land in the heap
    # space freed by that batch's audio, which then never fits the next
    # batch, so memory grew by one batch of audio per step (150 GB for
    # perch_v2) even though nothing leaked.
    embeddings, labels, n = None, [], 0
    # fork_rng: the loader draws from the global RNG, which would otherwise
    # make a run that builds the cache shuffle differently from one that loads it
    with torch.no_grad(), torch.random.fork_rng(devices=[]):
        for batch in progress(loader, desc=f"Caching {split} embeddings"):
            batch_embeddings = model.embed(batch[0].to(device))
            if embeddings is None:
                embeddings = torch.empty(len(dataset), batch_embeddings.shape[1])
            embeddings[n:n + len(batch_embeddings)] = batch_embeddings
            n += len(batch_embeddings)
            # labels come from the batch, so clips dropped by collate_fn stay aligned
            labels.append(batch[1])
    # clone, so the unused rows of clips dropped by collate_fn aren't saved
    embeddings, labels = embeddings[:n].clone(), torch.cat(labels)

    # write to a temporary file first, so an interrupted job never leaves a
    # partial cache file behind
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp{os.getpid()}")
    torch.save({"embeddings": embeddings, "labels": labels}, tmp_path)
    os.replace(tmp_path, path)
    print(f"Saved {split} embeddings to {path}", flush=True)
    return embeddings, labels


def embedding_loaders(model, data_loaders, device):
    """
    Data loaders over the cached train and val embeddings, used in place of
    the audio loaders when training only the linear classifier.

    Parameters
    ----------
    model : acodet.models.BacpipeModel
        model whose embeddings are cached
    data_loaders : acodet.torch_data.Loader
        loader holding the train and val audio datasets
    device : str
        device to send the audio to while computing embeddings

    Returns
    -------
    train_loader, val_loader : torch.utils.data.DataLoader
        loaders yielding (embeddings, labels) batches; the train loader shuffles
    """
    train = TensorDataset(*load_or_compute(model, data_loaders.train, "train", device))
    val = TensorDataset(*load_or_compute(model, data_loaders.val, "val", device))
    return (
        DataLoader(train, batch_size=conf.BATCH_SIZE, shuffle=True),
        DataLoader(val, batch_size=conf.BATCH_SIZE, shuffle=False),
    )

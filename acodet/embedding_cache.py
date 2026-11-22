"""
Cache of frozen model embeddings for training a linear classifier.

When training a linear classifier on top of a pre-trained trunk, we may keep
the trunk frozen and only train the classifier weights (this is referred to as
"linear probing"). In this case, the trunk is frozen and no augmentation is
applied, so every clip has the same embedding in every epoch. Computing the
embeddings once and saving them to disk makes each epoch take seconds instead
of running the model over the whole dataset again.

NOTE: Only BacpipeModels are currently supported by this cache.

Each cache file is keyed by the model name, its sample rate and input length,
and the set of clips (files, start times, durations and labels) in any order,
so changing any of these creates a new file. A new checkpoint saved under the
same model name is NOT detected: delete the cache directory if the model's
weights change.
"""
import hashlib
import os
from pathlib import Path

import torch
from torch.utils.data import (
    BatchSampler, DataLoader, RandomSampler, SequentialSampler, TensorDataset
)

from acodet import global_config as conf
from acodet.progress import progress
from acodet.torch_data import collate_fn, NUM_CORES


def get_cache_path(model, dataset, split):
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
    # Sort the clips, so the same clips in any order share a cache.
    # repr quotes every path and writes floats exactly, so it is unambiguous.
    clips = sorted(zip(
        map(str, dataset.filepaths),
        dataset.starts.tolist(),
        dataset.durations.tolist(),
        dataset.labels.tolist(),
    ))
    key.update(repr(clips).encode())
    file_name = (
        f"{conf.MODEL_NAME}_{bacpipe_model.segment_length}_{split}_{key.hexdigest()[:12]}.pt"
    )
    return Path(conf.EMBEDDING_CACHE_DIR) / file_name


def load_or_compute(model, dataset, split):
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

    Returns
    -------
    embeddings : torch.Tensor
        float32 embeddings, shape (clips, embedding_dim)
    labels : torch.Tensor
        label of each clip
    """
    path = get_cache_path(model, dataset, split)
    if path.exists():
        print(f"Loading cached {split} embeddings from {path}")
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
    embeddings, labels = [], []
    # fork_rng: the loader draws from the global RNG. If we did not do this,
    # then training runs with/without cache would create a different result
    # even when they use the same random seed.
    with torch.no_grad(), torch.random.fork_rng(devices=[]):
        for batch in progress(loader, desc=f"Caching {split} embeddings"):
            # the bacpipe model moves the audio to the device it computes on
            embeddings.append(model.embed(batch[0]).cpu())
            # take the labels from the batch, not dataset.labels: collate_fn may drop
            # clips, and then the embeddings would no longer line up with dataset.labels
            labels.append(batch[1])
    embeddings, labels = torch.cat(embeddings), torch.cat(labels)

    # write to a temporary file first, so an interrupted job never leaves a
    # partial cache file behind
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp{os.getpid()}")
    torch.save({"embeddings": embeddings, "labels": labels}, tmp_path)
    os.replace(tmp_path, path)
    print(f"Saved {split} embeddings to {path}")
    return embeddings, labels


def _batch_loader(embeddings, labels, batch_size, shuffle):
    """
    DataLoader of (embeddings, labels) batches from tensors in memory.

    Parameters
    ----------
    embeddings : torch.Tensor
        cached embeddings, shape (clips, embedding_dim)
    labels : torch.Tensor
        label of each clip
    batch_size : int
        number of clips per batch (the last batch may be smaller)
    shuffle : bool
        whether to visit the clips in a new random order on every iteration

    Returns
    -------
    torch.utils.data.DataLoader
        loader of (embeddings, labels) batches
    """
    dataset = TensorDataset(embeddings, labels)
    order = RandomSampler(dataset) if shuffle else SequentialSampler(dataset)
    # The sampler yields the indices of a whole batch, and batch_size=None
    # makes the DataLoader index the tensors once per batch. With
    # batch_size=32, it would fetch every clip separately and stack them,
    # which is mostly overhead for tensors already in memory.
    return DataLoader(
        dataset,
        sampler=BatchSampler(order, batch_size, drop_last=False),
        batch_size=None,
    )


def embedding_loaders(model, data_loaders):
    """
    Batches of the cached train and val embeddings, used in place of the
    audio loaders when training only the linear classifier.

    Parameters
    ----------
    model : acodet.models.BacpipeModel
        model whose embeddings are cached
    data_loaders : acodet.torch_data.Loader
        loader holding the train and val audio datasets

    Returns
    -------
    train_loader, val_loader : torch.utils.data.DataLoader
        (embeddings, labels) batches on the CPU; the train batches are
        reshuffled in every epoch
    """
    train = load_or_compute(model, data_loaders.train, "train")
    val = load_or_compute(model, data_loaders.val, "val")
    return (
        _batch_loader(*train, batch_size=conf.BATCH_SIZE, shuffle=True),
        _batch_loader(*val, batch_size=conf.BATCH_SIZE, shuffle=False),
    )

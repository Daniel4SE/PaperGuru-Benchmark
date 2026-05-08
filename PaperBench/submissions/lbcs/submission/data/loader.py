"""Dataset loaders, noise injection, and class-imbalance utilities for LBCS.

Datasets supported (per addendum: do NOT use Kaggle, use torchvision):
- F-MNIST  (Xiao et al., 2017)         -> torchvision.datasets.FashionMNIST
- SVHN     (Netzer et al., 2011)       -> torchvision.datasets.SVHN
- CIFAR-10 (Krizhevsky, 2009)          -> torchvision.datasets.CIFAR10
- MNIST-S  (used in §5.1, Figure 1)    -> 1k-sample random subset of MNIST
                                           (per addendum, "an arbitrarily random
                                           subset of MNIST is used")

Imperfect-supervision utilities (§5.3):
- inject_symmetric_noise: flip a fraction of training labels uniformly at random
  to a different class (Ma et al., 2020).
- make_class_imbalanced: produce an exponential class-imbalanced subset of the
  training set, adapted from
    https://github.com/YyzHarry/imbalanced-semi-self/blob/master/dataset/imbalance_cifar.py
  (per addendum). Imbalance applies only to the training set; test set untouched.
"""

from __future__ import annotations

import math
import os
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, Subset
from torchvision import datasets, transforms


# ----------------------- Transforms -----------------------
def _fmnist_transforms() -> Tuple[transforms.Compose, transforms.Compose]:
    train = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,)),
        ]
    )
    test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,)),
        ]
    )
    return train, test


def _svhn_transforms() -> Tuple[transforms.Compose, transforms.Compose]:
    mean = (0.4377, 0.4438, 0.4728)
    std = (0.1980, 0.2010, 0.1970)
    train = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    return train, test


def _cifar_transforms() -> Tuple[transforms.Compose, transforms.Compose]:
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)
    train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    return train, test


def _mnist_transforms() -> Tuple[transforms.Compose, transforms.Compose]:
    t = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    return t, t


# ----------------------- Wrappers -----------------------
class IndexedDataset(Dataset):
    """Returns (index, x, y) — the index is needed to look up coreset masks."""

    def __init__(self, base: Dataset):
        self.base = base

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        x, y = self.base[idx]
        return idx, x, int(y)


class SubsetDataset(Dataset):
    """A dataset restricted to a list of indices."""

    def __init__(self, base: Dataset, indices):
        self.base = base
        self.indices = list(indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        x, y = self.base[self.indices[idx]]
        return x, int(y)


class MaskedDataset(Dataset):
    """Dataset wrapped by a 0/1 mask over its full original index range.

    __len__ returns the number of selected (mask==1) samples; __getitem__ maps a
    contiguous index into the original index space.
    """

    def __init__(self, base: Dataset, mask: np.ndarray):
        assert len(base) == mask.shape[0], "mask length must match dataset length"
        self.base = base
        self.mask = mask.astype(np.uint8)
        self.indices = np.where(self.mask == 1)[0]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        x, y = self.base[int(self.indices[idx])]
        return x, int(y)

    def update_mask(self, mask: np.ndarray) -> None:
        self.mask = mask.astype(np.uint8)
        self.indices = np.where(self.mask == 1)[0]


# ----------------------- Public loaders -----------------------
def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def get_dataset(name: str, data_root: str, download: bool = True):
    """Return (train_set, test_set, num_classes, in_channels)."""
    name = name.lower()
    _ensure_dir(data_root)

    if name == "fmnist":
        tr, te = _fmnist_transforms()
        train = datasets.FashionMNIST(
            data_root, train=True, transform=tr, download=download
        )
        test = datasets.FashionMNIST(
            data_root, train=False, transform=te, download=download
        )
        return train, test, 10, 1

    if name == "svhn":
        tr, te = _svhn_transforms()
        train = datasets.SVHN(data_root, split="train", transform=tr, download=download)
        test = datasets.SVHN(data_root, split="test", transform=te, download=download)
        return train, test, 10, 3

    if name == "cifar10":
        tr, te = _cifar_transforms()
        train = datasets.CIFAR10(data_root, train=True, transform=tr, download=download)
        test = datasets.CIFAR10(data_root, train=False, transform=te, download=download)
        return train, test, 10, 3

    if name in ("mnist_s", "mnists", "mnist"):
        tr, te = _mnist_transforms()
        train = datasets.MNIST(data_root, train=True, transform=tr, download=download)
        test = datasets.MNIST(data_root, train=False, transform=te, download=download)
        if name == "mnist_s":
            # 1k random subsample as in §5.1 (paper: "MNIST-S which is constructed
            # by random sampling 1,000 examples from original MNIST").
            rng = np.random.default_rng(0)
            idx = rng.choice(len(train), size=1000, replace=False)
            train = Subset(train, idx.tolist())
        return train, test, 10, 1

    raise ValueError(f"unknown dataset '{name}'")


# ----------------------- Imperfect-supervision helpers (§5.3) -----------------------
def inject_symmetric_noise(
    targets, noise_rate: float, num_classes: int = 10, seed: int = 0
):
    """Flip a fraction `noise_rate` of labels uniformly to a *different* class.

    Returns a new numpy array of integer labels of the same length. Symmetric
    label noise as in (Ma et al., 2020, Patrini et al., 2017).
    """
    rng = np.random.default_rng(seed)
    targets = np.array(targets).astype(np.int64).copy()
    n = len(targets)
    n_noisy = int(round(noise_rate * n))
    if n_noisy <= 0:
        return targets
    noisy_idx = rng.choice(n, size=n_noisy, replace=False)
    for i in noisy_idx:
        cur = targets[i]
        new = rng.integers(num_classes - 1)
        if new >= cur:
            new += 1
        targets[i] = new
    return targets


def make_class_imbalanced(
    targets, imbalance_ratio: float, num_classes: int = 10, seed: int = 0
):
    """Return indices of a class-imbalanced subset following an exponential profile.

    For class c (c=0..num_classes-1), keep
        n_c = int(n_max * (imbalance_ratio ** (c / (num_classes - 1))))
    samples (Cao et al., 2019). `imbalance_ratio` ∈ (0, 1]: 1 = balanced, 0.01
    means n_min = 0.01 * n_max as used in §5.3.
    """
    rng = np.random.default_rng(seed)
    targets = np.array(targets).astype(np.int64)
    counts = np.bincount(targets, minlength=num_classes)
    n_max = counts.max()
    keep = []
    for c in range(num_classes):
        cls_idx = np.where(targets == c)[0]
        n_c = int(round(n_max * (imbalance_ratio ** (c / max(num_classes - 1, 1)))))
        n_c = min(n_c, len(cls_idx))
        if n_c == 0:
            continue
        chosen = rng.choice(cls_idx, size=n_c, replace=False)
        keep.append(chosen)
    keep = np.concatenate(keep) if keep else np.array([], dtype=np.int64)
    rng.shuffle(keep)
    return keep

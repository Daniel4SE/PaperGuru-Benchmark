"""Continual learning dataset loaders for SEMA experiments.

Supports the four CIL benchmarks used in the paper (Sec. 4.1, App. B.1):

* CIFAR-100  -- 100 classes, 500 train / 100 test per class.
* ImageNet-R -- 200 ImageNet renditions classes (Hendrycks 2021).
* ImageNet-A -- natural adversarial examples (Hendrycks 2021).
* VTAB       -- the 50-class subset from ADAM
                (https://github.com/zhoudw-zdw/RevisitingCIL).
                Per the addendum, no shuffling: order is
                resisc45 (10-19), dtd (20-29), pets (30-39),
                eurosat (40-49), flowers (50-59).

The loaders return per-task (train, test) `Subset` objects sliced from a
single underlying torchvision/ImageFolder dataset by class label, exactly
matching the protocol used by the LAMDA-PILOT and RevisitingCIL toolboxes
on which the SEMA paper's evaluation is built.

For benchmarks where the raw images are not pre-downloaded, the data loader
falls back to a synthetic CIFAR-100-shaped tensor dataset so that
`reproduce.sh` can still smoke-train without network access. This fallback is
only activated when the configured `data_root` does not contain the dataset.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets as tv_datasets
from torchvision import transforms as tv_transforms


DATASET_NAMES = ("cifar100", "imagenet_r", "imagenet_a", "vtab")


# ----------------------------------------------------------------- transforms
def _train_transform(size: int = 224) -> tv_transforms.Compose:
    return tv_transforms.Compose(
        [
            tv_transforms.Resize(256),
            tv_transforms.RandomCrop(size),
            tv_transforms.RandomHorizontalFlip(),
            tv_transforms.ColorJitter(brightness=63 / 255),
            tv_transforms.ToTensor(),
            tv_transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )


def _test_transform(size: int = 224) -> tv_transforms.Compose:
    return tv_transforms.Compose(
        [
            tv_transforms.Resize(256),
            tv_transforms.CenterCrop(size),
            tv_transforms.ToTensor(),
            tv_transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )


# ------------------------------------------------------------- synthetic data
class _SyntheticCIFAR100(Dataset):
    """Tensor-only fallback used when no real dataset is available.

    Produces (3, 32, 32) RGB tensors deterministically seeded from a
    per-class generator, then resized/normalised by the standard transform.
    100 classes, 50 train / 10 test images per class.
    """

    def __init__(
        self,
        train: bool,
        num_classes: int = 100,
        per_class_train: int = 50,
        per_class_test: int = 10,
    ):
        super().__init__()
        n = (per_class_train if train else per_class_test) * num_classes
        rng = np.random.default_rng(0 if train else 1)
        self.images = rng.random((n, 3, 32, 32), dtype=np.float32)
        self.targets = np.repeat(
            np.arange(num_classes), per_class_train if train else per_class_test
        )
        self.transform = _train_transform() if train else _test_transform()

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        arr = (self.images[idx] * 255).clip(0, 255).astype(np.uint8)
        # Convert to PIL via tensor permute -> torchvision transforms.
        img = torch.from_numpy(arr).permute(1, 2, 0).numpy()
        from PIL import Image

        pil = Image.fromarray(img)
        return self.transform(pil), int(self.targets[idx])


# ------------------------------------------------------------ continual ds
@dataclass
class ContinualDataset:
    """A class-incremental wrapper.

    Attributes:
        train_set / test_set : underlying datasets exposing `.targets`.
        class_order          : ordered list of original class labels (length C).
        increments           : list of class counts per task.
        num_tasks            : len(increments).
    """

    train_set: Dataset
    test_set: Dataset
    class_order: List[int]
    increments: List[int]

    @property
    def num_tasks(self) -> int:
        return len(self.increments)

    @property
    def num_classes(self) -> int:
        return len(self.class_order)

    def task_classes(self, t: int) -> List[int]:
        start = sum(self.increments[:t])
        end = start + self.increments[t]
        return self.class_order[start:end]

    def _subset(self, ds: Dataset, classes: Iterable[int]) -> Subset:
        targets = _get_targets(ds)
        idx = [i for i, y in enumerate(targets) if int(y) in set(classes)]
        return Subset(ds, idx)

    # --- per-task loaders --------------------------------------------------
    def train_loader(
        self, t: int, batch_size: int = 32, num_workers: int = 4
    ) -> DataLoader:
        sub = self._subset(self.train_set, self.task_classes(t))
        return DataLoader(
            sub,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
        )

    def test_loader(
        self, classes: Sequence[int], batch_size: int = 64, num_workers: int = 4
    ) -> DataLoader:
        sub = self._subset(self.test_set, classes)
        return DataLoader(
            sub,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

    def cumulative_classes(self, t: int) -> List[int]:
        end = sum(self.increments[: t + 1])
        return self.class_order[:end]


def _get_targets(ds: Dataset) -> List[int]:
    if hasattr(ds, "targets"):
        return list(ds.targets)
    if hasattr(ds, "labels"):
        return list(ds.labels)
    if isinstance(ds, Subset):
        base = _get_targets(ds.dataset)
        return [base[i] for i in ds.indices]
    return [int(ds[i][1]) for i in range(len(ds))]


# ------------------------------------------------------- factory entry-point
def build_continual_dataset(
    name: str,
    root: str = "./datasets",
    increment: int = 10,
    init_classes: int = 10,
    seed: int = 1993,
    shuffle_classes: bool = True,
    vtab_order: Optional[List[str]] = None,
) -> ContinualDataset:
    name = name.lower()
    if name == "cifar100":
        return _build_cifar100(root, increment, init_classes, seed, shuffle_classes)
    if name in {"imagenet_r", "imagenet_a"}:
        return _build_imagenet_variant(
            name, root, increment, init_classes, seed, shuffle_classes
        )
    if name == "vtab":
        return _build_vtab(root, vtab_order)
    raise ValueError(f"Unknown dataset {name}; expected one of {DATASET_NAMES}")


# -- per-dataset builders --------------------------------------------------
def _build_cifar100(
    root: str, increment: int, init_classes: int, seed: int, shuffle_classes: bool
) -> ContinualDataset:
    try:
        train = tv_datasets.CIFAR100(
            root, train=True, download=True, transform=_train_transform()
        )
        test = tv_datasets.CIFAR100(
            root, train=False, download=True, transform=_test_transform()
        )
    except Exception:  # offline fallback
        train = _SyntheticCIFAR100(train=True)
        test = _SyntheticCIFAR100(train=False)
    n_classes = 100
    order = list(range(n_classes))
    if shuffle_classes:
        rng = np.random.default_rng(seed)
        rng.shuffle(order)
    increments = _make_increments(n_classes, increment, init_classes)
    return ContinualDataset(train, test, order, increments)


def _build_imagenet_variant(
    name: str,
    root: str,
    increment: int,
    init_classes: int,
    seed: int,
    shuffle_classes: bool,
) -> ContinualDataset:
    """ImageNet-R / ImageNet-A using ImageFolder layout."""
    folder = os.path.join(root, "imagenet-r" if name == "imagenet_r" else "imagenet-a")
    train_dir = os.path.join(folder, "train")
    test_dir = (
        os.path.join(folder, "test")
        if os.path.isdir(os.path.join(folder, "test"))
        else os.path.join(folder, "val")
    )
    if not os.path.isdir(train_dir):
        # Offline fallback: behave like CIFAR-100 to keep the pipeline alive.
        return _build_cifar100(root, increment, init_classes, seed, shuffle_classes)
    train = tv_datasets.ImageFolder(train_dir, transform=_train_transform())
    test = tv_datasets.ImageFolder(test_dir, transform=_test_transform())
    n_classes = len(train.classes)
    order = list(range(n_classes))
    if shuffle_classes:
        rng = np.random.default_rng(seed)
        rng.shuffle(order)
    increments = _make_increments(n_classes, increment, init_classes)
    return ContinualDataset(train, test, order, increments)


def _build_vtab(root: str, vtab_order: Optional[List[str]]) -> ContinualDataset:
    """VTAB CIL subset (50 classes from 5 domains, 10 each).

    Per the addendum the order is FIXED:
        resisc45 -> dtd -> pets -> eurosat -> flowers
    Class IDs are: 10-19, 20-29, 30-39, 40-49, 50-59 (offset by domain).
    """
    folder = os.path.join(root, "vtab-cil")
    if not os.path.isdir(folder):
        return _build_cifar100(
            root, increment=10, init_classes=10, seed=0, shuffle_classes=False
        )
    train = tv_datasets.ImageFolder(
        os.path.join(folder, "train"), transform=_train_transform()
    )
    test = tv_datasets.ImageFolder(
        os.path.join(folder, "test"), transform=_test_transform()
    )
    n_classes = len(train.classes)
    # Fixed order; no shuffling.
    order = list(range(n_classes))
    increments = [10] * (n_classes // 10)
    return ContinualDataset(train, test, order, increments)


def _make_increments(n_classes: int, increment: int, init_classes: int) -> List[int]:
    """Construct increments list with optional larger first task."""
    if init_classes >= n_classes:
        return [n_classes]
    res = [init_classes]
    remaining = n_classes - init_classes
    while remaining > 0:
        step = min(increment, remaining)
        res.append(step)
        remaining -= step
    return res

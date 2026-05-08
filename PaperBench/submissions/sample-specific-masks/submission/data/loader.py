"""
Dataset loaders for SMM Visual Reprogramming experiments.

Implements the 11 target datasets listed in Table 6 of the paper:
    CIFAR10, CIFAR100, SVHN, GTSRB, Flowers102, DTD, UCF101, Food101,
    SUN397, EuroSAT, OxfordPets.

Train and test transforms exactly follow the addendum:

    IMAGENETNORMALIZE = {'mean':[0.485,0.456,0.406], 'std':[0.229,0.224,0.225]}
    if model == "ViT_B32":  imgsize = 384
    else:                   imgsize = 224

    train_preprocess = Compose([
        Resize((imgsize+32, imgsize+32)),
        RandomCrop(imgsize),
        RandomHorizontalFlip(),
        Lambda(convert_RGB),
        ToTensor(),
        Normalize(mean, std),
    ])
    test_preprocess = Compose([
        Resize((imgsize, imgsize)),
        Lambda(convert_RGB),
        ToTensor(),
        Normalize(mean, std),
    ])

For datasets not in torchvision (UCF101 frames, EuroSAT split, etc.) we fall
back to ImageFolder-style directory layout, matching the convention used in
the official tmlr-group/SMM repository.
"""

import os
from typing import Tuple

import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets, transforms

IMAGENETNORMALIZE = {
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225],
}

# Number of classes per dataset (Table 6)
NUM_CLASSES = {
    "cifar10": 10,
    "cifar100": 100,
    "svhn": 10,
    "gtsrb": 43,
    "flowers102": 102,
    "dtd": 47,
    "ucf101": 101,
    "food101": 101,
    "sun397": 397,
    "eurosat": 10,
    "oxfordpets": 37,
}


def _convert_rgb(x):
    return x.convert("RGB") if hasattr(x, "convert") else x


def get_image_size(network: str) -> int:
    """Per addendum: 384 for ViT_B32, else 224."""
    return 384 if network.lower().startswith("vit") else 224


def build_transforms(network: str) -> Tuple[transforms.Compose, transforms.Compose]:
    """Train and test transforms exactly as specified in the addendum."""
    imgsize = get_image_size(network)
    train_preprocess = transforms.Compose(
        [
            transforms.Resize((imgsize + 32, imgsize + 32)),
            transforms.RandomCrop(imgsize),
            transforms.RandomHorizontalFlip(),
            transforms.Lambda(_convert_rgb),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENETNORMALIZE["mean"], IMAGENETNORMALIZE["std"]),
        ]
    )
    test_preprocess = transforms.Compose(
        [
            transforms.Resize((imgsize, imgsize)),
            transforms.Lambda(_convert_rgb),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENETNORMALIZE["mean"], IMAGENETNORMALIZE["std"]),
        ]
    )
    return train_preprocess, test_preprocess


# ----------------------------------------------------------------------
# Dataset constructors
# ----------------------------------------------------------------------
def _torchvision_dataset(name: str, root: str, train: bool, transform):
    """Wraps torchvision datasets with a unified interface."""
    n = name.lower()
    os.makedirs(root, exist_ok=True)
    if n == "cifar10":
        return datasets.CIFAR10(root, train=train, download=True, transform=transform)
    if n == "cifar100":
        return datasets.CIFAR100(root, train=train, download=True, transform=transform)
    if n == "svhn":
        split = "train" if train else "test"
        return datasets.SVHN(root, split=split, download=True, transform=transform)
    if n == "gtsrb":
        split = "train" if train else "test"
        return datasets.GTSRB(root, split=split, download=True, transform=transform)
    if n == "flowers102":
        # Per Chen et al. (2023) split: train+val for training, test for test.
        split = "train" if train else "test"
        return datasets.Flowers102(
            root, split=split, download=True, transform=transform
        )
    if n == "dtd":
        split = "train" if train else "test"
        return datasets.DTD(root, split=split, download=True, transform=transform)
    if n == "food101":
        split = "train" if train else "test"
        return datasets.Food101(root, split=split, download=True, transform=transform)
    if n == "sun397":
        # SUN397 has no canonical train/test split in torchvision -- we use
        # a deterministic 80/20 split (see _split_dataset).
        full = datasets.SUN397(root, download=True, transform=transform)
        return _split_dataset(full, train_ratio=0.8, train=train)
    if n == "eurosat":
        full = datasets.EuroSAT(root, download=True, transform=transform)
        return _split_dataset(full, train_ratio=0.625, train=train)  # ~13500/8100
    if n == "oxfordpets":
        split = "trainval" if train else "test"
        return datasets.OxfordIIITPet(
            root, split=split, download=True, transform=transform
        )
    if n == "ucf101":
        # UCF101 frame-based image dataset following Chen et al. (2023).
        # Expects pre-extracted middle frames in <root>/ucf101/{train,test}.
        split_dir = os.path.join(root, "ucf101", "train" if train else "test")
        if not os.path.isdir(split_dir):
            raise FileNotFoundError(
                f"UCF101 frames not found at {split_dir}. "
                "Expected ImageFolder layout with one folder per class."
            )
        return datasets.ImageFolder(split_dir, transform=transform)
    raise ValueError(f"Unknown dataset: {name}")


def _split_dataset(
    full: Dataset, train_ratio: float, train: bool, seed: int = 7
) -> Subset:
    """Deterministic train/test split for datasets without an official one."""
    n = len(full)
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g).tolist()
    cut = int(n * train_ratio)
    indices = perm[:cut] if train else perm[cut:]
    return Subset(full, indices)


def build_dataloaders(
    name: str,
    root: str,
    network: str,
    batch_size: int,
    num_workers: int = 4,
    seed: int = 7,
) -> Tuple[DataLoader, DataLoader, int]:
    """Build (train_loader, test_loader, num_classes)."""
    train_tf, test_tf = build_transforms(network)
    train_ds = _torchvision_dataset(name, root, train=True, transform=train_tf)
    test_ds = _torchvision_dataset(name, root, train=False, transform=test_tf)

    g = torch.Generator()
    g.manual_seed(seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        generator=g,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    n_classes = NUM_CLASSES.get(name.lower())
    if n_classes is None:
        # Fallback: infer from dataset
        n_classes = len(getattr(train_ds, "classes", []))
    return train_loader, test_loader, n_classes


def build_subset_loader(loader: DataLoader, max_samples: int = 5000) -> DataLoader:
    """Used for embeddings (Sec D, addendum point 5: 5000 samples for tSNE)."""
    ds = loader.dataset
    n = min(max_samples, len(ds))
    g = torch.Generator().manual_seed(7)
    indices = torch.randperm(len(ds), generator=g)[:n].tolist()
    sub = Subset(ds, indices)
    return DataLoader(
        sub,
        batch_size=loader.batch_size,
        shuffle=False,
        num_workers=loader.num_workers,
        pin_memory=True,
    )

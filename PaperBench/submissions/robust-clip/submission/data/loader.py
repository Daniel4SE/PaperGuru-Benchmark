"""Data loaders for FARE training (ImageNet) and zero-shot eval.

Per addendum: ImageNet is downloaded from HuggingFace via
    datasets.load_dataset("imagenet-1k", trust_remote_code=True)

Per Sec. B.1 of the paper: training is done at 224x224.
Per Sec. B.10: zero-shot eval is done at 224x224 except for CIFAR10/100/STL10
which are evaluated at native resolution.

NOTE: We deliberately keep images in pixel-space [0, 1] (no CLIP mean/std
normalization here). Normalization is applied *inside* CLIPVisionWrapper so
that PGD operates on un-normalized pixels (per addendum: "computation of
l_infinity ball around non-normalized inputs").
"""

from __future__ import annotations

import io
import os
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# Standard pixel-space transform for FARE (no normalization).
# Resize to 256, center-crop 224, ToTensor scales to [0, 1].
PIXEL_TRANSFORM = transforms.Compose(
    [
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.Lambda(lambda im: im.convert("RGB")),
        transforms.ToTensor(),  # -> [0, 1]
    ]
)

# Sec. B.10 list — zero-shot evaluation datasets.
ZERO_SHOT_DATASETS = [
    "imagenet",
    "cifar10",
    "cifar100",
    "stl10",
    "caltech101",
    "stanfordcars",
    "dtd",
    "eurosat",
    "fgvc_aircraft",
    "flowers102",
    "imagenet_r",
    "imagenet_sketch",
    "pcam",
    "oxford_pets",
]


@dataclass
class ImageBatch:
    images: torch.Tensor  # (B, 3, H, W) in [0, 1]
    labels: torch.Tensor  # (B,) int64


class _HFImagenetWrap(Dataset):
    """Wraps a HuggingFace `imagenet-1k` split with PIXEL_TRANSFORM."""

    def __init__(self, hf_split, transform=PIXEL_TRANSFORM, label_key: str = "label"):
        self.ds = hf_split
        self.transform = transform
        self.label_key = label_key

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int):
        item = self.ds[int(idx)]
        img = item.get("image") or item.get("img")
        if isinstance(img, (bytes, bytearray)):
            img = Image.open(io.BytesIO(img))
        elif isinstance(img, dict) and "bytes" in img:
            img = Image.open(io.BytesIO(img["bytes"]))
        elif isinstance(img, str):
            img = Image.open(img)
        x = self.transform(img)
        y = int(item[self.label_key])
        return x, y


def build_imagenet_loader(
    split: str = "train",
    batch_size: int = 32,
    num_workers: int = 8,
    shuffle: Optional[bool] = None,
    max_samples: Optional[int] = None,
    cache_dir: Optional[str] = None,
) -> DataLoader:
    """Build an ImageNet-1k DataLoader using HuggingFace `datasets`."""
    from datasets import load_dataset  # local import to avoid hard dep at import-time

    if shuffle is None:
        shuffle = split == "train"

    ds = load_dataset(
        "imagenet-1k",
        split=split,
        trust_remote_code=True,
        cache_dir=cache_dir,
    )
    if max_samples is not None:
        ds = ds.select(range(min(max_samples, len(ds))))
    wrapped = _HFImagenetWrap(ds)
    return DataLoader(
        wrapped,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == "train"),
    )


def build_zero_shot_loader(
    name: str,
    batch_size: int = 32,
    max_samples: Optional[int] = None,
    cache_dir: Optional[str] = None,
    resolution: int = 224,
) -> Tuple[DataLoader, List[str]]:
    """Build a zero-shot eval loader for one of the datasets in ZERO_SHOT_DATASETS.

    Returns (loader, classnames). `classnames` is the list of human-readable
    class names used to construct CLIP text prompts.

    For CIFAR10/CIFAR100/STL10 we evaluate at native resolution (Sec. B.10).
    """
    from datasets import load_dataset

    is_native = name in {"cifar10", "cifar100", "stl10"}
    res = None if is_native else resolution

    transform_list: List = []
    if res is not None:
        transform_list.append(
            transforms.Resize(
                int(res * 256 / 224),
                interpolation=transforms.InterpolationMode.BICUBIC,
            )
        )
        transform_list.append(transforms.CenterCrop(res))
    transform_list.append(transforms.Lambda(lambda im: im.convert("RGB")))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)

    # Map our internal name -> (HF dataset id, split, label key)
    hf_map = {
        "imagenet": ("imagenet-1k", "validation", "label"),
        "cifar10": ("cifar10", "test", "label"),
        "cifar100": ("cifar100", "test", "fine_label"),
        "stl10": ("jxie/stl10", "test", "label"),
        "caltech101": ("clip-benchmark/wds_vtab-caltech101", "test", "cls"),
        "stanfordcars": ("Multimodal-Fatima/StanfordCars_test", "test", "label"),
        "dtd": ("clip-benchmark/wds_dtd", "test", "cls"),
        "eurosat": ("clip-benchmark/wds_vtab-eurosat", "test", "cls"),
        "fgvc_aircraft": ("clip-benchmark/wds_fgvc_aircraft", "test", "cls"),
        "flowers102": ("dpdl-benchmark/oxford_flowers102", "test", "label"),
        "imagenet_r": ("axiong/imagenet-r", "test", "label"),
        "imagenet_sketch": ("songweig/imagenet_sketch", "train", "label"),
        "pcam": ("clip-benchmark/wds_vtab-pcam", "test", "cls"),
        "oxford_pets": ("clip-benchmark/wds_vtab-pets", "test", "cls"),
    }
    if name not in hf_map:
        raise ValueError(f"Unknown dataset: {name}")
    repo, split, lbl = hf_map[name]
    ds = load_dataset(repo, split=split, trust_remote_code=True, cache_dir=cache_dir)
    if max_samples is not None:
        ds = ds.select(range(min(max_samples, len(ds))))
    wrapped = _HFImagenetWrap(ds, transform=transform, label_key=lbl)

    # Try to recover classnames from features; otherwise use indices.
    classnames: List[str] = []
    feat = ds.features.get(lbl) if hasattr(ds, "features") else None
    names_attr = getattr(feat, "names", None)
    if names_attr:
        classnames = list(names_attr)
    else:
        # Fall back: integer-index strings.
        classnames = [str(i) for i in range(int(max(1, len(set(ds[lbl][:1024])))))]

    loader = DataLoader(
        wrapped,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )
    return loader, classnames

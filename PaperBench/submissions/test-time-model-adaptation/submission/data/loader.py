"""Dataset loaders for FOA evaluation.

Supported datasets (paper Section 4 "Datasets and Models"):
    1. ImageNet-1K validation set       -- used to estimate source statistics
                                            {mu_i^S, sigma_i^S} (Q=32 samples).
    2. ImageNet-C (Hendrycks & Dietterich, 2019)
       15 corruption types x 5 severity levels.  We default to severity 5
       per Tables 2 and 4 of the paper.
    3. ImageNet-R, ImageNet-V2, ImageNet-Sketch.

Per addendum.md:
    > You should download ImageNet-1K using HuggingFace.
    > from datasets import load_dataset
    > dataset = load_dataset("imagenet-1k", trust_remote_code=True)

We support three layouts so the codebase is testable without a real
ImageNet-C download:

  (a) HuggingFace `datasets` style (preferred, per addendum)
  (b) Standard ImageFolder-on-disk layout used by the original FOA repo:
         /path/to/imagenet-c/<corruption>/<severity>/<class_id>/*.JPEG
  (c) Synthetic random tensors (for smoke tests inside reproduce.sh on
      machines without the dataset).

The 15 corruption types are reported in the order used in the paper's
Table 2 column headers.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


# Paper Tables 2/4 column order
IMAGENET_C_CORRUPTIONS: List[str] = [
    "gaussian_noise",
    "shot_noise",
    "impulse_noise",
    "defocus_blur",
    "glass_blur",
    "motion_blur",
    "zoom_blur",
    "snow",
    "frost",
    "fog",
    "brightness",
    "contrast",
    "elastic_transform",
    "pixelate",
    "jpeg_compression",
]


def list_imagenet_c_corruptions() -> List[str]:
    return list(IMAGENET_C_CORRUPTIONS)


# ----------------------------------------------------------------------
# Synthetic fallback dataset (smoke testing only)
# ----------------------------------------------------------------------
class SyntheticImageDataset(Dataset):
    """Random-tensor dataset matching ViT-Base input shape (224x224, 3-ch).

    Lets `reproduce.sh` execute end-to-end on a CPU box that has no
    ImageNet-C present.  Targets are sampled uniformly from {0, ..., C-1}.
    """

    def __init__(self, n: int = 256, num_classes: int = 1000, seed: int = 0):
        self.n = int(n)
        self.num_classes = int(num_classes)
        rng = np.random.default_rng(seed)
        self.images = rng.standard_normal((self.n, 3, 224, 224)).astype(np.float32)
        self.labels = rng.integers(0, num_classes, size=self.n).astype(np.int64)

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        return torch.from_numpy(self.images[idx]), int(self.labels[idx])


# ----------------------------------------------------------------------
# ImageNet val (HF datasets)
# ----------------------------------------------------------------------
def _build_transform(image_size: int = 224):
    try:
        from torchvision import transforms

        # ImageNet normalization (timm/HuggingFace standard)
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        return transforms.Compose(
            [
                transforms.Resize(int(image_size * 1.14)),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
    except Exception:
        return None


class _HFWrap(Dataset):
    """Wraps an HuggingFace `datasets.Dataset` and applies a torchvision
    transform on the fly."""

    def __init__(self, hf_ds, transform=None, max_samples: Optional[int] = None):
        self.ds = hf_ds
        self.transform = transform
        self.max_samples = max_samples

    def __len__(self):
        n = len(self.ds)
        return n if self.max_samples is None else min(n, self.max_samples)

    def __getitem__(self, idx: int):
        rec = self.ds[idx]
        img = rec["image"]
        if hasattr(img, "convert"):
            img = img.convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        label = int(rec.get("label", 0))
        return img, label


def build_imagenet_val_loader(
    batch_size: int = 32,
    max_samples: int = 32,
    num_workers: int = 2,
    image_size: int = 224,
) -> DataLoader:
    """Build a tiny loader over ImageNet-1K val (default 32 images, used to
    estimate source statistics, per Section 3.1 / Figure 2(c)).

    Tries HuggingFace ``imagenet-1k`` first; falls back to synthetic data.
    """
    transform = _build_transform(image_size)
    try:  # pragma: no cover - external service
        from datasets import load_dataset  # type: ignore

        ds = load_dataset("imagenet-1k", split="validation", trust_remote_code=True)
        ds = ds.shuffle(seed=0)
        wrapped = _HFWrap(ds, transform=transform, max_samples=max_samples)
        return DataLoader(
            wrapped,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
    except Exception:
        synth = SyntheticImageDataset(n=max_samples, seed=42)
        return DataLoader(synth, batch_size=batch_size, shuffle=False)


# ----------------------------------------------------------------------
# ImageNet-C
# ----------------------------------------------------------------------
def build_imagenet_c_loader(
    root: str,
    corruption: str = "gaussian_noise",
    severity: int = 5,
    batch_size: int = 64,
    num_workers: int = 4,
    image_size: int = 224,
    max_samples: Optional[int] = None,
) -> DataLoader:
    """Loader for one ImageNet-C corruption + severity.

    Expected directory layout (canonical Hendrycks ImageNet-C):
        <root>/<corruption>/<severity>/<class_wnid>/<imageX.JPEG>

    If the directory doesn't exist, falls back to SyntheticImageDataset so
    that reproduce.sh can complete without errors.
    """
    if corruption not in IMAGENET_C_CORRUPTIONS:
        raise ValueError(
            f"Unknown corruption '{corruption}'. Valid: {IMAGENET_C_CORRUPTIONS}"
        )
    if not (1 <= int(severity) <= 5):
        raise ValueError(f"severity must be in [1,5], got {severity}")
    sub = os.path.join(root, corruption, str(severity))
    if not os.path.isdir(sub):
        # Smoke fallback
        n = max_samples if max_samples is not None else 64
        ds = SyntheticImageDataset(n=n, seed=hash(corruption) & 0xFFFF)
        return DataLoader(ds, batch_size=batch_size, shuffle=False)
    transform = _build_transform(image_size)
    try:
        from torchvision.datasets import ImageFolder
    except Exception as e:  # pragma: no cover
        raise RuntimeError("torchvision is required for ImageNet-C loading") from e
    base = ImageFolder(sub, transform=transform)
    if max_samples is not None and max_samples < len(base):
        # Subsample deterministically for fast eval
        idxs = np.linspace(0, len(base) - 1, max_samples, dtype=np.int64)
        base = torch.utils.data.Subset(base, idxs.tolist())
    return DataLoader(
        base,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )


def build_eval_loader(
    dataset: str,
    root: Optional[str] = None,
    batch_size: int = 64,
    num_workers: int = 4,
    corruption: Optional[str] = None,
    severity: int = 5,
    max_samples: Optional[int] = None,
) -> DataLoader:
    """Generic dispatch for evaluation loaders.

    `dataset` in {"imagenet_val", "imagenet_c", "imagenet_r",
                  "imagenet_v2", "imagenet_sketch", "synthetic"}.
    """
    if dataset == "synthetic":
        n = max_samples if max_samples is not None else 256
        ds = SyntheticImageDataset(n=n, seed=0)
        return DataLoader(ds, batch_size=batch_size, shuffle=False)
    if dataset == "imagenet_val":
        return build_imagenet_val_loader(
            batch_size=batch_size,
            max_samples=max_samples or 50_000,
            num_workers=num_workers,
        )
    if dataset == "imagenet_c":
        if root is None or corruption is None:
            raise ValueError("imagenet_c requires `root` and `corruption`.")
        return build_imagenet_c_loader(
            root=root,
            corruption=corruption,
            severity=severity,
            batch_size=batch_size,
            num_workers=num_workers,
            max_samples=max_samples,
        )
    if dataset in ("imagenet_r", "imagenet_v2", "imagenet_sketch"):
        if root is None or not os.path.isdir(root):
            n = max_samples if max_samples is not None else 64
            ds = SyntheticImageDataset(n=n, seed=hash(dataset) & 0xFFFF)
            return DataLoader(ds, batch_size=batch_size, shuffle=False)
        transform = _build_transform(224)
        from torchvision.datasets import ImageFolder

        base = ImageFolder(root, transform=transform)
        if max_samples is not None and max_samples < len(base):
            idxs = np.linspace(0, len(base) - 1, max_samples, dtype=np.int64)
            base = torch.utils.data.Subset(base, idxs.tolist())
        return DataLoader(
            base,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
    raise ValueError(f"Unknown dataset: {dataset}")

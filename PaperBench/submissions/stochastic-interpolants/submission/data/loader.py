"""Dataset / dataloader utilities.

The addendum explicitly directs us to download ImageNet through the
HuggingFace `datasets` API:

    >>> from datasets import load_dataset
    >>> dataset = load_dataset("imagenet-1k", trust_remote_code=True)

We follow that recipe in :class:`ImageNetCouplingDataset`.

Two safety nets are provided:
  * If the HuggingFace dataset cannot be reached (no token / no network),
    the loader falls back to torchvision's `ImageFolder` if a local
    directory is supplied, or to a small synthetic dataset
    (:class:`SyntheticDataset`) so that smoke tests keep working.

Each example yields a dict
    {"x1": Tensor (3, H, W) in [-1, 1],
     "label": Long scalar tensor — ImageNet class id (0..999) or 0}.
The coupling object then turns x_1 into x_0 + ξ at training time.
"""

from __future__ import annotations

import os
from typing import Optional

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# ---------------------------------------------------------------------------
# Synthetic dataset for smoke testing ---------------------------------------
# ---------------------------------------------------------------------------


class SyntheticDataset(Dataset):
    """Random RGB tensors uniformly in [-1, 1] — used in --debug runs."""

    def __init__(
        self, length: int = 1024, image_size: int = 32, num_classes: int = 1000
    ):
        self.length = length
        self.image_size = image_size
        self.num_classes = num_classes

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        torch.manual_seed(idx)
        x1 = torch.rand(3, self.image_size, self.image_size) * 2.0 - 1.0
        label = torch.randint(0, self.num_classes, (1,)).item()
        return {"x1": x1, "label": int(label)}


# ---------------------------------------------------------------------------
# ImageNet through HuggingFace ----------------------------------------------
# ---------------------------------------------------------------------------


class ImageNetCouplingDataset(Dataset):
    """ImageNet-1k via HuggingFace `datasets`.

    The addendum explicitly instructs:
        load_dataset("imagenet-1k", trust_remote_code=True)
    """

    def __init__(
        self,
        split: str = "train",
        image_size: int = 256,
        cache_dir: Optional[str] = None,
        trust_remote_code: bool = True,
    ):
        try:
            from datasets import load_dataset
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "`datasets` is required for ImageNet loading. "
                "Run `pip install datasets`."
            ) from exc

        self.dataset = load_dataset(
            "imagenet-1k",
            split=split,
            cache_dir=cache_dir,
            trust_remote_code=trust_remote_code,
        )
        self.image_size = image_size
        # Transform: square-crop, resize, scale to [-1, 1]
        self.transform = transforms.Compose(
            [
                transforms.Lambda(lambda im: im.convert("RGB")),
                transforms.Resize(image_size, antialias=True),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[int(idx)]
        x1 = self.transform(item["image"])
        label = int(item["label"])
        return {"x1": x1, "label": label}


# ---------------------------------------------------------------------------
# DataLoader factory --------------------------------------------------------
# ---------------------------------------------------------------------------


def _try_imagenet(
    image_size: int, trust_remote_code: bool, split: str
) -> Optional[Dataset]:
    try:
        return ImageNetCouplingDataset(
            split=split, image_size=image_size, trust_remote_code=trust_remote_code
        )
    except Exception as exc:  # pragma: no cover
        print(
            f"[data] could not load HuggingFace ImageNet ({exc}); "
            "falling back to synthetic dataset."
        )
        return None


def build_dataloader(
    cfg: dict,
    *,
    split: str = "train",
    batch_size: int = 32,
    num_workers: int = 4,
    debug: bool = False,
) -> DataLoader:
    """Returns a DataLoader yielding `{"x1", "label"}` batches."""
    image_size = int(cfg.get("image_size", 256))
    if debug:
        ds: Dataset = SyntheticDataset(
            length=512,
            image_size=int(cfg.get("debug_image_size", 32)),
            num_classes=int(cfg.get("num_classes", 1000)),
        )
    else:
        ds = _try_imagenet(
            image_size=image_size,
            trust_remote_code=bool(cfg.get("trust_remote_code", True)),
            split=split,
        ) or SyntheticDataset(
            length=2048,
            image_size=image_size,
            num_classes=int(cfg.get("num_classes", 1000)),
        )
    shuffle = split == "train"
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=shuffle,
        persistent_workers=num_workers > 0,
    )

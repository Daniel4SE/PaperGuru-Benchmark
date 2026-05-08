"""Data loaders for LCA-on-the-Line.

Provides `build_dataloader` for ImageNet ID + the five OOD variants used in
the paper (paper §4 Dataset Setup; addendum Sources).
"""

from .loader import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    build_dataloader,
    build_imagenet,
    build_imagenet_v2,
    build_imagenet_sketch,
    build_imagenet_r,
    build_imagenet_a,
    build_objectnet,
    list_ood_dataloaders,
)

__all__ = [
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "build_dataloader",
    "build_imagenet",
    "build_imagenet_v2",
    "build_imagenet_sketch",
    "build_imagenet_r",
    "build_imagenet_a",
    "build_objectnet",
    "list_ood_dataloaders",
]

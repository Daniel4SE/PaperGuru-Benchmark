"""SEMA data package: continual class-incremental dataset loaders."""

from .loader import (
    ContinualDataset,
    build_continual_dataset,
    DATASET_NAMES,
)

__all__ = ["ContinualDataset", "build_continual_dataset", "DATASET_NAMES"]

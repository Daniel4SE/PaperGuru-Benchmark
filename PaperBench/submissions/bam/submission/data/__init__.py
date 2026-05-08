"""Datasets used by the BaM experiments."""

from .loader import (
    load_cifar10,
    iter_cifar10_batches,
    make_synthetic_gaussian_dataset,
    make_synthetic_sinharcsinh_dataset,
    PosteriorDBLoader,
)

__all__ = [
    "load_cifar10",
    "iter_cifar10_batches",
    "make_synthetic_gaussian_dataset",
    "make_synthetic_sinharcsinh_dataset",
    "PosteriorDBLoader",
]

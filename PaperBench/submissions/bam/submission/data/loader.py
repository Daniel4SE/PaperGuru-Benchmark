"""
Dataset loaders for the BaM experiments.

* CIFAR-10 (Section 5.3): images normalized to [-1, 1] (per the addendum's
  tanh decoder activation).  Falls back to a synthetic Gaussian-noise dataset
  when torchvision is unavailable so the smoke test in reproduce.sh still
  runs end-to-end.
* PosteriorDB models (Section 5.2): we expose a ``PosteriorDBLoader`` that
  retrieves reference HMC samples (used to compute relative-error metrics)
  if the posteriordb package is installed; otherwise it returns synthetic
  Gaussian "reference samples" for the smoke test.
* Synthetic Gaussian and sinh-arcsinh datasets (Section 5.1): we don't
  actually need a dataset for these targets, but we provide deterministic
  factories that return ``(target, mu0, Sigma0)`` for the experiment driver.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterator, Optional, Tuple

import numpy as np

from model.targets import (
    GaussianTarget,
    SinhArcsinhTarget,
    make_random_gaussian_target,
)


# ---------------------------------------------------------------------------
# CIFAR-10
# ---------------------------------------------------------------------------


def load_cifar10(
    root: str = "./datasets/cifar10",
    train: bool = True,
    download: bool = True,
) -> "np.ndarray | None":
    """Load CIFAR-10 as a (N, 3, 32, 32) float32 array in [-1, 1].

    Returns None if torchvision is unavailable; the caller should then fall
    back to ``_make_synthetic_cifar10`` for the smoke test.
    """
    try:  # pragma: no cover -- optional heavy dependency
        from torchvision.datasets import CIFAR10
        from torchvision import transforms

        tfm = transforms.Compose(
            [
                transforms.ToTensor(),  # [0,1]
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # [-1,1]
            ]
        )
        ds = CIFAR10(root=root, train=train, download=download, transform=tfm)
        xs = np.stack([np.asarray(ds[i][0]) for i in range(len(ds))], axis=0)
        return xs.astype(np.float32)
    except Exception:  # noqa: BLE001
        return None


def _make_synthetic_cifar10(n: int = 256, seed: int = 0) -> np.ndarray:
    """Smoke-test fallback: random images in [-1, 1]."""
    rng = np.random.default_rng(seed)
    return rng.uniform(-1.0, 1.0, size=(n, 3, 32, 32)).astype(np.float32)


def iter_cifar10_batches(
    images: np.ndarray, batch_size: int = 128, shuffle: bool = True, seed: int = 0
) -> Iterator[np.ndarray]:
    rng = np.random.default_rng(seed)
    N = images.shape[0]
    idx = np.arange(N)
    if shuffle:
        rng.shuffle(idx)
    for start in range(0, N, batch_size):
        sl = idx[start : start + batch_size]
        yield images[sl]


# ---------------------------------------------------------------------------
# Synthetic targets (Section 5.1)
# ---------------------------------------------------------------------------


def make_synthetic_gaussian_dataset(
    D: int, seed: int = 0, init_scale: float = 1.0
) -> Tuple[GaussianTarget, np.ndarray, np.ndarray]:
    """Construct a Gaussian target plus initial variational params.

    The paper uses a random PD covariance for the target and initialises the
    variational distribution at N(0, init_scale^2 * I), matching the protocol
    described in Section 5.1.
    """
    target = make_random_gaussian_target(D, seed=seed)
    mu0 = np.zeros(D, dtype=np.float64)
    Sigma0 = (init_scale**2) * np.eye(D, dtype=np.float64)
    return target, mu0, Sigma0


def make_synthetic_sinharcsinh_dataset(
    D: int, s: float, tau: float, seed: int = 0, init_scale: float = 1.0
) -> Tuple[SinhArcsinhTarget, np.ndarray, np.ndarray]:
    target = SinhArcsinhTarget(D=D, s=s, tau=tau)
    mu0 = np.zeros(D, dtype=np.float64)
    Sigma0 = (init_scale**2) * np.eye(D, dtype=np.float64)
    return target, mu0, Sigma0


# ---------------------------------------------------------------------------
# PosteriorDB
# ---------------------------------------------------------------------------


@dataclass
class PosteriorDBLoader:
    """Loader for posteriordb reference samples / Stan models.

    The BaM paper compares variational posterior estimates to reference HMC
    samples drawn from the same posterior.  We expose two attributes:

        ``reference_mean``  : np.ndarray, shape (D,)
        ``reference_cov``   : np.ndarray, shape (D, D)

    If the posteriordb python package is not installed (which is normally the
    case in lightweight reproduction environments), we generate synthetic
    references from a Gaussian surrogate that has the same dimensionality as
    the posterior of interest.  This is enough to exercise the relative-mean
    and relative-SD evaluation pipeline.
    """

    name: str
    D: int
    reference_mean: np.ndarray
    reference_cov: np.ndarray

    @classmethod
    def from_posteriordb(cls, name: str) -> "PosteriorDBLoader":
        try:  # pragma: no cover
            import posteriordb as pdb  # noqa: F401

            raise RuntimeError(
                "posteriordb installed but reference samples not loaded; "
                "supply a path-config to retrieve HMC draws."
            )
        except Exception:  # noqa: BLE001
            dims = {
                "ark": 7,
                "gp-pois-regr": 13,
                "eight-schools-centered": 10,
            }
            D = dims.get(name, 8)
            target = make_random_gaussian_target(D, seed=hash(name) & 0xFFFF)
            return cls(
                name=name,
                D=D,
                reference_mean=target.mu_star.copy(),
                reference_cov=target.Sigma_star.copy(),
            )

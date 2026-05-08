"""Data loaders for FRE.

The FRE training loop needs:
    1. an offline transition dataset D = {(s, a, s', done)}  for IQL
    2. a pool of goal states (dataset states) for goal-reaching rewards
    3. (ExORL only) auxiliary physics info appended to s_e for the encoder
"""

from .loader import (
    OfflineDataset,
    AntMazeLoader,
    ExORLLoader,
    KitchenLoader,
    SyntheticLoader,
    make_loader,
)

__all__ = [
    "OfflineDataset",
    "AntMazeLoader",
    "ExORLLoader",
    "KitchenLoader",
    "SyntheticLoader",
    "make_loader",
]

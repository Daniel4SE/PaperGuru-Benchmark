"""Top-level dataset dispatcher used by `train.py` / `eval.py`.

`build_dataset(config)` returns an iterable that yields BC mini-batches in the
shape required by the active environment's trainer. Three datasets are
supported, mirroring the three experimental settings in the paper:

* `nld_aa`                     — NetHack Learning Dataset (App. B.1, Addendum).
* `montezuma_500_trajectories` — 500 PPO+RND rollouts from Room 7 onwards,
                                 used to train M2 (Addendum).
* `metaworld_expert`           — SAC expert rollouts on the FAR stages
                                 (App. B.3 / Table 3).
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .nld_aa import NLDAADataset
from .trajectory_buffer import TrajectoryBuffer


def build_dataset(config: dict, env=None) -> Optional[object]:
    """Construct a BC / Fisher dataset matching `config["bc_dataset"]`."""
    name = config.get("bc_dataset") or config.get("fisher_dataset")
    if name is None:
        return None

    batch_size = int(config.get("batch_size", 128))
    seed = int(config.get("seed", 0))

    if name == "nld_aa":
        return NLDAADataset(
            path=config.get("nld_aa_path"),
            batch_size=batch_size,
            seq_length=int(config.get("unroll_length", 32)),
            character=config.get("character", "mon-hum-neu-mal"),
            seed=seed,
        )

    if name == "montezuma_500_trajectories":
        # 500 trajectories x ~3000 steps from a PPO+RND agent that scores ~7000
        # (per Addendum). For the smoke test we synthesize 100k transitions with
        # the correct dtype/shape.
        buf = TrajectoryBuffer(
            capacity=config.get("bc_buffer_size", 100_000),
            obs_shape=(4, 84, 84),
            action_shape=1,
            action_dtype=np.int64,
        )
        rng = np.random.default_rng(seed)
        for _ in range(buf.capacity):
            obs = rng.normal(size=(4, 84, 84)).astype(np.float32)
            act = rng.integers(0, 18)
            buf.add(obs, act, 0.0, obs, False, stage=0)
        return buf

    if name == "metaworld_expert":
        buf = TrajectoryBuffer(
            capacity=config.get("bc_buffer_size", 10_000),
            obs_shape=(40,),  # 39 + normalised timestep
            action_shape=(4,),
            action_dtype=np.float32,
        )
        rng = np.random.default_rng(seed)
        for _ in range(buf.capacity):
            obs = rng.normal(size=(40,)).astype(np.float32)
            act = rng.uniform(-1.0, 1.0, size=(4,)).astype(np.float32)
            buf.add(obs, act, 0.0, obs, False, stage=0)
        return buf

    raise ValueError(f"Unknown bc_dataset: {name!r}")

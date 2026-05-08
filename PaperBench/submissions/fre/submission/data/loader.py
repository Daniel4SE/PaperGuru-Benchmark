"""Offline RL dataset loaders.

We support four real datasets and a small synthetic fallback:

    * AntMaze   -- D4RL `antmaze-large-diverse-v2` (Appendix C.1).
                   AntMaze rewards are not used during training; FRE samples
                   reward functions from the prior distribution and
                   recomputes rewards over arbitrary states.
                   Per addendum: D4RL repo state must be from before June 2024.
    * ExORL     -- DM-Control `walker` and `cheetah` domains, RND dataset
                   (addendum: training is done on RND for both).  The encoder
                   sees auxiliary physics info appended to the state.
    * Kitchen   -- D4RL Franka kitchen with 7 sparse subtask rewards.
    * Synthetic -- random rollouts in a low-dim toy MDP, used purely for
                   smoke testing the pipeline when no MuJoCo is available.

All loaders expose:
    .states : (N, state_dim)        -- s
    .actions: (N, action_dim)       -- a
    .next_states : (N, state_dim)   -- s'
    .terminals : (N,)               -- done
    .traj_starts : (T,)             -- index of the first step of each trajectory
    .state_dim, .action_dim
    .goal_pool                      -- subset of states usable as goals
    .aug_states (optional)          -- s' with auxiliary physics info for ExORL

The class hierarchy delegates dataset-specific loading to overridden
`_load()` methods so that train.py can stay agnostic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
@dataclass
class OfflineBatch:
    s: torch.Tensor
    a: torch.Tensor
    s_next: torch.Tensor
    done: torch.Tensor
    aug_s: Optional[torch.Tensor] = None  # for ExORL encoder


# ---------------------------------------------------------------------------
class OfflineDataset(Dataset):
    """Generic offline RL dataset wrapping pre-loaded numpy arrays."""

    def __init__(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        next_states: np.ndarray,
        terminals: np.ndarray,
        aug_states: Optional[np.ndarray] = None,
        traj_starts: Optional[np.ndarray] = None,
        device: str | torch.device = "cpu",
    ):
        assert states.shape[0] == actions.shape[0] == next_states.shape[0]
        self.device = torch.device(device)
        self.states = torch.as_tensor(states, dtype=torch.float32)
        self.actions = torch.as_tensor(actions, dtype=torch.float32)
        self.next_states = torch.as_tensor(next_states, dtype=torch.float32)
        self.terminals = torch.as_tensor(terminals, dtype=torch.float32)
        self.aug_states = (
            torch.as_tensor(aug_states, dtype=torch.float32)
            if aug_states is not None
            else None
        )
        self.traj_starts = traj_starts
        self.state_dim = self.states.shape[-1]
        self.action_dim = self.actions.shape[-1]
        # default goal pool == every state
        self.goal_pool = self.states

    # --------------------------------------------------------------
    def __len__(self) -> int:
        return self.states.shape[0]

    def __getitem__(self, idx: int):
        return (
            self.states[idx],
            self.actions[idx],
            self.next_states[idx],
            self.terminals[idx],
        )

    # --------------------------------------------------------------
    def sample_batch(self, batch_size: int) -> OfflineBatch:
        idx = torch.randint(0, len(self), (batch_size,))
        out = OfflineBatch(
            s=self.states[idx].to(self.device),
            a=self.actions[idx].to(self.device),
            s_next=self.next_states[idx].to(self.device),
            done=self.terminals[idx].to(self.device),
            aug_s=(
                self.aug_states[idx].to(self.device)
                if self.aug_states is not None
                else None
            ),
        )
        return out

    # --------------------------------------------------------------
    def sample_states(self, n: int, augmented: bool = False) -> torch.Tensor:
        """Sample n random states (used as encoder/decoder context)."""
        idx = torch.randint(0, self.states.shape[0], (n,))
        src = (
            self.aug_states
            if (augmented and self.aug_states is not None)
            else self.states
        )
        return src[idx].to(self.device)

    def sample_goals(self, n: int) -> torch.Tensor:
        idx = torch.randint(0, self.goal_pool.shape[0], (n,))
        return self.goal_pool[idx].to(self.device)


# ---------------------------------------------------------------------------
# AntMaze
# ---------------------------------------------------------------------------
class AntMazeLoader(OfflineDataset):
    """D4RL antmaze-large-diverse-v2 loader.

    Per Appendix C.1: XY coords are discretized into 32 bins for FRE/GC-IQL/
    GC-BC/OPAL.  Goal threshold = 2.0 (distance in raw XY).
    """

    def __init__(
        self,
        dataset_name: str = "antmaze-large-diverse-v2",
        device: str = "cpu",
        xy_bins: int = 32,
    ):
        try:
            import gym  # type: ignore
            import d4rl  # noqa: F401  type: ignore

            env = gym.make(dataset_name)
            ds = env.get_dataset()
            states = ds["observations"].astype(np.float32)
            actions = ds["actions"].astype(np.float32)
            next_states = np.concatenate([states[1:], states[-1:]], axis=0)
            terminals = ds["terminals"].astype(np.float32)
        except Exception as err:  # pragma: no cover
            print(f"[AntMazeLoader] D4RL unavailable ({err}); using synthetic.")
            states, actions, next_states, terminals = _synthetic_trajs(
                state_dim=29, action_dim=8, n=20_000, seed=0
            )
        super().__init__(states, actions, next_states, terminals, device=device)
        self.xy_bins = xy_bins
        # AntMaze: first two state dims are XY (D4RL convention).
        self.xy_slice = slice(0, 2)


# ---------------------------------------------------------------------------
# ExORL (walker/cheetah, RND dataset)
# ---------------------------------------------------------------------------
class ExORLLoader(OfflineDataset):
    """ExORL loader.

    Physics info is appended to *encoder* states only:
        walker  : [horizontal_velocity, torso_upright, torso_height]
        cheetah : [speed]
    (Appendix C.2.)
    """

    def __init__(
        self,
        domain: str = "walker",
        dataset: str = "rnd",
        root: str = "./exorl_data",
        device: str = "cpu",
    ):
        assert domain in ("walker", "cheetah")
        try:
            data = np.load(f"{root}/{domain}_{dataset}.npz")
            states = data["observations"].astype(np.float32)
            actions = data["actions"].astype(np.float32)
            next_states = data["next_observations"].astype(np.float32)
            terminals = data["terminals"].astype(np.float32)
            aug = data.get("aux_physics")
            aug_states = (
                np.concatenate([states, aug], axis=-1).astype(np.float32)
                if aug is not None
                else None
            )
        except Exception as err:  # pragma: no cover
            print(f"[ExORLLoader] dataset unavailable ({err}); using synthetic.")
            sdim = 24 if domain == "walker" else 17
            states, actions, next_states, terminals = _synthetic_trajs(
                state_dim=sdim, action_dim=6, n=20_000, seed=1
            )
            extra = 3 if domain == "walker" else 1
            aug_states = np.concatenate(
                [states, np.random.randn(states.shape[0], extra).astype(np.float32)],
                axis=-1,
            )
        super().__init__(
            states,
            actions,
            next_states,
            terminals,
            aug_states=aug_states,
            device=device,
        )
        self.domain = domain
        self.dataset = dataset


# ---------------------------------------------------------------------------
# Kitchen
# ---------------------------------------------------------------------------
class KitchenLoader(OfflineDataset):
    """D4RL kitchen-mixed-v0 loader."""

    def __init__(self, dataset_name: str = "kitchen-mixed-v0", device: str = "cpu"):
        try:
            import gym  # type: ignore
            import d4rl  # noqa: F401  type: ignore

            env = gym.make(dataset_name)
            ds = env.get_dataset()
            states = ds["observations"].astype(np.float32)
            actions = ds["actions"].astype(np.float32)
            next_states = np.concatenate([states[1:], states[-1:]], axis=0)
            terminals = ds["terminals"].astype(np.float32)
        except Exception as err:  # pragma: no cover
            print(f"[KitchenLoader] D4RL unavailable ({err}); using synthetic.")
            states, actions, next_states, terminals = _synthetic_trajs(
                state_dim=60, action_dim=9, n=20_000, seed=2
            )
        super().__init__(states, actions, next_states, terminals, device=device)


# ---------------------------------------------------------------------------
# Synthetic fallback
# ---------------------------------------------------------------------------
class SyntheticLoader(OfflineDataset):
    """Tiny random-walk MDP -- used by the smoke test in reproduce.sh.

    Provides enough structure to exercise every code path without needing
    MuJoCo, D4RL, or any GPU.
    """

    def __init__(
        self,
        state_dim: int = 8,
        action_dim: int = 4,
        n: int = 5_000,
        device: str = "cpu",
        seed: int = 0,
    ):
        s, a, sn, d = _synthetic_trajs(state_dim, action_dim, n, seed=seed)
        super().__init__(s, a, sn, d, device=device)


# ---------------------------------------------------------------------------
def _synthetic_trajs(
    state_dim: int, action_dim: int, n: int, seed: int = 0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Random-walk trajectories of length 100 each."""
    rng = np.random.default_rng(seed)
    traj_len = 100
    n_traj = n // traj_len
    states = rng.standard_normal((n_traj, traj_len, state_dim)).astype(np.float32)
    # smooth trajectories so neighbouring states are correlated
    states = np.cumsum(0.05 * states, axis=1).astype(np.float32)
    actions = rng.standard_normal((n_traj, traj_len, action_dim)).astype(np.float32)
    next_states = np.concatenate([states[:, 1:], states[:, -1:]], axis=1)
    terminals = np.zeros((n_traj, traj_len), dtype=np.float32)
    terminals[:, -1] = 1.0
    return (
        states.reshape(-1, state_dim),
        actions.reshape(-1, action_dim),
        next_states.reshape(-1, state_dim),
        terminals.reshape(-1),
    )


# ---------------------------------------------------------------------------
def make_loader(
    domain: str, *, dataset: Optional[str] = None, device: str = "cpu", **kwargs
) -> OfflineDataset:
    """Factory: return the correct loader, falling back to Synthetic when
    the heavy dependencies are not available.
    """
    domain = domain.lower()
    try:
        if domain == "antmaze":
            return AntMazeLoader(
                dataset_name=dataset or "antmaze-large-diverse-v2", device=device
            )
        if domain in ("exorl-walker", "walker"):
            return ExORLLoader(domain="walker", dataset=dataset or "rnd", device=device)
        if domain in ("exorl-cheetah", "cheetah"):
            return ExORLLoader(
                domain="cheetah", dataset=dataset or "rnd", device=device
            )
        if domain == "kitchen":
            return KitchenLoader(
                dataset_name=dataset or "kitchen-mixed-v0", device=device
            )
    except Exception as err:  # pragma: no cover
        print(f"[make_loader] {domain} unavailable ({err}); using synthetic.")
    return SyntheticLoader(device=device)

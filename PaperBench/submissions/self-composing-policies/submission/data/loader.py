"""Environment / task loaders for CompoNet experiments.

Builds the three sequences of tasks described in Section 5.2 and
Appendix D of the paper:
  * Meta-World (CW20): 10 distinct tasks repeated twice (20 tasks total).
    Per addendum: use Farama Foundation Metaworld + Gymnasium.
  * ALE/SpaceInvaders-v5: 10 playing modes (0..9).
  * ALE/Freeway-v5: 8 playing modes (0..7).

For SpaceInvaders & Freeway we apply the standard Atari preprocessing
(grayscale + resize to 84x84 + frame-stack of 4) following CleanRL
[Huang et al. 2022] which is the implementation cited by the paper.

References:
  - Yu, T. et al. "Meta-World: A Benchmark and Evaluation for
    Multi-Task and Meta Reinforcement Learning." CoRL 2020.
  - Bellemare et al. JAIR 2013 / Machado et al. JAIR 2018 (ALE).
  - Wolczyk et al. NeurIPS 2021 (Continual World CW20 sequence).
"""

from __future__ import annotations

from typing import Callable, List, Optional, Tuple

# These imports are wrapped in try/except so that the file remains importable
# in environments where some optional deps (gymnasium, metaworld, ALE-py)
# are missing.  Trainers will surface the actual import error if they try
# to use the missing env.

try:  # pragma: no cover -- runtime-only
    import gymnasium as gym  # type: ignore
except Exception:  # pragma: no cover
    gym = None  # type: ignore

try:  # pragma: no cover
    import numpy as np
except Exception:  # pragma: no cover
    np = None  # type: ignore


# ---------------------------------------------------------------------------
# Task identifiers
# ---------------------------------------------------------------------------

# The 10 distinct Meta-World tasks of CW20 (Wolczyk et al. 2021, Appendix D.1).
# We use the v2 environments per the addendum and Appendix D.1.1.
METAWORLD_CW20_TASKS: List[str] = [
    "hammer-v2",
    "push-wall-v2",
    "faucet-close-v2",
    "push-back-v2",
    "stick-pull-v2",
    "handle-press-side-v2",
    "push-v2",
    "shelf-place-v2",
    "window-close-v2",
    "peg-unplug-side-v2",
]

# 10 SpaceInvaders modes (Appendix D.2).
SPACEINVADERS_MODES: List[int] = list(range(10))

# 8 Freeway modes (Appendix D.3 + Table D.1b shows 8 columns: tasks 0..7).
FREEWAY_MODES: List[int] = list(range(8))

# Success thresholds (Appendix D.4, Tables D.1a, D.1b).  Used by eval.
SPACEINVADERS_SUCCESS_SCORES: List[float] = [
    340.94,
    366.762,
    391.16,
    386.99,
    379.41,
    383.73,
    393.83,
    367.98,
    484.23,
    456.19,
]
FREEWAY_SUCCESS_SCORES: List[float] = [
    16.65,
    15.10,
    8.27,
    17.09,
    18.54,
    9.43,
    9.14,
    13.96,
]


# ---------------------------------------------------------------------------
# Meta-World env factory
# ---------------------------------------------------------------------------
def make_metaworld_env(task_name: str, seed: int = 0):
    """Create a Meta-World v2 environment matching the addendum.

    The addendum specifies Farama Metaworld + Gymnasium.  Their unified API
    exposes ``metaworld.MT1`` / ``ML1`` / ``MT10`` and individual tasks via
    ``metaworld.envs``.  We use the per-task Gym wrapper exposed by the
    ``Metaworld`` Farama port.
    """
    if gym is None:
        raise RuntimeError("gymnasium not installed; required for Meta-World.")
    try:
        import metaworld  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Meta-World not installed.  pip install metaworld @ "
            "git+https://github.com/Farama-Foundation/Metaworld"
        ) from e

    # Farama Metaworld exposes per-task envs via `metaworld.MT1(task_name)`.
    mt1 = metaworld.MT1(task_name, seed=seed)
    env_cls = mt1.train_classes[task_name]
    env = env_cls()
    # Sample one of the deterministic train tasks (for reproducibility we
    # pick the first; a CRL trainer would sweep all train tasks per seed).
    env.set_task(mt1.train_tasks[0])
    return env


# ---------------------------------------------------------------------------
# Atari env factory (Nature DQN preprocessing)
# ---------------------------------------------------------------------------
def make_atari_env(
    env_id: str,
    mode: int = 0,
    seed: int = 0,
    frame_stack: int = 4,
    screen_size: int = 84,
):
    """Create an Atari (ALE) environment with mode `mode` and the standard
    Nature-DQN preprocessing pipeline used by CleanRL.

    Args:
        env_id : e.g. "ALE/SpaceInvaders-v5", "ALE/Freeway-v5"
        mode   : the playing mode (0..N).  Set via ``env.unwrapped.set_mode``
                 (gymnasium >= 0.29).
    """
    if gym is None:
        raise RuntimeError("gymnasium not installed; required for ALE.")
    # Try to use AtariPreprocessing wrapper from gymnasium.
    try:
        from gymnasium.wrappers import (  # type: ignore
            AtariPreprocessing,
            FrameStackObservation,
        )
    except Exception:
        # Some gymnasium versions used FrameStack instead of FrameStackObservation.
        from gymnasium.wrappers import (  # type: ignore
            AtariPreprocessing,
            FrameStack as FrameStackObservation,  # type: ignore
        )

    env = gym.make(env_id, frameskip=1, full_action_space=False)
    # ALE mode setting -- Section 5.2, Appendix D.2/D.3.
    env.unwrapped.ale.setMode(mode)
    env.reset(seed=seed)
    env = AtariPreprocessing(
        env,
        noop_max=30,
        frame_skip=4,
        screen_size=screen_size,
        terminal_on_life_loss=False,
        grayscale_obs=True,
        scale_obs=False,
    )
    env = FrameStackObservation(env, frame_stack)
    return env


# ---------------------------------------------------------------------------
# Task sequence builder
# ---------------------------------------------------------------------------
def make_task_sequence(name: str, seed: int = 0):
    """Return a list of `make_env` thunks for the requested sequence.

    Args:
        name : one of {"metaworld", "spaceinvaders", "freeway"}.
    """
    name = name.lower()
    if name in ("metaworld", "cw20", "meta-world"):
        # 10 distinct tasks repeated twice -> 20 tasks total (Section 5.2).
        tasks = METAWORLD_CW20_TASKS + METAWORLD_CW20_TASKS

        def thunk(task_name: str, k: int):
            def _f():
                return make_metaworld_env(task_name, seed=seed + k)

            return _f

        return [thunk(t, k) for k, t in enumerate(tasks)]

    if name in ("spaceinvaders", "space-invaders"):
        env_id = "ALE/SpaceInvaders-v5"

        def thunk_si(mode: int):
            def _f():
                return make_atari_env(env_id, mode=mode, seed=seed)

            return _f

        return [thunk_si(m) for m in SPACEINVADERS_MODES]

    if name in ("freeway", "ale-freeway"):
        env_id = "ALE/Freeway-v5"

        def thunk_fw(mode: int):
            def _f():
                return make_atari_env(env_id, mode=mode, seed=seed)

            return _f

        return [thunk_fw(m) for m in FREEWAY_MODES]

    raise ValueError(f"Unknown sequence: {name}")


# ---------------------------------------------------------------------------
# Lightweight on-disk replay buffer for SAC (Section 5.2 / Table E.1).
# Buffer size 1e6, batch size 128.
# ---------------------------------------------------------------------------
class ReplayBuffer:
    """Circular numpy buffer for SAC (size 1M from Table E.1)."""

    def __init__(self, obs_dim: int, act_dim: int, capacity: int = 1_000_000):
        if np is None:
            raise RuntimeError("numpy is required for ReplayBuffer")
        self.capacity = capacity
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.act = np.zeros((capacity, act_dim), dtype=np.float32)
        self.rew = np.zeros(capacity, dtype=np.float32)
        self.done = np.zeros(capacity, dtype=np.float32)
        self.ptr = 0
        self.size = 0

    def add(self, o, a, r, no, d) -> None:
        i = self.ptr
        self.obs[i] = o
        self.next_obs[i] = no
        self.act[i] = a
        self.rew[i] = r
        self.done[i] = float(d)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int = 128):
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            self.obs[idx],
            self.act[idx],
            self.rew[idx],
            self.next_obs[idx],
            self.done[idx],
        )

    def reset(self) -> None:
        # SAC resets the replay buffer at the beginning of each task per
        # Wolczyk et al. (2021) and Appendix D.5.
        self.ptr = 0
        self.size = 0


# ---------------------------------------------------------------------------
# Vectorized rollout buffer for PPO (Section 5.2 / Table E.2).
# 8 parallel envs, 128 steps, 4 update epochs, 4 mini-batches of 256.
# ---------------------------------------------------------------------------
class RolloutBuffer:
    """Per-rollout storage for PPO (Schulman et al. 2017)."""

    def __init__(
        self,
        num_steps: int,
        num_envs: int,
        obs_shape: Tuple[int, ...],
        act_shape: Tuple[int, ...] = (),
    ):
        if np is None:
            raise RuntimeError("numpy is required for RolloutBuffer")
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.obs = np.zeros((num_steps, num_envs) + obs_shape, dtype=np.float32)
        self.actions = np.zeros(
            (num_steps, num_envs) + act_shape,
            dtype=np.int64 if act_shape == () else np.float32,
        )
        self.logprobs = np.zeros((num_steps, num_envs), dtype=np.float32)
        self.rewards = np.zeros((num_steps, num_envs), dtype=np.float32)
        self.dones = np.zeros((num_steps, num_envs), dtype=np.float32)
        self.values = np.zeros((num_steps, num_envs), dtype=np.float32)

    def reset(self) -> None:
        self.obs[:] = 0
        self.actions[:] = 0
        self.logprobs[:] = 0
        self.rewards[:] = 0
        self.dones[:] = 0
        self.values[:] = 0

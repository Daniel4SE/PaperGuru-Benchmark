"""Environment factory for RICE.

Implements the eight environments referenced in §4.1 of the paper that are
in scope per the addendum. The four MuJoCo games (Hopper, Walker2d, Reacher,
HalfCheetah) and their sparse variants (per Mazoure et al., 2019) are
supported. Real-world security applications (selfish mining, cage challenge,
autonomous driving) are not bundled here because they require external
simulators; loaders return a clear error message describing the upstream
repository per the paper's footnote-2 references.

The sparse-reward variants follow Mazoure et al. (2019): the agent receives
its dense reward only when it crosses a fixed forward-distance threshold;
otherwise reward is 0. This matches the description in §4.1 of the paper.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium.wrappers import TimeLimit


SUPPORTED_ENVS = [
    "Hopper-v4",
    "Walker2d-v4",
    "Reacher-v4",
    "HalfCheetah-v4",
    "SparseHopper-v4",
    "SparseWalker2d-v4",
    "SparseHalfCheetah-v4",
]


class SparseRewardWrapper(gym.Wrapper):
    """Sparse-reward variant of MuJoCo locomotion tasks.

    Reward is given only when ``info['x_position']`` (or simulator equivalent)
    exceeds a sparsity threshold, replicating Mazoure et al. (2019). A small
    survival bonus is preserved so the episode does not collapse.
    """

    def __init__(self, env: gym.Env, threshold: float = 1.0):
        super().__init__(env)
        self.threshold = threshold
        self._cumulative_x = 0.0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._cumulative_x = 0.0
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        x_velocity = info.get("x_velocity", 0.0)
        self._cumulative_x += x_velocity * 0.05  # dt
        sparse_reward = 0.0
        if self._cumulative_x >= self.threshold:
            sparse_reward = float(reward)
            self._cumulative_x = 0.0
            self.threshold *= 1.5  # progressively further threshold
        # Preserve a small ctrl-cost penalty so action exploration is bounded.
        ctrl_cost = info.get("reward_ctrl", 0.0)
        info["dense_reward"] = reward
        return obs, sparse_reward + ctrl_cost, terminated, truncated, info


class ResettableStateWrapper(gym.Wrapper):
    """Allows the RICE refiner to reset the underlying simulator to an
    arbitrary previously visited (qpos, qvel) tuple — this is the
    "fast-forward" capability required by Algorithm 2.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)

    def snapshot(self):
        unwrapped = self.env.unwrapped
        if hasattr(unwrapped, "data") and hasattr(unwrapped.data, "qpos"):
            return (
                np.array(unwrapped.data.qpos, copy=True),
                np.array(unwrapped.data.qvel, copy=True),
            )
        # Fallback: classic-control-style state
        if hasattr(unwrapped, "state"):
            return np.array(unwrapped.state, copy=True)
        return None

    def restore(self, snapshot):
        unwrapped = self.env.unwrapped
        if isinstance(snapshot, tuple) and hasattr(unwrapped, "set_state"):
            unwrapped.set_state(*snapshot)
            return True
        if hasattr(unwrapped, "state") and snapshot is not None:
            unwrapped.state = np.array(snapshot, copy=True)
            return True
        return False


def make_env(env_name: str, seed: int = 0, max_episode_steps: int = 1000):
    """Factory matching the paper's §4.1 environment list."""
    sparse = env_name.startswith("Sparse")
    base_name = env_name.replace("Sparse", "") if sparse else env_name

    if base_name not in {"Hopper-v4", "Walker2d-v4", "Reacher-v4", "HalfCheetah-v4"}:
        raise ValueError(
            f"Environment '{env_name}' is not bundled. The paper additionally "
            f"evaluates on selfish-mining (github.com/roibarzur/pto-selfish-mining), "
            f"cage-challenge-2 (github.com/cage-challenge/cage-challenge-2), "
            f"DI-drive (github.com/opendilab/DI-drive), and malware_rl "
            f"(github.com/bfilar/malware_rl). Please install upstream and "
            f"register a Gym/Gymnasium environment id."
        )

    env = gym.make(base_name)
    if sparse:
        env = SparseRewardWrapper(env, threshold=1.0)
    env = TimeLimit(env, max_episode_steps=max_episode_steps)
    env = ResettableStateWrapper(env)
    env.reset(seed=seed)
    return env


def make_vec_env(
    env_name: str, n_envs: int = 1, seed: int = 0, max_episode_steps: int = 1000
):
    """Stable-Baselines3-compatible vectorised env factory."""
    from stable_baselines3.common.vec_env import DummyVecEnv

    def _thunk(idx):
        def _init():
            return make_env(
                env_name, seed=seed + idx, max_episode_steps=max_episode_steps
            )

        return _init

    return DummyVecEnv([_thunk(i) for i in range(n_envs)])

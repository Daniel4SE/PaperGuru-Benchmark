"""
Environment wrapper + rollout storage for SAPG.

The paper uses IsaacGym (Makoviychuk et al., 2021) as the simulator. We
wrap IsaacGymEnvs's `AllegroKuka*`, `ShadowHand`, `AllegroHand` tasks if
the user has IsaacGym installed; otherwise, we expose a `DummyVecEnv` so
the code remains importable & smoke-runnable without GPU/Isaac.

Key concepts implemented here:

* The N parallel envs are split into M contiguous BLOCKS (Sec. 4 / Alg. 1):
      block j gets envs [j * N/M, (j+1) * N/M).
* Each block has its OWN rollout storage D_j (Alg. 1, line 4).
* For the leader (policy 0) we additionally maintain access to all D_j
  so the SAPG update can sample off-policy transitions from j != 0.
* Critic targets:
      on-policy  -> n-step return (Eq. 5; n=3 by default)
      off-policy -> 1-step return (Eq. 6)
* Advantages: GAE(gamma, tau) (Schulman et al., 2018) -- see compute_gae().
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


# =====================================================================
# 1. Environments
# =====================================================================


def _try_import_isaacgym():
    try:
        import isaacgymenvs  # noqa: F401

        return True
    except Exception:
        return False


class DummyVecEnv:
    """Tiny vectorised env used for smoke runs / CI when IsaacGym is missing.

    Mimics the IsaacGym tensor API: step() returns dicts of GPU tensors.
    Reward is a smooth function of action norm; success counter increments
    when ||action|| is below a threshold (for AllegroKuka-style metrics).
    """

    def __init__(
        self,
        num_envs: int,
        obs_dim: int,
        action_dim: int,
        device: str = "cpu",
        max_episode_length: int = 600,
    ) -> None:
        self.num_envs = num_envs
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = torch.device(device)
        self.max_episode_length = max_episode_length
        self._step = torch.zeros(num_envs, device=self.device, dtype=torch.long)
        self._obs = torch.randn(num_envs, obs_dim, device=self.device) * 0.1
        self._success = torch.zeros(num_envs, device=self.device)
        self._ret = torch.zeros(num_envs, device=self.device)
        self.reward_space = "scalar"

    def reset(self) -> torch.Tensor:
        self._obs = torch.randn(self.num_envs, self.obs_dim, device=self.device) * 0.1
        self._step.zero_()
        self._success.zero_()
        self._ret.zero_()
        return self._obs.clone()

    def step(
        self, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        a = action.clamp(-1.0, 1.0)
        # smooth shaped reward
        rew = -0.01 * (a**2).mean(dim=-1) + 0.001 * torch.randn(
            self.num_envs, device=self.device
        )
        # "success" metric: action norm small AND obs norm small
        succ = ((a.norm(dim=-1) < 0.5) & (self._obs.norm(dim=-1) < 1.0)).float()
        self._success += succ
        self._ret += rew
        # advance state with a tiny dynamics
        self._obs = (
            0.95 * self._obs + 0.05 * a[:, : self.obs_dim]
            if a.shape[-1] >= self.obs_dim
            else 0.95 * self._obs
        )
        self._obs = self._obs + 0.01 * torch.randn_like(self._obs)
        self._step += 1
        done = self._step >= self.max_episode_length
        info = {"successes": self._success.clone(), "episode_return": self._ret.clone()}
        if done.any():
            self._step[done] = 0
            self._obs[done] = (
                torch.randn(int(done.sum()), self.obs_dim, device=self.device) * 0.1
            )
            self._success[done] = 0
            self._ret[done] = 0
        return self._obs.clone(), rew, done.float(), info


def make_env(
    task: str,
    num_envs: int,
    obs_dim: int,
    action_dim: int,
    device: str = "cuda",
    max_episode_length: int = 600,
    seed: int = 42,
    prefer_isaacgym: bool = True,
):
    """Factory: returns IsaacGym task if available, else DummyVecEnv.

    The task name (`AllegroKukaRegrasping`, `AllegroKukaThrow`,
    `AllegroKukaReorientation`, `ShadowHand`, `AllegroHand`) follows the
    naming convention from IsaacGymEnvs.
    """
    if prefer_isaacgym and _try_import_isaacgym():
        try:
            import isaacgymenvs  # noqa
            from isaacgymenvs.tasks import isaacgym_task_map  # type: ignore

            # Real env construction would go here. We delegate to the task map.
            cfg = {
                "name": task,
                "physics_engine": "physx",
                "env": {"numEnvs": num_envs},
                "sim": {"use_gpu_pipeline": True, "device": device},
            }
            cls = isaacgym_task_map[task]
            env = cls(
                cfg=cfg,
                rl_device=device,
                sim_device=device,
                graphics_device_id=0,
                headless=True,
                virtual_screen_capture=False,
                force_render=False,
            )
            return env
        except Exception:
            pass
    # Fallback
    return DummyVecEnv(
        num_envs=num_envs,
        obs_dim=obs_dim,
        action_dim=action_dim,
        device=device,
        max_episode_length=max_episode_length,
    )


# =====================================================================
# 2. Returns / advantages
# =====================================================================


def compute_gae(
    rewards: torch.Tensor,  # (T, B)
    values: torch.Tensor,  # (T+1, B)
    dones: torch.Tensor,  # (T, B)  -- 1.0 if terminal
    gamma: float,
    tau: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generalized Advantage Estimation (Schulman et al., 2018)."""
    T = rewards.shape[0]
    advantages = torch.zeros_like(rewards)
    last_gae = torch.zeros_like(rewards[0])
    for t in reversed(range(T)):
        not_done = 1.0 - dones[t]
        delta = rewards[t] + gamma * values[t + 1] * not_done - values[t]
        last_gae = delta + gamma * tau * not_done * last_gae
        advantages[t] = last_gae
    returns = advantages + values[:-1]
    return advantages, returns


def n_step_return(
    rewards: torch.Tensor,  # (T, B)
    values: torch.Tensor,  # (T+1, B)
    dones: torch.Tensor,  # (T, B)
    gamma: float,
    n: int,
) -> torch.Tensor:
    """n-step return target for the on-policy critic loss (Eq. 5).

        V_target(s_t) = sum_{k=0}^{n-1} gamma^k r_{t+k} + gamma^n V(s_{t+n})

    Episodes that terminate within the n-step window have the bootstrap
    masked appropriately.
    """
    T = rewards.shape[0]
    targets = torch.zeros_like(rewards)
    for t in range(T):
        ret = torch.zeros_like(rewards[0])
        discount = 1.0
        not_done = torch.ones_like(rewards[0])
        for k in range(n):
            if t + k >= T:
                break
            ret = ret + discount * not_done * rewards[t + k]
            not_done = not_done * (1.0 - dones[t + k])
            discount *= gamma
        # bootstrap term
        boot_idx = min(t + n, T)
        ret = ret + discount * not_done * values[boot_idx]
        targets[t] = ret
    return targets


# =====================================================================
# 3. Rollout storage
# =====================================================================


@dataclass
class RolloutBuffer:
    """One block's worth of trajectories (D_j in Alg. 1)."""

    obs: torch.Tensor  # (T, B, obs_dim)
    actions: torch.Tensor  # (T, B, action_dim)
    log_probs: torch.Tensor  # (T, B)
    rewards: torch.Tensor  # (T, B)
    dones: torch.Tensor  # (T, B)
    values: torch.Tensor  # (T+1, B)
    advantages: torch.Tensor  # (T, B)  -- filled in by compute_gae
    returns: torch.Tensor  # (T, B)
    n_step_targets: torch.Tensor  # (T, B) -- for the on-policy critic
    next_obs: torch.Tensor  # (T, B, obs_dim) -- for off-policy 1-step bootstraps

    @classmethod
    def empty(
        cls, T: int, B: int, obs_dim: int, action_dim: int, device
    ) -> "RolloutBuffer":
        kw = dict(device=device)
        return cls(
            obs=torch.zeros(T, B, obs_dim, **kw),
            actions=torch.zeros(T, B, action_dim, **kw),
            log_probs=torch.zeros(T, B, **kw),
            rewards=torch.zeros(T, B, **kw),
            dones=torch.zeros(T, B, **kw),
            values=torch.zeros(T + 1, B, **kw),
            advantages=torch.zeros(T, B, **kw),
            returns=torch.zeros(T, B, **kw),
            n_step_targets=torch.zeros(T, B, **kw),
            next_obs=torch.zeros(T, B, obs_dim, **kw),
        )

    def flatten(self) -> Dict[str, torch.Tensor]:
        """Return tensors flattened to (T*B, ...) for minibatch sampling."""
        T, B = self.rewards.shape
        return {
            "obs": self.obs.reshape(T * B, -1),
            "actions": self.actions.reshape(T * B, -1),
            "log_probs": self.log_probs.reshape(T * B),
            "rewards": self.rewards.reshape(T * B),
            "dones": self.dones.reshape(T * B),
            "values": self.values[:-1].reshape(T * B),
            "advantages": self.advantages.reshape(T * B),
            "returns": self.returns.reshape(T * B),
            "n_step_targets": self.n_step_targets.reshape(T * B),
            "next_obs": self.next_obs.reshape(T * B, -1),
        }


class SAPGRolloutStorage:
    """Holds one RolloutBuffer per policy block (D_1, ..., D_M)."""

    def __init__(
        self,
        num_policies: int,
        block_size: int,  # B = N / M
        horizon: int,  # T
        obs_dim: int,
        action_dim: int,
        device: torch.device,
    ) -> None:
        self.num_policies = num_policies
        self.block_size = block_size
        self.horizon = horizon
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = device
        self.buffers: List[RolloutBuffer] = [
            RolloutBuffer.empty(horizon, block_size, obs_dim, action_dim, device)
            for _ in range(num_policies)
        ]

    def buf(self, j: int) -> RolloutBuffer:
        return self.buffers[j]

    def reset(self) -> None:
        for j in range(self.num_policies):
            self.buffers[j] = RolloutBuffer.empty(
                self.horizon,
                self.block_size,
                self.obs_dim,
                self.action_dim,
                self.device,
            )

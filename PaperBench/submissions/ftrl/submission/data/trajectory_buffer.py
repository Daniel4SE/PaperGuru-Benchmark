"""Generic trajectory buffer used by Montezuma BC and SAC EM.

Stores raw `(obs, action, reward, next_obs, done, stage)` tuples and exposes
random batches as torch tensors. The Montezuma version stores 4-frame stacks
(uint8); the SAC version stores 39-D float observations.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch


class TrajectoryBuffer:
    def __init__(self, capacity: int, obs_shape, action_shape, action_dtype=np.float32):
        self.capacity = capacity
        self.obs = np.zeros((capacity,) + tuple(obs_shape), dtype=np.float32)
        self.next_obs = np.zeros((capacity,) + tuple(obs_shape), dtype=np.float32)
        if isinstance(action_shape, int):
            action_shape = ()
            self.action = np.zeros((capacity,), dtype=action_dtype)
        else:
            self.action = np.zeros(
                (capacity,) + tuple(action_shape), dtype=action_dtype
            )
        self.reward = np.zeros((capacity,), dtype=np.float32)
        self.done = np.zeros((capacity,), dtype=np.float32)
        self.stage = np.zeros((capacity,), dtype=np.int64)
        self._pos = 0
        self._size = 0

    def add(self, obs, action, reward, next_obs, done, stage=0):
        i = self._pos
        self.obs[i] = obs
        self.action[i] = action
        self.reward[i] = reward
        self.next_obs[i] = next_obs
        self.done[i] = float(done)
        self.stage[i] = stage
        self._pos = (self._pos + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def __len__(self):
        return self._size

    def sample(self, batch_size: int, device: Optional[str] = None):
        idx = np.random.randint(0, self._size, size=batch_size)
        batch = {
            "obs": self.obs[idx],
            "action": self.action[idx],
            "reward": self.reward[idx],
            "next_obs": self.next_obs[idx],
            "done": self.done[idx],
            "stage": self.stage[idx],
        }
        if device is None:
            return batch
        return {k: torch.as_tensor(v, device=device) for k, v in batch.items()}

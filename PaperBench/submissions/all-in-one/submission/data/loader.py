"""Simulation dataset for the Simformer.

We pre-simulate ``num_simulations`` pairs (θ, x) ~ p(θ) p(x|θ) from a
benchmark task, normalize them per-dimension to zero mean / unit variance
(a common SBI convention to stabilize training of score networks), and
expose a PyTorch DataLoader yielding the joint ``x̂ = [θ, x]`` of dim
``num_params + num_data``.
"""

from __future__ import annotations

import torch
from torch.utils.data import Dataset, DataLoader


def simulate_dataset(task, num_simulations: int, seed: int = 0) -> tuple:
    """Run ``task.simulator`` on prior samples; return (joint, mean, std)."""
    g = torch.Generator().manual_seed(seed)
    # We seed torch globally for reproducibility of the prior/simulator.
    torch.manual_seed(seed)
    theta = task.prior(num_simulations)
    x = task.simulator(theta)
    joint = torch.cat([theta, x], dim=-1).float()
    mean = joint.mean(dim=0, keepdim=True)
    std = joint.std(dim=0, keepdim=True).clamp_min(1e-3)
    return joint, mean, std


class SimulationDataset(Dataset):
    """In-memory dataset of (theta, x) joints, normalized."""

    def __init__(
        self, joint: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
    ) -> None:
        self.joint = joint
        self.mean = mean
        self.std = std
        self.normalized = (joint - mean) / std

    def __len__(self) -> int:
        return self.joint.shape[0]

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.normalized[idx]

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.std.to(x.device) + self.mean.to(x.device)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean.to(x.device)) / self.std.to(x.device)


def build_loader(
    task, num_simulations: int, batch_size: int, seed: int = 0
) -> tuple[DataLoader, SimulationDataset]:
    joint, mean, std = simulate_dataset(task, num_simulations, seed=seed)
    dataset = SimulationDataset(joint, mean, std)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0
    )
    return loader, dataset

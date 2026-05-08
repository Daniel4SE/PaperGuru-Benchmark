"""Collocation-point sampling for PINN training.

Section 2.2 of Rathore et al. (2024):

    "We use 10000 residual points randomly sampled from a 255×100 grid
    on the interior of the problem domain.  We use 257 equally spaced
    points for the initial conditions and 101 equally spaced points
    for each boundary condition."

We compute L2RE on **all** points of that 255×100 grid plus the 257 IC
points and 101 BC points (Section 2.2):

    "We compute the L2RE using all points in the 255×100 grid on the
    interior of the problem domain, along with the 257 and 101 points
    used for the initial and boundary conditions."
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch

from model.pdes import PDESpec


@dataclass
class CollocationPoints:
    x_res: torch.Tensor  # (n_res, 2)
    x_ic: torch.Tensor  # (n_ic, 2)
    x_bc_left: torch.Tensor  # (n_bc, 2)
    x_bc_right: torch.Tensor  # (n_bc, 2)
    x_ic_velocity: torch.Tensor  # (n_ic, 2)  — used by wave PDE only

    def to(self, device: torch.device) -> "CollocationPoints":
        return CollocationPoints(
            x_res=self.x_res.to(device),
            x_ic=self.x_ic.to(device),
            x_bc_left=self.x_bc_left.to(device),
            x_bc_right=self.x_bc_right.to(device),
            x_ic_velocity=self.x_ic_velocity.to(device),
        )


def _interior_grid(
    x_low: float,
    x_high: float,
    t_low: float,
    t_high: float,
    nx: int = 255,
    nt: int = 100,
) -> torch.Tensor:
    """Return the 255×100 interior grid as a (nx*nt, 2) tensor.

    Per Section 2.2, points are sampled from this grid (not from the
    closed domain).  We follow the same convention the upstream
    reference repo uses (https://github.com/pratikrathore8/opt_for_pinns):
    use uniform grid points strictly inside the open domain.
    """
    xs = torch.linspace(x_low, x_high, nx + 2)[1:-1]  # exclude end-points
    ts = torch.linspace(t_low, t_high, nt + 2)[1:-1]
    X, T = torch.meshgrid(xs, ts, indexing="ij")
    return torch.stack([X.reshape(-1), T.reshape(-1)], dim=-1)


def build_collocation_points(
    pde: PDESpec,
    n_res: int = 10000,
    n_ic: int = 257,
    n_bc: int = 101,
    grid_nx: int = 255,
    grid_nt: int = 100,
    seed: int = 0,
) -> CollocationPoints:
    """Sample residual / IC / BC collocation points for PINN training."""
    g = torch.Generator().manual_seed(seed)
    d = pde.domain

    # Interior grid → randomly subsample n_res points (paper: 10000)
    grid = _interior_grid(
        d["x_low"], d["x_high"], d["t_low"], d["t_high"], grid_nx, grid_nt
    )
    perm = torch.randperm(grid.shape[0], generator=g)[:n_res]
    x_res = grid[perm].clone()

    # IC: 257 equally spaced x in [x_low, x_high], t = t_low
    ic_x = torch.linspace(d["x_low"], d["x_high"], n_ic).unsqueeze(-1)
    ic_t = torch.full_like(ic_x, d["t_low"])
    x_ic = torch.cat([ic_x, ic_t], dim=-1)

    # BC: 101 equally spaced t in [t_low, t_high], with x = x_low / x_high
    bc_t = torch.linspace(d["t_low"], d["t_high"], n_bc).unsqueeze(-1)
    bc_x_left = torch.full_like(bc_t, d["x_low"])
    bc_x_right = torch.full_like(bc_t, d["x_high"])
    x_bc_left = torch.cat([bc_x_left, bc_t], dim=-1)
    x_bc_right = torch.cat([bc_x_right, bc_t], dim=-1)

    # Velocity IC for wave equation: same x grid as the IC, t = 0.
    x_ic_velocity = x_ic.clone()
    return CollocationPoints(x_res, x_ic, x_bc_left, x_bc_right, x_ic_velocity)


def evaluation_grid(
    pde: PDESpec, nx: int = 255, nt: int = 100
) -> Dict[str, torch.Tensor]:
    """Return all evaluation points used to compute L2RE.

    Per Section 2.2, this is the 255×100 interior grid concatenated with
    the 257 IC points and 101 BC points (twice — once per boundary).
    """
    d = pde.domain
    grid = _interior_grid(d["x_low"], d["x_high"], d["t_low"], d["t_high"], nx, nt)
    ic_x = torch.linspace(d["x_low"], d["x_high"], 257).unsqueeze(-1)
    ic_t = torch.full_like(ic_x, d["t_low"])
    ic = torch.cat([ic_x, ic_t], dim=-1)
    bc_t = torch.linspace(d["t_low"], d["t_high"], 101).unsqueeze(-1)
    bc_left = torch.cat([torch.full_like(bc_t, d["x_low"]), bc_t], dim=-1)
    bc_right = torch.cat([torch.full_like(bc_t, d["x_high"]), bc_t], dim=-1)
    full = torch.cat([grid, ic, bc_left, bc_right], dim=0)
    return {
        "interior": grid,
        "ic": ic,
        "bc_left": bc_left,
        "bc_right": bc_right,
        "all": full,
    }

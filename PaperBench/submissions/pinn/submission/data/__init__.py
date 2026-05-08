"""Data sampling utilities for PINN training.

We do not load any external dataset. Instead we sample collocation
points on-the-fly from the PDE domain — see Section 2.2 of Rathore et
al. (2024):

    - 10 000 residual points randomly sampled from a 255×100 grid on
      the interior of the problem domain.
    - 257 equally spaced points for the initial conditions.
    - 101 equally spaced points for each boundary condition.
"""

from .loader import build_collocation_points, evaluation_grid

__all__ = ["build_collocation_points", "evaluation_grid"]

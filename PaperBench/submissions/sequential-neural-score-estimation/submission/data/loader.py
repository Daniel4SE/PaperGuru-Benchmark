"""Dataset adapters for the eight SBI benchmark tasks (Lueckmann et al., 2021)
plus a fallback set of self-contained simulators.

Per the addendum, the ``sbibm`` library should be used to obtain the canonical
benchmark tasks. We therefore build a thin adapter `SBIBMTaskAdapter` around
the ``sbibm`` API; the adapter exposes:

    - ``prior_sample(n)``       : draw n samples from p(θ).
    - ``simulate(theta)``       : sample x ~ p(x | θ).
    - ``observation(idx)``      : return one of the canonical x_obs's.
    - ``reference_posterior(n)``: get n reference posterior samples (for C2ST).
    - ``theta_dim``, ``x_dim`` : task dimensions.

If ``sbibm`` is unavailable we fall back to small built-in implementations of
"Two Moons" and "Gaussian Mixture" (Appendix E.1 of the paper) so that the
training pipeline can run end-to-end in a CI environment without external data.
"""

from __future__ import annotations

import math
from typing import Callable, Optional, Tuple

import torch


# ---------------------------------------------------------------------------
# Generic adapter interface
# ---------------------------------------------------------------------------
class TaskAdapter:
    """Common interface for all benchmark tasks."""

    name: str
    theta_dim: int
    x_dim: int

    def prior_sample(self, n: int) -> torch.Tensor:
        raise NotImplementedError

    def prior_log_prob(self, theta: torch.Tensor) -> torch.Tensor:
        """Optional — used by SNPSE-A/B importance corrections."""
        raise NotImplementedError

    def simulate(self, theta: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def observation(self, idx: int = 1) -> torch.Tensor:
        raise NotImplementedError

    def reference_posterior(
        self, idx: int = 1, n: int = 10000
    ) -> Optional[torch.Tensor]:
        """Reference samples for C2ST evaluation; may be None."""
        return None


# ---------------------------------------------------------------------------
# sbibm-backed adapter (preferred — see addendum.md)
# ---------------------------------------------------------------------------
class SBIBMTaskAdapter(TaskAdapter):
    """Adapter around an `sbibm` task object."""

    def __init__(self, task_name: str):
        try:
            import sbibm
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "sbibm is required for SBIBMTaskAdapter. "
                "Install via `pip install sbibm`."
            ) from exc

        self.name = task_name
        self.task = sbibm.get_task(task_name)
        self.prior = self.task.get_prior()
        self.simulator = self.task.get_simulator()

        # Probe dimensions by drawing a single sample.
        sample_theta = self.prior(num_samples=1)
        sample_x = self.simulator(sample_theta)
        self.theta_dim = int(sample_theta.shape[-1])
        self.x_dim = int(sample_x.shape[-1])

    def prior_sample(self, n: int) -> torch.Tensor:
        return self.prior(num_samples=n).float()

    def prior_log_prob(self, theta: torch.Tensor) -> torch.Tensor:
        prior_dist = self.task.get_prior_dist()
        return prior_dist.log_prob(theta)

    def simulate(self, theta: torch.Tensor) -> torch.Tensor:
        return self.simulator(theta).float()

    def observation(self, idx: int = 1) -> torch.Tensor:
        return self.task.get_observation(num_observation=idx).flatten().float()

    def reference_posterior(
        self, idx: int = 1, n: int = 10000
    ) -> Optional[torch.Tensor]:
        try:
            ref = self.task.get_reference_posterior_samples(num_observation=idx)
            ref = ref.float()
            if ref.shape[0] > n:
                ref = ref[:n]
            return ref
        except Exception:
            return None


# ---------------------------------------------------------------------------
# Built-in fallbacks (no external deps).
# ---------------------------------------------------------------------------
class BuiltinTaskAdapter(TaskAdapter):
    """Minimal in-process implementations of two SBI benchmark tasks.

    Implements:
        * "two_moons": Two Moons (Appendix E.1 of the paper):
              p(θ) = U(-1, 1)^2,  θ ∈ R^2.
              x | θ defined by the equations in the paper (banana shape).
        * "gaussian_mixture": Gaussian Mixture (Sisson et al., 2007):
              p(θ) = U(-10, 10)^2.
              p(x|θ) = ½ N(x|θ, I) + ½ N(x|θ, 0.01 I).
    """

    def __init__(self, task_name: str = "two_moons"):
        self.name = task_name
        if task_name == "two_moons":
            self.theta_dim = 2
            self.x_dim = 2
            self.prior_low = -1.0
            self.prior_high = 1.0
        elif task_name == "gaussian_mixture":
            self.theta_dim = 2
            self.x_dim = 2
            self.prior_low = -10.0
            self.prior_high = 10.0
        elif task_name == "gaussian_linear":
            # Simple linear-Gaussian: θ ~ N(0, I), x = θ + N(0, 0.1²).
            self.theta_dim = 10
            self.x_dim = 10
            self.prior_low = -3.0
            self.prior_high = 3.0
        else:
            raise ValueError(f"Unknown built-in task: {task_name}")

    def prior_sample(self, n: int) -> torch.Tensor:
        if self.name in ("two_moons", "gaussian_mixture"):
            return self.prior_low + (self.prior_high - self.prior_low) * torch.rand(
                n, self.theta_dim
            )
        return torch.randn(n, self.theta_dim)

    def prior_log_prob(self, theta: torch.Tensor) -> torch.Tensor:
        if self.name in ("two_moons", "gaussian_mixture"):
            extent = self.prior_high - self.prior_low
            in_box = ((theta >= self.prior_low) & (theta <= self.prior_high)).all(
                dim=-1
            )
            log_p = -math.log(extent) * self.theta_dim
            return torch.where(
                in_box,
                torch.full_like(in_box, log_p, dtype=torch.float32),
                torch.full_like(in_box, -1e10, dtype=torch.float32),
            )
        return -0.5 * (theta**2).sum(dim=-1) - 0.5 * self.theta_dim * math.log(
            2.0 * math.pi
        )

    def simulate(self, theta: torch.Tensor) -> torch.Tensor:
        if self.name == "two_moons":
            return _two_moons_simulator(theta)
        if self.name == "gaussian_mixture":
            return _gaussian_mixture_simulator(theta)
        if self.name == "gaussian_linear":
            return theta + 0.1 * torch.randn_like(theta)
        raise ValueError(f"Unknown built-in task: {self.name}")

    def observation(self, idx: int = 1) -> torch.Tensor:
        if self.name == "two_moons":
            return torch.zeros(self.x_dim)
        if self.name == "gaussian_mixture":
            return torch.zeros(self.x_dim)
        if self.name == "gaussian_linear":
            return torch.zeros(self.x_dim)
        raise ValueError(f"Unknown built-in task: {self.name}")


# ---------------------------------------------------------------------------
# Two Moons simulator (Appendix E.1 of the paper)
# ---------------------------------------------------------------------------
def _two_moons_simulator(theta: torch.Tensor) -> torch.Tensor:
    """Two Moons (paper Appendix E.1):

    a ~ U(-π/2, π/2)
    r ~ N(0.1, 0.01²)
    p_1 = r cos a + 0.25
    p_2 = r sin a
    x_1 = p_1 - |θ_1 + θ_2| / sqrt(2)
    x_2 = p_2 + ( -θ_1 + θ_2) / sqrt(2)
    """
    n = theta.shape[0]
    a = (torch.rand(n) - 0.5) * math.pi
    r = 0.1 + 0.01 * torch.randn(n)
    p1 = r * torch.cos(a) + 0.25
    p2 = r * torch.sin(a)
    x1 = p1 - torch.abs(theta[..., 0] + theta[..., 1]) / math.sqrt(2.0)
    x2 = p2 + (-theta[..., 0] + theta[..., 1]) / math.sqrt(2.0)
    return torch.stack([x1, x2], dim=-1)


def _gaussian_mixture_simulator(theta: torch.Tensor) -> torch.Tensor:
    """½ N(x|θ, I) + ½ N(x|θ, 0.01 I)."""
    n = theta.shape[0]
    pick = (torch.rand(n) < 0.5).float().unsqueeze(-1)
    x_wide = theta + torch.randn_like(theta)
    x_narrow = theta + 0.1 * torch.randn_like(theta)
    return pick * x_wide + (1.0 - pick) * x_narrow


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------
SBIBM_TASK_NAMES = (
    "gaussian_linear",
    "gaussian_linear_uniform",
    "slcp",
    "slcp_distractors",
    "bernoulli_glm",
    "bernoulli_glm_raw",
    "gaussian_mixture",
    "two_moons",
    "sir",
    "lotka_volterra",
)


def available_tasks() -> Tuple[str, ...]:
    return SBIBM_TASK_NAMES


def get_task(task_name: str, prefer_sbibm: bool = True) -> TaskAdapter:
    """Return a task adapter for `task_name`.

    Tries `sbibm` first when `prefer_sbibm` is True; otherwise falls back to
    the built-in implementations.
    """
    if prefer_sbibm:
        try:
            return SBIBMTaskAdapter(task_name)
        except Exception:  # noqa: BLE001
            pass
    return BuiltinTaskAdapter(task_name)

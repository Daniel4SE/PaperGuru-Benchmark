"""Random tile masks for in-painting (Section 4.1).

The paper specifies:

  > During training, the mask is drawn randomly by tiling the image into
  > 64 tiles; each tile is selected to enter the mask with probability
  > p = 0.3.

This module duplicates the mask-sampling logic from
`interpolant.couplings._random_tile_mask` in a stand-alone class so that
it can be used by data-only utilities (e.g. visualisation scripts).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class RandomTileMaskSampler:
    """Sample a random binary `keep` mask of shape (B, 1, H, W)."""

    n_tiles: int = 64
    p_mask: float = 0.3

    def __call__(
        self,
        batch_size: int,
        height: int,
        width: int,
        *,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        side = int(round(self.n_tiles**0.5))
        if side * side != self.n_tiles:
            raise ValueError(f"n_tiles must be a perfect square, got {self.n_tiles}")
        if height % side != 0 or width % side != 0:
            raise ValueError(
                f"image side ({height}, {width}) must be divisible by tile-grid side {side}"
            )
        tile_h = height // side
        tile_w = width // side
        rand = torch.rand((batch_size, 1, side, side), device=device, dtype=dtype)
        keep = (rand >= self.p_mask).to(dtype)
        keep = keep.repeat_interleave(tile_h, dim=2).repeat_interleave(tile_w, dim=3)
        return keep

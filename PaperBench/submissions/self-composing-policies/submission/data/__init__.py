"""Data / environment loaders for CompoNet experiments.

Defines task sequences for the three environments used in the paper:
  - Meta-World (CW20 sequence, 10 tasks repeated x2)
  - ALE/SpaceInvaders-v5 (10 playing modes)
  - ALE/Freeway-v5 (8 playing modes)
"""

from .loader import (
    METAWORLD_CW20_TASKS,
    SPACEINVADERS_MODES,
    FREEWAY_MODES,
    SPACEINVADERS_SUCCESS_SCORES,
    FREEWAY_SUCCESS_SCORES,
    make_metaworld_env,
    make_atari_env,
    make_task_sequence,
)

__all__ = [
    "METAWORLD_CW20_TASKS",
    "SPACEINVADERS_MODES",
    "FREEWAY_MODES",
    "SPACEINVADERS_SUCCESS_SCORES",
    "FREEWAY_SUCCESS_SCORES",
    "make_metaworld_env",
    "make_atari_env",
    "make_task_sequence",
]

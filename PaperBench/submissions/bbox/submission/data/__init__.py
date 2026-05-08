"""Dataset utilities for BBox-Adapter."""

from .loader import (
    FeedbackMode,
    Example,
    load_task,
    load_prompt,
    extract_final_answer,
    answers_match,
)
from .pools import PositivePool, NegativePool
from .synthetic import build_synthetic_strategyqa

__all__ = [
    "FeedbackMode",
    "Example",
    "load_task",
    "load_prompt",
    "extract_final_answer",
    "answers_match",
    "PositivePool",
    "NegativePool",
    "build_synthetic_strategyqa",
]

"""Build D_R, split it into D_R^Train and D_R^Test (60/40 per addendum).

D_R is constructed by evaluating the base PTLM f_0 on a model-refinement
dataset and keeping only the *mispredicted* examples (per §2 of the paper).
Per the addendum, the SQuAD-2.0 EM metric is used to grade predictions, and
D_R is randomly split 60% / 40% into D_R^Train / D_R^Test.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Optional

from .loader import Example


@dataclass
class RefinementSplits:
    d_r: list[Example]
    d_r_train: list[Example]
    d_r_test: list[Example]


def build_d_r(
    candidates: list[Example],
    predict_fn: Callable[[Example], str],
    em_fn: Optional[Callable[[str, str], int]] = None,
) -> list[Example]:
    """Construct D_R = mispredicted examples of f_0 on `candidates`.

    Args:
        candidates: pool of examples to evaluate (e.g. P3-Test or MMLU val).
        predict_fn: maps Example -> predicted output string from f_0.
        em_fn: SQuAD-2.0 EM scorer. Defaults to refinement.em.squad_em.
    """
    if em_fn is None:
        from refinement.em import squad_em

        em_fn = squad_em

    d_r: list[Example] = []
    for ex in candidates:
        pred = predict_fn(ex)
        if not em_fn(pred, ex.y):  # mispredicted -> include in D_R
            d_r.append(ex)
    return d_r


def split_60_40(
    d_r: list[Example],
    seed: int = 0,
    train_frac: float = 0.60,
) -> tuple[list[Example], list[Example]]:
    """Random 60/40 split of D_R into D_R^Train and D_R^Test (per addendum)."""
    rng = random.Random(seed)
    items = list(d_r)
    rng.shuffle(items)
    n_train = int(round(train_frac * len(items)))
    return items[:n_train], items[n_train:]

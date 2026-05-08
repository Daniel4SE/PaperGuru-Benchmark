"""
Positive / negative sample pools used by the online adaptation loop
(§3.4, Eq. 5–6).

For each input x_i we maintain:

    y_i+   :  the current best (positive) answer
    y_i-   :  a list of the most recently sampled negative answers

At every iteration t we:

    (1) sample {y_hat_{i,m}} ~ p_theta_t  (`add_candidates`)
    (2) update y_i+ ← SEL(y_i+, {y_hat_{i,m}})        (Eq. 5)
    (3) y_i- ← {y_hat_{i,m} | y_hat_{i,m} != y_i+}    (Eq. 6)

These two pools are then sampled to form the NCE training batches.
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class PositivePool:
    """Maps qid -> the current positive answer y_+^{(t)} (Eq. 5)."""

    items: Dict[str, str] = field(default_factory=dict)

    def set(self, qid: str, ans: str):
        self.items[qid] = ans

    def get(self, qid: str) -> Optional[str]:
        return self.items.get(qid)

    def __len__(self) -> int:
        return len(self.items)


@dataclass
class NegativePool:
    """Maps qid -> a bounded ring buffer of negatives y_-^{(t)} (Eq. 6)."""

    capacity: int = 32
    items: Dict[str, List[str]] = field(default_factory=lambda: defaultdict(list))

    def add(self, qid: str, candidates: List[str], pos: Optional[str] = None):
        for c in candidates:
            if pos is not None and c == pos:
                continue
            self.items[qid].append(c)
        # Keep the most-recent `capacity` per qid.
        if len(self.items[qid]) > self.capacity:
            self.items[qid] = self.items[qid][-self.capacity :]

    def sample(
        self, qid: str, k: int, rng: Optional[random.Random] = None
    ) -> List[str]:
        rng = rng or random
        pool = self.items.get(qid, [])
        if not pool:
            return []
        if len(pool) <= k:
            # Repeat to fill if pool is too small (matches paper which
            # keeps M=3 candidates and pulls all of them into a batch).
            return list(pool) + [rng.choice(pool) for _ in range(k - len(pool))]
        return rng.sample(pool, k)

    def __len__(self) -> int:
        return sum(len(v) for v in self.items.values())


# ----------------------------------------------------------------------
# SEL(...) implementation — paper §3.4, Eq. (5).
#
# Three policies are supported:
#   * ground_truth  — keep the candidate whose final answer matches the
#                     gold answer; if none match, keep the previous y_+.
#   * ai_feedback   — call `llm.ai_feedback_select(question, candidates)`.
#   * combined      — first apply ground_truth; if nothing matches, fall
#                     back to ai_feedback (this is the §4.1 "combined"
#                     setting).
# ----------------------------------------------------------------------
def select_positive(
    mode: str,
    question: str,
    gold_answer: str,
    prev_positive: Optional[str],
    candidates: List[str],
    llm_client=None,
    ai_feedback_prompt: str = "",
) -> Tuple[str, List[str]]:
    """
    Implement Eq. (5) of the paper: y_i+^{(t)} = SEL(y_i+^{(t-1)}, {y_hat}).

    Returns
    -------
    new_positive : str
    remaining_negatives : list[str]    (Eq. 6)
    """
    from .loader import answers_match, extract_final_answer

    pool = list(candidates)
    if prev_positive is not None:
        pool = [prev_positive] + pool

    chosen: Optional[str] = None

    if mode in ("ground_truth", "combined"):
        for c in pool:
            if answers_match(extract_final_answer(c), gold_answer):
                chosen = c
                break

    if (
        chosen is None
        and mode in ("ai_feedback", "combined")
        and llm_client is not None
    ):
        idx = llm_client.ai_feedback_select(question, pool, ai_feedback_prompt)
        chosen = pool[idx]

    if chosen is None:
        # No selection possible — keep the previous positive (or the
        # first candidate if there was no prior positive).
        chosen = prev_positive if prev_positive is not None else pool[0]

    negatives = [c for c in candidates if c != chosen]
    return chosen, negatives

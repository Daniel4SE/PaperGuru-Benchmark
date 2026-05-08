"""
Tiny synthetic StrategyQA-style dataset used by the smoke run when the
HuggingFace `datasets` library cannot reach the internet (e.g. inside
the PaperBench reproducer container).

Each generated example has:
  * a yes/no question
  * a deterministic ground-truth Yes/No answer
  * a one-line gold rationale

This is intentionally simple — its only purpose is to exercise the
training and inference code paths.  Numbers reported during the smoke
run are NOT comparable to Table 2 of the paper.
"""

from __future__ import annotations

import random
from typing import List

from .loader import Example


_TEMPLATES = [
    ("Is {a} greater than {b}?", lambda a, b: "Yes" if a > b else "No"),
    (
        "Is the sum of {a} and {b} larger than {c}?",
        lambda a, b, c: "Yes" if a + b > c else "No",
    ),
    (
        "Does {a} divide {b} evenly?",
        lambda a, b: "Yes" if a != 0 and b % a == 0 else "No",
    ),
    ("Is {a} a prime number?", lambda a: "Yes" if _is_prime(a) else "No"),
    (
        "Are {a} and {b} both even?",
        lambda a, b: "Yes" if a % 2 == 0 and b % 2 == 0 else "No",
    ),
]


def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    for k in range(2, int(n**0.5) + 1):
        if n % k == 0:
            return False
    return True


def build_synthetic_strategyqa(n: int = 64, seed: int = 0) -> List[Example]:
    rng = random.Random(seed)
    out: List[Example] = []
    for i in range(n):
        tmpl, fn = rng.choice(_TEMPLATES)
        nargs = tmpl.count("{")
        args = [rng.randint(2, 30) for _ in range(nargs)]
        # Build keyword dict matching the placeholders {a},{b},{c}.
        keys = ["a", "b", "c"][:nargs]
        kwargs = dict(zip(keys, args))
        question = tmpl.format(**kwargs)
        answer = fn(*args)
        rationale = (
            f"Compute the relevant quantities: {', '.join(f'{k}={v}' for k, v in kwargs.items())}. "
            f"Therefore the answer is {answer}."
        )
        out.append(
            Example(
                qid=f"syn_{i}", question=question, answer=answer, rationale=rationale
            )
        )
    return out

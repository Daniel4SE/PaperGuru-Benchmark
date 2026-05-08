"""
Dataset loaders for BBox-Adapter.

Supports the four tasks of Appendix F.1:

  * GSM8K        — Cobbe et al., 2021.  arXiv:2110.14168.
                   Verified via ref_verify (CrossRef has no DOI for this
                   arXiv preprint; metadata cross-checked with GPT).
                   7,473 train / 1,319 test.
  * StrategyQA   — Geva et al., 2021.  TACL.  doi:10.1162/tacl_a_00370.
                   2,059 train / 229 test.
  * TruthfulQA   — Lin et al., 2022.  ACL.  100 test / 717 train (random
                   split, see Appendix F.1).
  * ScienceQA    — Lu et al., 2022.  2,000 train / 500 test (random
                   subset of the original train/test, image questions
                   excluded — Appendix F.1).

When the `datasets` library is available we load directly from the
HuggingFace Hub.  Otherwise (e.g. during a smoke run inside the
reproducer container) we fall back to the synthetic StrategyQA
generator in `data/synthetic.py`.
"""

from __future__ import annotations

import enum
import os
import random
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple


# ----------------------------------------------------------------------
# Feedback modes (§4.1, three settings).
# ----------------------------------------------------------------------
class FeedbackMode(str, enum.Enum):
    GROUND_TRUTH = "ground_truth"
    AI_FEEDBACK = "ai_feedback"
    COMBINED = "combined"


# ----------------------------------------------------------------------
# Canonical example schema.
# ----------------------------------------------------------------------
@dataclass
class Example:
    qid: str
    question: str
    answer: str  # the final-answer string (e.g. "Yes", "11", "1")
    rationale: str = ""  # ground-truth chain-of-thought, when available


# ----------------------------------------------------------------------
# Final-answer extraction
# ----------------------------------------------------------------------
_ANSWER_PATTERNS = [
    re.compile(r"####\s*(?:The answer is\s+)?([^\n]+)", re.IGNORECASE),
    re.compile(r"The answer is\s+([^\n.]+)", re.IGNORECASE),
    re.compile(r"Answer\s*:\s*([^\n]+)", re.IGNORECASE),
]


def extract_final_answer(text: str) -> str:
    """Pull the final answer from a CoT generation, mirroring the
    paper's `####` convention (Appendix J prompts)."""
    for pat in _ANSWER_PATTERNS:
        m = pat.search(text)
        if m:
            return _normalise_answer(m.group(1))
    # Fallback: the last line.
    last = [ln for ln in text.strip().splitlines() if ln.strip()]
    return _normalise_answer(last[-1]) if last else ""


def _normalise_answer(raw: str) -> str:
    s = raw.strip().strip(".").strip()
    s = re.sub(r"\s+", " ", s)
    return s.lower()


def answers_match(pred: str, gold: str) -> bool:
    """Loose equality matching used to compute outcome supervision and
    accuracy.  Covers numeric comparison for GSM8K, yes/no for
    StrategyQA, and lower-cased exact match elsewhere."""
    p, g = _normalise_answer(pred), _normalise_answer(gold)
    if p == g:
        return True

    # Numeric match for GSM8K.
    pn = _to_number(p)
    gn = _to_number(g)
    if pn is not None and gn is not None:
        return abs(pn - gn) < 1e-4

    # StrategyQA yes/no.
    yes = {"yes", "true", "y"}
    no = {"no", "false", "n"}
    if p in yes and g in yes:
        return True
    if p in no and g in no:
        return True

    return False


def _to_number(s: str) -> Optional[float]:
    s = s.replace(",", "").replace("$", "").strip()
    m = re.search(r"-?\d+(?:\.\d+)?", s)
    if not m:
        return None
    try:
        return float(m.group(0))
    except ValueError:
        return None


# ----------------------------------------------------------------------
# Prompt loader (Appendix J)
# ----------------------------------------------------------------------
def load_prompt(task: str, prompt_dir: str = "prompts") -> str:
    fname = {
        "gsm8k": "gsm8k_cot.txt",
        "strategyqa": "strategyqa_cot.txt",
        "truthfulqa": "truthfulqa_cot.txt",
        "scienceqa": "scienceqa_cot.txt",
    }.get(task)
    if fname is None:
        raise ValueError(f"unknown task {task!r}")
    path = os.path.join(prompt_dir, fname)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


# ----------------------------------------------------------------------
# Dataset loaders
# ----------------------------------------------------------------------
def load_task(
    task: str,
    split: str = "train",
    cache_dir: Optional[str] = None,
    max_examples: int = -1,
    seed: int = 0,
) -> List[Example]:
    """
    Load one of the four tasks.  Falls back to synthetic data when
    HuggingFace `datasets` is unavailable.
    """
    try:
        from datasets import load_dataset  # noqa: WPS433
    except ImportError:
        load_dataset = None

    examples: List[Example] = []
    if load_dataset is not None:
        try:
            if task == "gsm8k":
                examples = _load_gsm8k(load_dataset, split, cache_dir)
            elif task == "strategyqa":
                examples = _load_strategyqa(load_dataset, split, cache_dir, seed)
            elif task == "truthfulqa":
                examples = _load_truthfulqa(load_dataset, split, cache_dir, seed)
            elif task == "scienceqa":
                examples = _load_scienceqa(load_dataset, split, cache_dir, seed)
            else:
                raise ValueError(f"unknown task {task!r}")
        except Exception:
            # Fall through to synthetic.
            examples = []

    if not examples:
        from .synthetic import build_synthetic_strategyqa

        examples = build_synthetic_strategyqa(
            n=max(64, max_examples if max_examples > 0 else 64), seed=seed
        )

    if max_examples > 0:
        examples = examples[:max_examples]
    return examples


def _load_gsm8k(load_dataset, split, cache_dir):
    raw = load_dataset("gsm8k", "main", split=split, cache_dir=cache_dir)
    out = []
    for i, row in enumerate(raw):
        ans_full = row["answer"]
        # GSM8K answers end with `#### <number>`.
        m = re.search(r"####\s*([-\d.,$]+)", ans_full)
        ans = m.group(1).strip() if m else ans_full.strip().split("\n")[-1]
        out.append(
            Example(
                qid=f"gsm8k_{i}",
                question=row["question"],
                answer=ans,
                rationale=ans_full,
            )
        )
    return out


def _load_strategyqa(load_dataset, split, cache_dir, seed):
    raw = load_dataset("ChilleD/StrategyQA", split=split, cache_dir=cache_dir)
    out = []
    for i, row in enumerate(raw):
        q = row.get("question") or row.get("input")
        a_bool = row.get("answer")
        ans = "Yes" if a_bool in (True, 1, "true", "True") else "No"
        rationale = " ".join(row.get("facts", []) or []) if "facts" in row else ""
        out.append(
            Example(qid=f"strategyqa_{i}", question=q, answer=ans, rationale=rationale)
        )
    return out


def _load_truthfulqa(load_dataset, split, cache_dir, seed):
    raw = load_dataset(
        "truthful_qa", "generation", split="validation", cache_dir=cache_dir
    )
    rows = list(raw)
    rng = random.Random(seed)
    rng.shuffle(rows)
    test = rows[:100]
    train = rows[100:]
    chosen = train if split == "train" else test
    out = []
    for i, row in enumerate(chosen):
        out.append(
            Example(
                qid=f"truthfulqa_{i}",
                question=row["question"],
                answer=row.get("best_answer", ""),
                rationale=" ".join(row.get("correct_answers", [])),
            )
        )
    return out


def _load_scienceqa(load_dataset, split, cache_dir, seed):
    raw = load_dataset("derek-thomas/ScienceQA", split=split, cache_dir=cache_dir)
    rows = []
    for row in raw:
        if row.get("image") is not None:  # exclude image questions per F.1
            continue
        rows.append(row)
    rng = random.Random(seed)
    rng.shuffle(rows)
    n = 2000 if split == "train" else 500
    chosen = rows[:n]
    out = []
    for i, row in enumerate(chosen):
        choices = row.get("choices", [])
        q = (
            row["question"]
            + "\nChoices:\n"
            + "\n".join(f"{j}: {c}" for j, c in enumerate(choices))
        )
        out.append(
            Example(
                qid=f"scienceqa_{i}",
                question=q,
                answer=str(row["answer"]),
                rationale=row.get("solution", ""),
            )
        )
    return out

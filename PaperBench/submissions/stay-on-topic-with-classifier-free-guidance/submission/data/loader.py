"""HuggingFace dataset adapters for every benchmark used in the paper.

The benchmark identifiers are taken from the addendum's "Dataset links"
section, so they correspond exactly to the canonical sources used by the
authors.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Optional

try:
    from datasets import load_dataset
except ImportError:  # pragma: no cover
    load_dataset = None  # type: ignore


# ---------------------------------------------------------------------------
# Generic record container
# ---------------------------------------------------------------------------
@dataclass
class BenchmarkSample:
    """Unified record consumed by all evaluators.

    Attributes
    ----------
    context     : prompt fed to the LM
    choices     : list of candidate continuations (multiple-choice tasks)
    answer_idx  : index into `choices` that is correct (-1 if free-form)
    answer_text : free-form ground-truth string for non-MC tasks
    metadata    : original dataset row, kept for debugging / logging
    """

    context: str
    choices: List[str] = field(default_factory=list)
    answer_idx: int = -1
    answer_text: Optional[str] = None
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Zero-shot benchmarks (§3.1)
# ---------------------------------------------------------------------------
def _arc_records(split):
    for r in split:
        ctx = f"Question: {r['question']}\nAnswer:"
        labels = list(r["choices"]["label"])
        choices = list(r["choices"]["text"])
        ans_idx = labels.index(r["answerKey"]) if r["answerKey"] in labels else -1
        yield BenchmarkSample(
            context=ctx, choices=choices, answer_idx=ans_idx, metadata=dict(r)
        )


def _boolq_records(split):
    for r in split:
        ctx = f"{r['passage']}\nQuestion: {r['question']}?\nAnswer:"
        yield BenchmarkSample(
            context=ctx,
            choices=[" no", " yes"],
            answer_idx=int(bool(r["answer"])),
            metadata=dict(r),
        )


def _hellaswag_records(split):
    for r in split:
        ctx = r["ctx"]
        yield BenchmarkSample(
            context=ctx,
            choices=list(r["endings"]),
            answer_idx=int(r["label"]) if r["label"] != "" else -1,
            metadata=dict(r),
        )


def _piqa_records(split):
    for r in split:
        ctx = f"Question: {r['goal']}\nAnswer:"
        yield BenchmarkSample(
            context=ctx,
            choices=[r["sol1"], r["sol2"]],
            answer_idx=int(r["label"]),
            metadata=dict(r),
        )


def _sciq_records(split):
    for r in split:
        ctx = f"{r['support']}\nQuestion: {r['question']}\nAnswer:"
        choices = [
            r["distractor1"],
            r["distractor2"],
            r["distractor3"],
            r["correct_answer"],
        ]
        yield BenchmarkSample(
            context=ctx, choices=choices, answer_idx=3, metadata=dict(r)
        )


def _triviaqa_records(split):
    for r in split:
        ctx = f"Question: {r['question']}\nAnswer:"
        ans = (
            r["answer"]["value"]
            if isinstance(r.get("answer"), dict)
            else r.get("answer", "")
        )
        yield BenchmarkSample(context=ctx, answer_text=ans, metadata=dict(r))


def _winogrande_records(split):
    for r in split:
        sentence = r["sentence"]
        choices = [
            sentence.replace("_", r["option1"]),
            sentence.replace("_", r["option2"]),
        ]
        ans_idx = int(r["answer"]) - 1 if r.get("answer") not in ("", None) else -1
        yield BenchmarkSample(
            context="", choices=choices, answer_idx=ans_idx, metadata=dict(r)
        )


def _lambada_records(split):
    for r in split:
        # LAMBADA: predict the final word given the rest of the passage.
        text = r["text"].rsplit(" ", 1)
        ctx, ans = (text[0], text[1]) if len(text) == 2 else (r["text"], "")
        yield BenchmarkSample(context=ctx + " ", answer_text=ans, metadata=dict(r))


_REGISTRY = {
    "arc_challenge": _arc_records,
    "arc_easy": _arc_records,
    "boolq": _boolq_records,
    "hellaswag": _hellaswag_records,
    "piqa": _piqa_records,
    "sciq": _sciq_records,
    "triviaqa": _triviaqa_records,
    "winogrande": _winogrande_records,
    "lambada": _lambada_records,
}


def load_zero_shot(
    name: str, hf_path: str, config: Optional[str] = None, split: str = "validation"
) -> Iterable[BenchmarkSample]:
    if load_dataset is None:  # pragma: no cover
        raise RuntimeError("`datasets` is required to load benchmarks.")
    ds = load_dataset(hf_path, config) if config else load_dataset(hf_path)
    if split not in ds:
        # fall back to test split if validation isn't available
        split = "test" if "test" in ds else next(iter(ds.keys()))
    record_fn = _REGISTRY[name]
    yield from record_fn(ds[split])


# ---------------------------------------------------------------------------
# Chain-of-Thought (§3.2)
# ---------------------------------------------------------------------------
def load_cot(
    name: str, hf_path: str, config: Optional[str] = None, split: str = "test"
) -> Iterable[BenchmarkSample]:
    if load_dataset is None:  # pragma: no cover
        raise RuntimeError("`datasets` is required.")
    ds = load_dataset(hf_path, config) if config else load_dataset(hf_path)
    if split not in ds:
        split = next(iter(ds.keys()))
    for r in ds[split]:
        if name == "gsm8k":
            ctx = r["question"]
            ans = r["answer"].split("####")[-1].strip()
            yield BenchmarkSample(context=ctx, answer_text=ans, metadata=dict(r))
        elif name == "aqua":
            opts = r.get("options") or [r.get(f"option_{c}") for c in "ABCDE"]
            ctx = f"{r['question']}\nOptions: {', '.join(opts)}\n"
            yield BenchmarkSample(
                context=ctx, answer_text=r["correct"], metadata=dict(r)
            )
        else:
            raise ValueError(f"Unknown CoT dataset: {name}")


# ---------------------------------------------------------------------------
# HumanEval (§3.3.1)
# ---------------------------------------------------------------------------
def load_humaneval(
    hf_path: str = "openai/openai_humaneval",
) -> Iterable[BenchmarkSample]:
    if load_dataset is None:  # pragma: no cover
        raise RuntimeError("`datasets` is required.")
    ds = load_dataset(hf_path)["test"]
    for r in ds:
        # The prompt is exactly the function signature + docstring.
        yield BenchmarkSample(
            context=r["prompt"],
            answer_text=r["canonical_solution"],
            metadata=dict(
                task_id=r["task_id"], test=r["test"], entry_point=r["entry_point"]
            ),
        )

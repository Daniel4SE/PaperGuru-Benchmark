"""P3 sampling per the addendum's recipe (§5).

The addendum specifies:

    * 32,902 datapoints from P3
    * ~50 samples per dataset across the 660 P3 sub-datasets
    * Skip datasets with fewer than 50 samples (take all)
    * Filter out samples whose tokenized input length > 200 tokens
      (called out as a *speed* optimisation -- does not impact results)
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterable, List, Optional

try:
    from datasets import load_dataset, get_dataset_config_names
except ImportError:  # pragma: no cover
    load_dataset = None  # type: ignore
    get_dataset_config_names = None  # type: ignore


@dataclass
class P3SamplerConfig:
    """All knobs needed to reproduce the §5 sampling exactly."""

    hf_path: str = "bigscience/P3"
    total_samples: int = 32_902
    per_dataset_cap: int = 50
    max_input_tokens: int = 200
    seed: int = 1234
    split: str = "train"


def _input_field(row) -> str:
    """P3 records use either `inputs` (text) or `inputs_pretokenized`."""
    if "inputs_pretokenized" in row:
        return row["inputs_pretokenized"]
    return row.get("inputs", "")


def _target_field(row) -> str:
    if "targets_pretokenized" in row:
        return row["targets_pretokenized"]
    return row.get("targets", "")


def sample_p3(cfg: P3SamplerConfig, tokenizer=None) -> Iterable[dict]:
    """Yield a deterministic 32,902-sample slice of P3.

    Parameters
    ----------
    cfg       : P3SamplerConfig
    tokenizer : optional HF tokenizer used for the 200-token length filter.
                If None, no length filtering is performed.
    """
    if load_dataset is None:  # pragma: no cover
        raise RuntimeError("`datasets` is required.")

    rng = random.Random(cfg.seed)
    config_names = get_dataset_config_names(cfg.hf_path)
    rng.shuffle(config_names)

    yielded = 0
    for name in config_names:
        try:
            ds = load_dataset(cfg.hf_path, name, split=cfg.split, streaming=False)
        except Exception:  # pragma: no cover
            continue  # some configs may fail

        # Sample up to `per_dataset_cap` rows
        n = len(ds)
        idx = list(range(n))
        rng.shuffle(idx)
        idx = idx[: cfg.per_dataset_cap]

        for i in idx:
            row = ds[int(i)]
            text_in = _input_field(row)
            text_out = _target_field(row)

            # Length filter (only if tokenizer provided)
            if tokenizer is not None:
                ids = tokenizer(text_in, add_special_tokens=False).input_ids
                if len(ids) > cfg.max_input_tokens:
                    continue

            yield {
                "dataset": name,
                "input": text_in,
                "target": text_out,
                "row": row,
            }
            yielded += 1
            if yielded >= cfg.total_samples:
                return

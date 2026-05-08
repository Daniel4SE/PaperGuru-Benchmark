"""Dataset loaders matching the conventions in Lee et al. 2024.

All four datasets used in the paper are wired up here:

  1. Jigsaw toxic-comment-classification        -- Section 3.1 probe training.
     Per addendum.md, reproductions should use
     ``thesofakillers/jigsaw-toxic-comment-classification-challenge`` on
     HuggingFace (not the Kaggle hosted version).

  2. RealToxicityPrompts (challenge subset)     -- Section 3.3, 5.2 evaluation.
     1,199 prompts that elicit extremely toxic outputs.

  3. Wikitext-2 (raw)                           -- Section 3.3 perplexity, and
     also the source of prompts for the PPLM-paired DPO data (Section 4.2).

  4. Pairwise toxicity preference data          -- the 24,576 (prompt, y+, y-)
     triples produced by ``scripts/build_pairwise_data.py``.

All loaders return either a HuggingFace ``Dataset`` or a PyTorch ``Dataset``
subclass, depending on what the caller needs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import torch
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# 1. Jigsaw -- toxicity probe training data
# ---------------------------------------------------------------------------
def load_jigsaw(split: str = "train", max_samples: int | None = None):
    """Load the HuggingFace mirror specified in addendum.md.

    Returns
    -------
    A list of ``(text, label)`` tuples where label is 0 (non-toxic) or 1 (toxic).
    The Jigsaw dataset has 6 fine-grained labels; per the paper Section 3.1
    we collapse them into a single binary label
        toxic = 1 if any of {toxic, severe_toxic, obscene, threat, insult,
                              identity_hate} is 1, else 0.
    """
    from datasets import load_dataset

    ds = load_dataset(
        "thesofakillers/jigsaw-toxic-comment-classification-challenge",
        split=split,
    )
    out: list[tuple[str, int]] = []
    label_cols = (
        "toxic",
        "severe_toxic",
        "obscene",
        "threat",
        "insult",
        "identity_hate",
    )
    for i, row in enumerate(ds):
        if max_samples is not None and i >= max_samples:
            break
        text = row["comment_text"]
        label = int(any(int(row.get(c, 0)) == 1 for c in label_cols))
        out.append((text, label))
    return out


# ---------------------------------------------------------------------------
# 2. RealToxicityPrompts -- "challenge" subset
# ---------------------------------------------------------------------------
def load_real_toxicity_prompts(
    max_prompts: int = 1199, split: str = "challenge"
) -> list[str]:
    """Load the 1,199 challenge prompts used in Section 3.3 / 5.2.

    Per Gehman et al. 2020, the challenge subset consists of prompts whose
    continuations have the highest Perspective-API toxicity scores in the
    base RealToxicityPrompts corpus.  HuggingFace mirrors it as
    ``allenai/real-toxicity-prompts`` -- we filter the ``challenging`` flag.
    """
    from datasets import load_dataset

    ds = load_dataset("allenai/real-toxicity-prompts", split="train")
    if split == "challenge":
        ds = ds.filter(lambda r: bool(r.get("challenging", False)))

    out: list[str] = []
    for i, row in enumerate(ds):
        if i >= max_prompts:
            break
        prompt = (
            row["prompt"]["text"]
            if isinstance(row.get("prompt"), dict)
            else row["prompt"]
        )
        out.append(prompt)
    return out


# ---------------------------------------------------------------------------
# 3. Wikitext-2 -- perplexity + DPO prompt source
# ---------------------------------------------------------------------------
def load_wikitext_prompts(
    n_prompts: int = 24576, split: str = "train", min_len: int = 16
) -> list[str]:
    """Section 4.2 / 3.3.

    Returns a list of Wikipedia-style sentences from Wikitext-2-raw-v1 to be
    used either as DPO prompts (Section 4.2) or as F1 prompts (Section 3.3).
    """
    from datasets import load_dataset

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    out: list[str] = []
    for row in ds:
        text = (row["text"] or "").strip()
        if len(text.split()) >= min_len and not text.startswith("="):
            out.append(text)
        if len(out) >= n_prompts:
            break
    return out


def iter_wikitext_chunks(seq_len: int = 1024, split: str = "test") -> Iterable[str]:
    """Yield concatenated chunks for sliding-window perplexity on Wikitext-2 test."""
    from datasets import load_dataset

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    full = "\n\n".join(r["text"] for r in ds if r["text"])
    for i in range(0, len(full), seq_len):
        yield full[i : i + seq_len]


# ---------------------------------------------------------------------------
# 4. Pairwise preference data for DPO
# ---------------------------------------------------------------------------
def load_pairwise_dataset(path: str | Path) -> list[dict]:
    """Read the 24,576 (prompt, y+, y-) triples saved by build_pairwise_data.py.

    File format (one JSON object per line):
        {"prompt": "...", "chosen": "y+ ...", "rejected": "y- ..."}
    """
    path = Path(path)
    triples: list[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            triples.append(json.loads(line))
    return triples


class PairwiseToxicityDataset(Dataset):
    """Tokenized DPO dataset.

    Each item returns the four tensors needed by the DPO loss:
        chosen_input_ids, chosen_labels, rejected_input_ids, rejected_labels
    All padded to ``max_seq_len``.  ``labels = -100`` for prompt tokens so the
    DPO loss only sums log-probs of the *response* tokens (standard DPO).
    """

    def __init__(self, triples: list[dict], tokenizer, max_seq_len: int = 128):
        self.triples = triples
        self.tok = tokenizer
        self.max_seq_len = max_seq_len
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token

    def __len__(self):
        return len(self.triples)

    def _encode(self, prompt: str, response: str):
        prompt_ids = self.tok(prompt, add_special_tokens=False).input_ids
        resp_ids = self.tok(response, add_special_tokens=False).input_ids
        ids = (prompt_ids + resp_ids)[: self.max_seq_len]
        labels = ([-100] * len(prompt_ids) + resp_ids)[: self.max_seq_len]
        # right-pad
        pad = self.max_seq_len - len(ids)
        attn = [1] * len(ids) + [0] * pad
        ids = ids + [self.tok.pad_token_id] * pad
        labels = labels + [-100] * pad
        return (
            torch.tensor(ids, dtype=torch.long),
            torch.tensor(labels, dtype=torch.long),
            torch.tensor(attn, dtype=torch.long),
        )

    def __getitem__(self, idx):
        t = self.triples[idx]
        c_ids, c_lab, c_attn = self._encode(t["prompt"], t["chosen"])
        r_ids, r_lab, r_attn = self._encode(t["prompt"], t["rejected"])
        return {
            "chosen_input_ids": c_ids,
            "chosen_labels": c_lab,
            "chosen_attention_mask": c_attn,
            "rejected_input_ids": r_ids,
            "rejected_labels": r_lab,
            "rejected_attention_mask": r_attn,
        }

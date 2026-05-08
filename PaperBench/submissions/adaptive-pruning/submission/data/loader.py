"""Dataset loaders for tasks evaluated in the APT paper (§5.1).

Replication scope (per addendum):
  * GLUE / SST-2          — accuracy
  * GLUE / MNLI           — accuracy (matched dev set)
  * SQuAD v2.0            — F1
  * CNN/DailyMail         — ROUGE-1/2/L

LLaMA / Alpaca / lm-eval-harness are explicitly NOT required.

Datasets are loaded via the HuggingFace `datasets` library; tokenisation
follows the conventions of each task's published baseline.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch.utils.data import DataLoader, Dataset


# --------------------------------------------------------------------------- #
GLUE_TASK_KEYS = {
    "sst2": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "mrpc": ("sentence1", "sentence2"),
    "rte": ("sentence1", "sentence2"),
    "cola": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
}

GLUE_NUM_LABELS = {
    "sst2": 2,
    "mnli": 3,
    "qnli": 2,
    "qqp": 2,
    "mrpc": 2,
    "rte": 2,
    "cola": 2,
    "stsb": 1,
}


# --------------------------------------------------------------------------- #
def get_task_metric(task: str) -> str:
    """Primary dev-set metric for each task (paper §5.1)."""
    if task in GLUE_TASK_KEYS:
        return "accuracy" if task != "stsb" else "spearman"
    if task in {"squad_v2", "squad"}:
        return "f1"
    if task in {"cnn_dm", "cnn_dailymail"}:
        return "rougeL"
    return "accuracy"


# --------------------------------------------------------------------------- #
class _DictDataset(Dataset):
    """Tiny wrapper turning a list of dicts into a torch Dataset."""

    def __init__(self, rows):
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int):
        return self.rows[i]


def _collate(batch):
    out: Dict[str, torch.Tensor] = {}
    keys = batch[0].keys()
    for k in keys:
        if isinstance(batch[0][k], torch.Tensor):
            out[k] = torch.stack([b[k] for b in batch])
        else:
            out[k] = torch.tensor([b[k] for b in batch])
    return out


# --------------------------------------------------------------------------- #
def _tokenise_glue(ds, tokenizer, task, max_len):
    a_key, b_key = GLUE_TASK_KEYS[task]

    def fn(ex):
        if b_key is None:
            enc = tokenizer(
                ex[a_key], truncation=True, padding="max_length", max_length=max_len
            )
        else:
            enc = tokenizer(
                ex[a_key],
                ex[b_key],
                truncation=True,
                padding="max_length",
                max_length=max_len,
            )
        enc["labels"] = ex["label"]
        return enc

    cols_to_remove = [c for c in ds.column_names if c not in {"label", "labels"}]
    return ds.map(fn, batched=False, remove_columns=cols_to_remove)


def _tokenise_squad(ds, tokenizer, max_len, doc_stride=128):
    def fn(ex):
        enc = tokenizer(
            ex["question"].lstrip(),
            ex["context"],
            max_length=max_len,
            truncation="only_second",
            stride=doc_stride,
            return_overflowing_tokens=False,
            return_offsets_mapping=False,
            padding="max_length",
        )
        ans = ex["answers"]
        if not ans["text"]:
            enc["start_positions"] = 0
            enc["end_positions"] = 0
        else:
            start = ans["answer_start"][0]
            end = start + len(ans["text"][0])
            enc["start_positions"] = start
            enc["end_positions"] = end
        return enc

    return ds.map(fn, batched=False, remove_columns=ds.column_names)


# --------------------------------------------------------------------------- #
def build_dataloaders(cfg) -> Tuple[DataLoader, DataLoader, int]:
    """Return (train_loader, eval_loader, num_labels)."""
    from datasets import load_dataset
    from transformers import AutoTokenizer

    task = cfg["data"]["task_name"]
    max_len = cfg["data"]["max_seq_length"]
    tok = AutoTokenizer.from_pretrained(cfg["model"]["name"])

    if task in GLUE_TASK_KEYS:
        raw = load_dataset("glue", task, cache_dir=cfg["data"].get("cache_dir"))
        train = _tokenise_glue(raw["train"], tok, task, max_len)
        # MNLI dev set is "validation_matched"; everything else is "validation".
        val_split = "validation_matched" if task == "mnli" else "validation"
        val = _tokenise_glue(raw[val_split], tok, task, max_len)
        train.set_format(type="torch")
        val.set_format(type="torch")
        train_loader = DataLoader(
            train,
            batch_size=cfg["data"]["train_batch_size"],
            shuffle=True,
            num_workers=cfg["data"].get("num_workers", 0),
        )
        eval_loader = DataLoader(
            val,
            batch_size=cfg["data"]["eval_batch_size"],
            shuffle=False,
            num_workers=cfg["data"].get("num_workers", 0),
        )
        return train_loader, eval_loader, GLUE_NUM_LABELS[task]

    if task in {"squad_v2", "squad"}:
        raw = load_dataset(task, cache_dir=cfg["data"].get("cache_dir"))
        train = _tokenise_squad(raw["train"], tok, max_len)
        val = _tokenise_squad(raw["validation"], tok, max_len)
        train.set_format(type="torch")
        val.set_format(type="torch")
        train_loader = DataLoader(
            train,
            batch_size=cfg["data"]["train_batch_size"],
            shuffle=True,
            num_workers=cfg["data"].get("num_workers", 0),
        )
        eval_loader = DataLoader(
            val,
            batch_size=cfg["data"]["eval_batch_size"],
            shuffle=False,
            num_workers=cfg["data"].get("num_workers", 0),
        )
        return (
            train_loader,
            eval_loader,
            2,
        )  # span start/end heads (handled by AutoModelForQA)

    if task in {"cnn_dm", "cnn_dailymail"}:
        raw = load_dataset(
            "cnn_dailymail", "3.0.0", cache_dir=cfg["data"].get("cache_dir")
        )

        def fn(ex):
            inp = tok(
                ex["article"], truncation=True, padding="max_length", max_length=max_len
            )
            with tok.as_target_tokenizer():
                tgt = tok(
                    ex["highlights"],
                    truncation=True,
                    padding="max_length",
                    max_length=142,
                )
            inp["labels"] = tgt["input_ids"]
            return inp

        train = raw["train"].map(
            fn, batched=False, remove_columns=raw["train"].column_names
        )
        val = raw["validation"].map(
            fn, batched=False, remove_columns=raw["validation"].column_names
        )
        train.set_format(type="torch")
        val.set_format(type="torch")
        train_loader = DataLoader(
            train,
            batch_size=cfg["data"]["train_batch_size"],
            shuffle=True,
            num_workers=cfg["data"].get("num_workers", 0),
        )
        eval_loader = DataLoader(
            val,
            batch_size=cfg["data"]["eval_batch_size"],
            shuffle=False,
            num_workers=cfg["data"].get("num_workers", 0),
        )
        return train_loader, eval_loader, 1

    raise ValueError(f"Unknown task: {task}")

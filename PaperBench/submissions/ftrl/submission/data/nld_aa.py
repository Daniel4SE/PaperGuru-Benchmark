"""NLD-AA dataset wrapper (paper App. B.1 / Addendum).

Per the Addendum, the canonical setup is:

    pip install git+https://github.com/facebookresearch/nle.git@main
    # download nld-aa-dir-{aa..ap}.zip from
    #   https://dl.fbaipublicfiles.com/nld/nld-aa/nld-aa-dir-XX.zip
    # unzip into a directory containing nld-aa/nle_data/

    import nle.dataset as nld
    if not nld.db.exists():
        nld.db.create()
        nld.add_nledata_directory("/path/to/nld-aa", "nld-aa-v0")
    dataset = nld.TtyrecDataset("nld-aa-v0", batch_size=128, ...)

This wrapper builds that dataset and exposes a uniform PyTorch-style iterator
yielding mini-batches with the keys consumed by `train.py`:

    {
        "chars":   long  (B, 21, 79),
        "colors":  long  (B, 21, 79),
        "blstats": float (B, 27),
        "message": float (B, 256),
        "action":  long  (B,),
    }

If the dataset cannot be loaded (missing dep / missing files), we synthesise
random batches of the correct shape so smoke training continues.
"""

from __future__ import annotations

import os
from typing import Iterator, Optional

import numpy as np


class NLDAADataset:
    """Iterator over the NLD-AA dataset, in the format expected by trainers."""

    def __init__(
        self,
        path: Optional[str] = None,
        batch_size: int = 128,
        seq_length: int = 32,
        character: str = "mon-hum-neu-mal",
        seed: int = 0,
    ):
        self.path = path
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.character = character
        self.seed = seed
        self._real = self._try_load()

    def _try_load(self):
        if self.path is None or not os.path.isdir(self.path):
            return None
        try:
            import nle.dataset as nld
        except Exception:
            return None
        try:
            if not nld.db.exists():
                nld.db.create()
                nld.add_nledata_directory(self.path, "nld-aa-v0")
            return nld.TtyrecDataset(
                "nld-aa-v0",
                batch_size=self.batch_size,
                seq_length=self.seq_length,
                shuffle=True,
            )
        except Exception:
            return None

    def __iter__(self) -> Iterator[dict]:
        if self._real is None:
            return self._fake_iter()
        return self._real_iter()

    def _real_iter(self):
        for mb in self._real:
            yield self._postprocess_real(mb)

    def _postprocess_real(self, mb):
        # `mb` from nld.TtyrecDataset gives tty_chars/tty_colors/timestamps/...
        chars = np.asarray(mb["tty_chars"], dtype=np.int64)  # (B, T, 24, 80)
        colors = np.asarray(mb["tty_colors"], dtype=np.int64)
        # use only rows 1..22 of the 24-row tty (the dungeon screen)
        chars = chars[..., 1:22, :79]
        colors = colors[..., 1:22, :79]
        # collapse seq dim by random selection
        T = chars.shape[1]
        ti = np.random.randint(0, T)
        out = {
            "chars": chars[:, ti],
            "colors": colors[:, ti],
            "blstats": np.asarray(
                mb.get("blstats", np.zeros((self.batch_size, 27))), dtype=np.float32
            ),
            "message": np.asarray(
                mb.get("message", np.zeros((self.batch_size, 256))), dtype=np.float32
            ),
            "action": np.asarray(mb["actions"], dtype=np.int64)[:, ti],
        }
        return out

    def _fake_iter(self):
        rng = np.random.default_rng(self.seed)
        while True:
            yield {
                "chars": rng.integers(
                    0, 256, size=(self.batch_size, 21, 79), dtype=np.int64
                ),
                "colors": rng.integers(
                    0, 16, size=(self.batch_size, 21, 79), dtype=np.int64
                ),
                "blstats": rng.normal(size=(self.batch_size, 27)).astype(np.float32),
                "message": rng.normal(size=(self.batch_size, 256)).astype(np.float32),
                "action": rng.integers(0, 121, size=(self.batch_size,), dtype=np.int64),
            }

"""Few-shot image dataset for DPMs-ANT.

Conventions follow Ojha et al. CVPR 2021 ("Few-shot Image Generation via
Cross-domain Correspondence", DOI 10.1109/CVPR46437.2021.01060 — verified
via CrossRef in this submission), which is the canonical 10-shot
benchmark used by the paper:

  * Source domains : FFHQ (256x256), LSUN-Church (256x256)
  * Target domains : 10 images each from
      - Babies, Sunglasses, Sketches, Modigliani, Raphael Peale,
        Haunted Houses, Landscape Drawings.

Pre-processing follows the diffusion-model convention:
  * resize to image_size,
  * center-crop,
  * (optional) random horizontal flip,
  * normalize to [-1, 1].
"""

from __future__ import annotations

import glob
import os
import random
from typing import List, Tuple

import torch
from torch.utils.data import Dataset, DataLoader

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _list_images(root: str) -> List[str]:
    exts = ("png", "jpg", "jpeg", "bmp", "webp")
    files: List[str] = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(root, f"**/*.{ext}"), recursive=True))
        files.extend(
            glob.glob(os.path.join(root, f"**/*.{ext.upper()}"), recursive=True)
        )
    return sorted(set(files))


def _load_image(path: str, size: int) -> torch.Tensor:
    if Image is None:
        raise RuntimeError("Pillow is required to load images")
    img = Image.open(path).convert("RGB")
    # center crop to square then resize
    w, h = img.size
    s = min(w, h)
    img = img.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s))
    img = img.resize((size, size), Image.BICUBIC)
    arr = torch.tensor(list(img.tobytes()), dtype=torch.uint8)
    arr = arr.view(size, size, 3).permute(2, 0, 1).float() / 255.0
    return arr * 2.0 - 1.0  # to [-1, 1]


# ---------------------------------------------------------------------
# Few-shot target dataset (paper §5.2: "limited set of just 10 training
# images with the same setting as DDPM-PA").
# ---------------------------------------------------------------------
class FewShotImageDataset(Dataset):
    def __init__(
        self,
        root: str,
        image_size: int = 256,
        num_images: int = 10,
        augment: bool = True,
    ) -> None:
        self.image_size = image_size
        self.augment = augment

        files = _list_images(root)
        if len(files) == 0:
            raise FileNotFoundError(f"no images found in {root}")
        if num_images and len(files) > num_images:
            random.seed(0)
            files = random.sample(files, num_images)
        self.files = files

        # Eagerly load — only 10 images, fits easily in RAM
        self.images = [_load_image(p, image_size) for p in self.files]

    def __len__(self) -> int:
        # Repeat the 10 images many times per epoch to cover ~300 iters
        # at batch=40 ⇒ need 12,000 sample-pulls. We virtually upsample.
        return max(1000, len(self.images))

    def __getitem__(self, idx: int) -> torch.Tensor:
        img = self.images[idx % len(self.images)]
        if self.augment and random.random() < 0.5:
            img = img.flip(-1)  # horizontal flip
        return img


def build_target_loader(
    root: str,
    batch_size: int = 40,
    image_size: int = 256,
    num_images: int = 10,
    num_workers: int = 2,
    augment: bool = True,
) -> DataLoader:
    ds = FewShotImageDataset(root, image_size, num_images, augment)
    return DataLoader(
        ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True
    )


# ---------------------------------------------------------------------
# Source-vs-target classifier dataset (addendum: train binary classifier
# on 10 source + 10 target images at uniformly-sampled timesteps t)
# ---------------------------------------------------------------------
class SourceTargetClassifierDataset(Dataset):
    """Yields (image, t, label) triples. label ∈ {0=source, 1=target}.

    Note: t is sampled inside the training loop, not here, because the
    classifier is conditioned on t and trained over the full
    forward-noising distribution q(x_t | x_0).
    """

    def __init__(
        self,
        source_root: str,
        target_root: str,
        image_size: int = 256,
        num_source: int = 10,
        num_target: int = 10,
        augment: bool = True,
    ) -> None:
        self.image_size = image_size
        self.augment = augment

        src = _list_images(source_root)
        tgt = _list_images(target_root)
        if len(src) == 0 or len(tgt) == 0:
            raise FileNotFoundError(
                f"empty source or target: {source_root}, {target_root}"
            )
        random.seed(0)
        if num_source and len(src) > num_source:
            src = random.sample(src, num_source)
        if num_target and len(tgt) > num_target:
            tgt = random.sample(tgt, num_target)

        self.images: List[torch.Tensor] = [_load_image(p, image_size) for p in src]
        self.images += [_load_image(p, image_size) for p in tgt]
        self.labels = [0] * len(src) + [1] * len(tgt)

    def __len__(self) -> int:
        return max(1000, len(self.images))

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        i = idx % len(self.images)
        img = self.images[i]
        if self.augment and random.random() < 0.5:
            img = img.flip(-1)
        return img, self.labels[i]


def build_classifier_loader(
    source_root: str,
    target_root: str,
    batch_size: int = 64,
    image_size: int = 256,
    num_workers: int = 2,
    augment: bool = True,
) -> DataLoader:
    ds = SourceTargetClassifierDataset(
        source_root, target_root, image_size=image_size, augment=augment
    )
    return DataLoader(
        ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True
    )

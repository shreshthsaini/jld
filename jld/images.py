"""Image loading with the crop rule used for every JLD number in the paper.

Images are kept at native resolution and centre-cropped so that height and
width are multiples of the encoder patch size (14 for DINOv2). When a reference
and a distorted image differ in size, both are cropped to their common
top-left extent.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image


def to_tensor(image: Image.Image) -> torch.Tensor:
    """PIL image to a float tensor [1, 3, H, W] in [0, 1]."""
    array = np.asarray(image.convert("RGB")).copy()
    return torch.from_numpy(array).permute(2, 0, 1).float().unsqueeze(0) / 255.0


def center_crop_to_multiple(x: torch.Tensor, patch: int) -> torch.Tensor:
    """Centre-crop a [..., H, W] tensor so that H and W are multiples of ``patch``."""
    h, w = x.shape[-2:]
    hc, wc = (h // patch) * patch, (w // patch) * patch
    if hc == 0 or wc == 0:
        raise ValueError(f"image {h}x{w} is smaller than one {patch}x{patch} patch")
    top, left = (h - hc) // 2, (w - wc) // 2
    return x[..., top : top + hc, left : left + wc]


def load_image(path: str | Path, patch: int = 14) -> torch.Tensor:
    """Load an image as [1, 3, H, W] in [0, 1], centre-cropped to multiples of ``patch``."""
    with Image.open(path) as image:
        x = to_tensor(image)
    return center_crop_to_multiple(x, patch)


def load_pair(ref: str | Path, dist: str | Path, patch: int = 14) -> tuple[torch.Tensor, torch.Tensor]:
    """Load a reference/distorted pair and crop both to their common size.

    ``patch=14`` is the JLD protocol. ``patch=1`` keeps every pixel and is the
    protocol used for the PSNR, SSIM, LPIPS and DISTS baselines.
    """
    x1, x2 = load_image(ref, patch), load_image(dist, patch)
    if x1.shape != x2.shape:
        h = min(x1.shape[2], x2.shape[2]) // patch * patch
        w = min(x1.shape[3], x2.shape[3]) // patch * patch
        x1, x2 = x1[:, :, :h, :w], x2[:, :, :h, :w]
    return x1, x2


__all__ = ["center_crop_to_multiple", "load_image", "load_pair", "to_tensor"]

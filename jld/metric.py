"""The Jacobian-lens distance (JLD).

    JLD(x, y) = sqrt( mean_t || U^T (h_t(x') - h_t(y')) ||^2 ) + w * (1 - cos(c(x'), c(y')))

x' and y' are the images after the chroma prefilter, h_t are the block-1 patch
tokens of the frozen encoder, U is the fitted lens (top-64 eigenvectors of
M = E[J^T J]) and c is the encoder's final CLS token. Lower means more similar.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import Tensor

from jld.encoder import DEFAULT_ENCODER, ViTEncoder
from jld.frontend import chroma_lowpass
from jld.images import center_crop_to_multiple, load_pair, to_tensor
from jld.lens import Lens

PRETRAINED_LENS = Path(__file__).resolve().parent / "data" / "jld_dinov2_s14_block1_k64.npz"

# name: (use the lens, chroma sigma in pixels, CLS weight)
VARIANTS = {
    "full": (True, 2.0, 0.5),   # JLD: lens + chroma prefilter + CLS term
    "fast": (True, 2.0, 0.0),   # JLD-fast: stops after block 1
    "lens": (True, 0.0, 0.0),   # lens alone (ablation)
    "raw": (False, 0.0, 0.0),   # raw 384-d block-1 tokens, no lens (ablation)
}


class JLD:
    """Full-reference distance between a reference and a distorted image.

    ``m(ref, dist)`` scores without building an autograd graph and returns a
    CPU tensor with one distance per pair. ``m.distance_per_sample_tensor`` and
    ``m.distance_tensor`` keep the graph so the distance can be used as a loss.
    """

    def __init__(
        self,
        encoder: ViTEncoder,
        U: np.ndarray | Tensor | None,
        chroma_sigma: float = 2.0,
        cls_weight: float = 0.5,
    ) -> None:
        self.encoder = encoder
        if U is None:  # identity: raw tokens
            U = np.eye(encoder.dim, dtype=np.float32)
        self.U = torch.as_tensor(np.asarray(U, dtype=np.float32), device=encoder.device)
        if self.U.shape[0] != encoder.dim:
            raise ValueError(f"lens has {self.U.shape[0]} rows, encoder width is {encoder.dim}")
        self.chroma_sigma = float(chroma_sigma)
        self.cls_weight = float(cls_weight)

    @classmethod
    def pretrained(
        cls,
        variant: str = "full",
        device: str | torch.device | None = None,
        lens_path: str | Path | None = None,
    ) -> "JLD":
        """JLD with the shipped DINOv2-S/14 block-1 lens (k = 64)."""
        if variant not in VARIANTS:
            raise ValueError(f"variant must be one of {sorted(VARIANTS)}")
        use_lens, sigma, weight = VARIANTS[variant]
        lens = Lens.load(lens_path or PRETRAINED_LENS)
        encoder = ViTEncoder(
            lens.meta.get("encoder", DEFAULT_ENCODER), lens.meta.get("layer", 1), device
        )
        return cls(encoder, lens.U if use_lens else None, sigma, weight)

    @classmethod
    def from_lens(
        cls,
        lens_path: str | Path,
        device: str | torch.device | None = None,
        chroma_sigma: float = 2.0,
        cls_weight: float = 0.5,
    ) -> "JLD":
        """JLD with any lens written by ``fit_lens.py`` (its encoder and block are read from the file)."""
        lens = Lens.load(lens_path)
        encoder = ViTEncoder(lens.meta["encoder"], lens.meta["layer"], device)
        return cls(encoder, lens.U, chroma_sigma, cls_weight)

    # ------------------------------------------------------------------ inputs
    def _prepare(self, ref, dist) -> tuple[Tensor, Tensor]:
        patch = self.encoder.patch
        if isinstance(ref, (str, Path)) and isinstance(dist, (str, Path)):
            x1, x2 = load_pair(ref, dist, patch)
        else:
            x1, x2 = (self._as_tensor(v) for v in (ref, dist))
            x1, x2 = center_crop_to_multiple(x1, patch), center_crop_to_multiple(x2, patch)
        if x1.shape[-2:] != x2.shape[-2:]:
            raise ValueError(f"image sizes differ: {tuple(x1.shape[-2:])} vs {tuple(x2.shape[-2:])}")
        if x1.shape[0] != x2.shape[0] and 1 not in (x1.shape[0], x2.shape[0]):
            raise ValueError(f"batch sizes {x1.shape[0]} and {x2.shape[0]} are not broadcastable")
        device = self.encoder.device
        return x1.to(device, torch.float32), x2.to(device, torch.float32)

    @staticmethod
    def _as_tensor(value) -> Tensor:
        if isinstance(value, (str, Path)):
            with Image.open(value) as image:
                return to_tensor(image)
        if isinstance(value, Image.Image):
            return to_tensor(value)
        if isinstance(value, np.ndarray):  # H x W x 3, uint8 or float in [0, 1]
            value = torch.from_numpy(value.astype(np.float32) / (255.0 if value.dtype == np.uint8 else 1.0))
            value = value.permute(2, 0, 1)
        if not isinstance(value, Tensor):
            raise TypeError(f"unsupported image type {type(value).__name__}")
        return value.unsqueeze(0) if value.ndim == 3 else value

    # ---------------------------------------------------------------- distance
    def _encode(self, x: Tensor) -> tuple[Tensor, Tensor | None]:
        if self.chroma_sigma > 0:
            x = chroma_lowpass(x, self.chroma_sigma)
        hidden, cls = self.encoder(x, need_cls=self.cls_weight > 0)
        return self.encoder.patch_tokens(hidden).float(), cls

    def _patch_sq(self, x1: Tensor, x2: Tensor) -> tuple[Tensor, Tensor | None]:
        """Squared projected displacement per patch [B, N] and the CLS term [B] (or None)."""
        if x1.shape == x2.shape:  # one forward pass for both images
            tokens, cls = self._encode(torch.cat([x1, x2]))
            b = x1.shape[0]
            t1, t2 = tokens[:b], tokens[b:]
            c1, c2 = (cls[:b], cls[b:]) if cls is not None else (None, None)
        else:
            t1, c1 = self._encode(x1)
            t2, c2 = self._encode(x2)
        delta = (t1 - t2) @ self.U
        squared = delta.square().sum(dim=-1)
        cls_term = None
        if self.cls_weight > 0:
            cls_term = (1 - torch.nn.functional.cosine_similarity(c1, c2, dim=-1)).clamp_min(0)
        return squared, cls_term

    def distance_per_sample_tensor(self, ref, dist) -> Tensor:
        """Differentiable distance, one value per pair (either batch may be 1)."""
        x1, x2 = self._prepare(ref, dist)
        squared, cls_term = self._patch_sq(x1, x2)
        distance = squared.mean(dim=1).clamp_min(0).sqrt()
        if cls_term is not None:
            distance = distance + self.cls_weight * cls_term
        return distance

    def distance_tensor(self, ref, dist) -> Tensor:
        """Differentiable scalar: the distance averaged over the batch."""
        return self.distance_per_sample_tensor(ref, dist).mean()

    def __call__(self, ref, dist) -> Tensor:
        """Score without gradients; returns a CPU tensor [B]."""
        with torch.inference_mode():
            return self.distance_per_sample_tensor(ref, dist).float().cpu()

    def map(self, ref, dist) -> Tensor:
        """Per-patch projected displacement ||U^T (h_t(x) - h_t(y))|| as a [B, H/14, W/14] map."""
        with torch.inference_mode():
            x1, x2 = self._prepare(ref, dist)
            squared, _ = self._patch_sq(x1, x2)
            gh, gw = x1.shape[-2] // self.encoder.patch, x1.shape[-1] // self.encoder.patch
            return squared.clamp_min(0).sqrt().reshape(-1, gh, gw).float().cpu()


__all__ = ["JLD", "PRETRAINED_LENS", "VARIANTS"]

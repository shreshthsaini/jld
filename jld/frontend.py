"""Chroma prefilter applied to both images before the encoder.

Humans resolve chroma more coarsely than luminance. The prefilter converts RGB
to YCbCr, blurs only Cb and Cr with a Gaussian of ``sigma`` pixels, and converts
back. It is linear (up to the final clamp), fixed and differentiable.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def _gauss1d(sigma: float, device: torch.device) -> torch.Tensor:
    radius = max(1, int(math.ceil(3 * sigma)))
    t = torch.arange(-radius, radius + 1, device=device, dtype=torch.float32)
    k = torch.exp(-0.5 * (t / sigma) ** 2)
    return k / k.sum()


def gaussian_blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable Gaussian blur with reflect padding for [B, C, H, W] tensors."""
    if sigma <= 0:
        return x
    k = _gauss1d(sigma, x.device).to(x.dtype)
    r = (k.numel() - 1) // 2
    c = x.shape[1]
    kh = k.view(1, 1, 1, -1).repeat(c, 1, 1, 1)
    kv = k.view(1, 1, -1, 1).repeat(c, 1, 1, 1)
    rw, rh = min(r, x.shape[-1] - 1), min(r, x.shape[-2] - 1)
    kh = kh[..., r - rw : r + rw + 1]
    kh = kh / kh.sum(dim=-1, keepdim=True)
    kv = kv[:, :, r - rh : r + rh + 1]
    kv = kv / kv.sum(dim=2, keepdim=True)
    y = F.conv2d(F.pad(x, (rw, rw, 0, 0), mode="reflect"), kh, groups=c)
    y = F.conv2d(F.pad(y, (0, 0, rh, rh), mode="reflect"), kv, groups=c)
    return y


def to_ycbcr(x: torch.Tensor) -> torch.Tensor:
    r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    return torch.cat([y, (b - y) * 0.564 + 0.5, (r - y) * 0.713 + 0.5], 1)


def from_ycbcr(x: torch.Tensor) -> torch.Tensor:
    y, cb, cr = x[:, 0:1], x[:, 1:2] - 0.5, x[:, 2:3] - 0.5
    return torch.cat([y + 1.403 * cr, y - 0.344 * cb - 0.714 * cr, y + 1.773 * cb], 1).clamp(0, 1)


def chroma_lowpass(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Blur the Cb and Cr channels of an RGB image in [0, 1]; luminance is untouched."""
    if sigma <= 0:
        return x
    ycc = to_ycbcr(x)
    return from_ycbcr(torch.cat([ycc[:, 0:1], gaussian_blur(ycc[:, 1:3], sigma)], 1))


__all__ = ["chroma_lowpass", "from_ycbcr", "gaussian_blur", "to_ycbcr"]

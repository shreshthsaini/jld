"""A registered PyTorch module using JLD's unchanged scoring operations."""
from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor, nn

from jld.metric import JLD


class JLDModule(nn.Module):
    """Differentiable per-pair JLD for RGB tensors in [0, 1].

    The encoder is a frozen child module and the fitted lens is a buffer, so
    ``.to(device)`` moves both. Native resolution and the official center crop
    to a multiple of 14 are preserved. No resize is applied. Inputs may also
    use the image types accepted by :class:`jld.JLD`.

    Pass an existing ``metric`` to reuse its encoder and lens. The default
    constructor loads pretrained weights on CPU; move the module explicitly
    for accelerator evaluation. Scores stay on the module's device.
    """

    def __init__(
        self,
        variant: str = "full",
        device: str | torch.device = "cpu",
        lens_path: str | Path | None = None,
        metric: JLD | None = None,
    ) -> None:
        super().__init__()
        if metric is not None and (variant != "full" or lens_path is not None):
            raise ValueError("metric cannot be combined with variant or lens_path overrides")
        source = metric if metric is not None else JLD.pretrained(variant, device, lens_path)
        self.encoder = source.encoder
        self.register_buffer("U", source.U.detach().clone())
        self.chroma_sigma = source.chroma_sigma
        self.cls_weight = source.cls_weight
        self.to(device)
        self.eval()

    def train(self, mode: bool = True) -> "JLDModule":
        # Input gradients remain available while the pretrained encoder stays
        # in evaluation mode, including when used in a training loss.
        super().train(mode)
        self.encoder.eval()
        return self

    # Delegate the complete scoring path to the official implementation. These
    # methods access this module's registered encoder and lens directly, which
    # also avoids stale tensor references after a device move.
    _as_tensor = staticmethod(JLD._as_tensor)
    _prepare = JLD._prepare
    _encode = JLD._encode
    _patch_sq = JLD._patch_sq
    distance_per_sample_tensor = JLD.distance_per_sample_tensor
    distance_tensor = JLD.distance_tensor
    map = JLD.map

    def forward(self, ref, dist) -> Tensor:
        """Return one differentiable score for each image pair, shape [B]."""
        return self.distance_per_sample_tensor(ref, dist)


__all__ = ["JLDModule"]

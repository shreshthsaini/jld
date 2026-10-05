"""Frozen ViT encoder with residual-stream capture.

The encoder returns the residual stream after block ``layer`` (all tokens,
including prefix tokens) and, when requested, the final CLS token after the
last LayerNorm. Any timm ViT that exposes ``patch_embed``, ``_pos_embed``,
``blocks`` and ``norm`` works, so swapping the encoder is a one-argument change.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn

DEFAULT_ENCODER = "vit_small_patch14_dinov2.lvd142m"


class ViTEncoder(nn.Module):
    """A frozen timm ViT that exposes one intermediate block and the final CLS token."""

    def __init__(
        self,
        model_name: str = DEFAULT_ENCODER,
        layer: int = 1,
        device: str | torch.device | None = None,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        import timm

        self.model = timm.create_model(model_name, pretrained=pretrained, dynamic_img_size=True)
        self.model_name = model_name
        self.layer = int(layer)
        depth = len(self.model.blocks)
        if not 0 <= self.layer <= depth:
            raise ValueError(f"layer must be in 0..{depth} for {model_name}, got {layer}")
        self.patch = int(self.model.patch_embed.patch_size[0])
        self.n_prefix = int(self.model.num_prefix_tokens)
        self.dim = int(self.model.embed_dim)
        cfg = self.model.pretrained_cfg
        mean = cfg.get("mean", (0.485, 0.456, 0.406))
        std = cfg.get("std", (0.229, 0.224, 0.225))
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1), persistent=False)
        self._is_eva = self.model.__class__.__name__ == "Eva"  # DINOv3 and other RoPE ViTs
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.to(device=device, dtype=torch.float32)
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    @property
    def device(self) -> torch.device:
        return self.mean.device

    def forward(self, x: Tensor, need_cls: bool = True) -> tuple[Tensor, Tensor | None]:
        """Encode RGB images in [0, 1] of shape [B, 3, H, W].

        Returns ``(hidden, cls)``: ``hidden`` is the residual stream after block
        ``layer`` with shape [B, n_prefix + N, dim]; ``cls`` is the final CLS token
        [B, dim] (the mean final patch token for encoders without a CLS token), or
        ``None`` when ``need_cls`` is False, in which case the forward pass stops
        after block ``layer``.
        """
        model = self.model
        hidden = model.patch_embed((x - self.mean) / self.std)
        rope = None
        if self._is_eva:
            hidden, rope = model._pos_embed(hidden)
        else:
            hidden = model._pos_embed(hidden)
            hidden = model.patch_drop(hidden)
        hidden = model.norm_pre(hidden)

        captured = hidden if self.layer == 0 else None
        for index, block in enumerate(model.blocks, start=1):
            if captured is not None and not need_cls:
                break
            if self._is_eva:
                mixed = getattr(model, "rope_mixed", False) and rope is not None
                hidden = block(hidden, rope=rope[index - 1] if mixed else rope)
            else:
                hidden = block(hidden)
            if index == self.layer:
                captured = hidden
        if not need_cls:
            return captured, None
        final = model.norm(hidden)
        cls = final[:, 0] if self.n_prefix else final.mean(dim=1)
        return captured, cls.float()

    def patch_tokens(self, hidden: Tensor) -> Tensor:
        """Drop the prefix (CLS and register) tokens."""
        return hidden[:, self.n_prefix :]


__all__ = ["DEFAULT_ENCODER", "ViTEncoder"]

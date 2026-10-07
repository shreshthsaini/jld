"""Author-maintained TorchMetrics-compatible JLD metric."""
from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor

try:
    from torchmetrics import Metric
except ModuleNotFoundError as error:
    if error.name == "torchmetrics":
        raise ImportError("Install TorchMetrics with: uv pip install 'jacobian-lens-distance[torchmetrics]'") from error
    raise

from jld.integrations.module import JLDModule
from jld.metric import JLD


class JacobianLensDistance(Metric):
    """Sample-weighted mean JLD with distributed sum and count reduction.

    ``metric(preds, target)`` returns the current batch mean and accumulates
    evaluation state. ``update(preds, target)`` followed by ``compute()``
    returns the mean over all pairs, even for batches or ranks of unequal
    size. ``reset()`` clears evaluation state. ``forward`` supports gradients;
    use :class:`JLDModule` when per-pair losses are needed.

    Inputs are RGB float tensors in [0, 1], shape [B, 3, H, W]. A single
    reference or distorted image may broadcast over the other batch, as in
    the standalone implementation. The encoder stays frozen in eval mode.
    """

    is_differentiable = True
    higher_is_better = False
    full_state_update = False

    def __init__(
        self,
        variant: str = "full",
        device: str | torch.device = "cpu",
        lens_path: str | Path | None = None,
        metric: JLD | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.scorer = JLDModule(variant, device, lens_path, metric)
        self.add_state("score_sum", default=torch.tensor(0.0, dtype=torch.float64), dist_reduce_fx="sum")
        self.add_state("pair_count", default=torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum")
        self.to(device)

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Accumulate all per-pair distances without averaging batches first."""
        if preds.ndim != 4 or target.ndim != 4 or preds.shape[1] != 3 or target.shape[1] != 3:
            raise ValueError("JLD requires RGB tensors with shape [B, 3, H, W]")
        scores = self.scorer(target, preds)
        self.score_sum = self.score_sum + scores.to(self.score_sum).sum()
        self.pair_count = self.pair_count + scores.numel()

    def compute(self) -> Tensor:
        """Return the pair mean; no observations yield NaN."""
        return self.score_sum / self.pair_count


__all__ = ["JacobianLensDistance"]

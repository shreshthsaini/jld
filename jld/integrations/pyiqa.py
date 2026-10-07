"""Author-maintained PyIQA plugin for ``jld`` and ``jld_fast``.

Call ``register()`` before ``pyiqa.create_metric`` or ``pyiqa.list_models``.
PyIQA's architecture entry points also invoke this function, but its factory
checks default configurations before loading architecture plugins. Explicit
registration is therefore required for reliable discovery.
"""
from __future__ import annotations

from copy import deepcopy

from torch import Tensor

from jld.integrations.module import JLDModule


class JLD(JLDModule):
    """PyIQA architecture, accepting distorted image first and reference second."""

    def forward(self, target: Tensor, ref: Tensor) -> Tensor:
        if target.ndim != 4 or ref.ndim != 4 or target.shape[1] != 3 or ref.shape[1] != 3:
            raise ValueError("JLD requires RGB tensors with shape [B, 3, H, W]")
        return self.distance_per_sample_tensor(ref, target)


_CONFIGS = {
    "jld": {
        "metric_opts": {"type": "JLD", "variant": "full"},
        "metric_mode": "FR",
        "lower_better": True,
        "score_range": "0, +inf",
    },
    "jld_fast": {
        "metric_opts": {"type": "JLD", "variant": "fast"},
        "metric_mode": "FR",
        "lower_better": True,
        "score_range": "0, +inf",
    },
}


def register() -> None:
    """Register both metrics without overwriting another implementation.

    Repeated calls are safe. Install the ``pyiqa`` extra before calling this
    function. The plugin registers no weights and downloads none until a
    metric is constructed.
    """
    try:
        from pyiqa.default_model_configs import DEFAULT_CONFIGS
        from pyiqa.utils.registry import ARCH_REGISTRY
    except ModuleNotFoundError as error:
        if error.name == "pyiqa":
            raise ImportError("Install PyIQA with: uv pip install 'jacobian-lens-distance[pyiqa]'") from error
        raise

    # Validate every collision before making changes. The registry may load
    # this same entry point during the membership check; its loader guards
    # recursive entry-point loading, and the inner call registers this class.
    if "JLD" in ARCH_REGISTRY and ARCH_REGISTRY.get("JLD") is not JLD:
        raise RuntimeError("PyIQA already contains a different JLD architecture")
    for name, config in _CONFIGS.items():
        if name in DEFAULT_CONFIGS and DEFAULT_CONFIGS[name] != config:
            raise RuntimeError(f"PyIQA already contains a different {name} configuration")
    if "JLD" not in ARCH_REGISTRY:
        ARCH_REGISTRY.register(JLD)
    for name, config in _CONFIGS.items():
        DEFAULT_CONFIGS.setdefault(name, deepcopy(config))


__all__ = ["JLD", "register"]

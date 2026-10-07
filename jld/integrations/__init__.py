"""Optional evaluation-tool adapters for the official JLD implementation.

Import an adapter explicitly to avoid requiring PyIQA or TorchMetrics when
using the standalone package.
"""

from jld.integrations.module import JLDModule

__all__ = ["JLDModule"]

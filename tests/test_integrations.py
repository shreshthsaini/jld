"""Scoring parity and evaluation lifecycle of the optional adapters."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

from jld import JLD, load_image
from jld.integrations.module import JLDModule

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


class TinyEncoder(nn.Module):
    """A deterministic differentiable encoder for lifecycle and DDP checks."""

    dim, patch, n_prefix = 4, 14, 1

    def __init__(self):
        super().__init__()
        self.register_buffer("mean", torch.zeros(1))
        self.model = nn.Conv2d(3, self.dim, self.patch, stride=self.patch, bias=False)
        with torch.no_grad():
            self.model.weight.copy_(torch.arange(self.model.weight.numel()).reshape_as(self.model.weight) / 10000)
        self.model.requires_grad_(False)

    @property
    def device(self):
        return self.mean.device

    def forward(self, x, need_cls=True):
        patches = self.model(x).flatten(2).transpose(1, 2)
        cls = patches.mean(dim=1)
        return torch.cat([cls[:, None], patches], dim=1), cls if need_cls else None

    def patch_tokens(self, hidden):
        return hidden[:, 1:]


def tiny_metric():
    return JLD(TinyEncoder(), np.eye(4, dtype=np.float32), chroma_sigma=0, cls_weight=0.5)


def pairs():
    rng = torch.Generator().manual_seed(7)
    return torch.rand(3, 3, 31, 44, generator=rng), torch.rand(3, 3, 31, 44, generator=rng)


@pytest.fixture(scope="module", params=["full", "fast"])
def pretrained_adapter(request):
    official = JLD.pretrained(request.param, device="cpu")
    return official, JLDModule(metric=official)


def test_pretrained_module_parity_and_gradients(pretrained_adapter):
    official, adapter = pretrained_adapter
    rng = torch.Generator().manual_seed(11)
    ref = torch.rand(1, 3, 45, 60, generator=rng)
    target = torch.rand(2, 3, 45, 60, generator=rng, requires_grad=True)
    expected = official(ref, target)
    actual = adapter(ref, target)
    torch.testing.assert_close(actual.detach(), expected, rtol=1e-5, atol=1e-6)
    actual.mean().backward()
    assert target.grad is not None and torch.isfinite(target.grad).all()
    assert target.grad.abs().sum() > 0
    assert all(p.grad is None for p in adapter.encoder.parameters())
    adapter.train()
    assert not adapter.encoder.training
    adapter.to("cpu")
    assert dict(adapter.named_children())["encoder"] is official.encoder
    assert "U" in dict(adapter.named_buffers())
    assert "U" in adapter.state_dict()
    torch.testing.assert_close(adapter.U, official.U)


def test_pretrained_pyiqa_factory_and_loss(pretrained_adapter):
    pyiqa = pytest.importorskip("pyiqa")
    from jld.integrations.pyiqa import register

    official, adapter = pretrained_adapter
    register()
    register()
    assert {"jld", "jld_fast"} <= set(pyiqa.list_models(metric_mode="FR"))
    name = "jld_fast" if official.cls_weight == 0 else "jld"
    # Reuse the actual pretrained encoder. The factory exercises registration,
    # metadata, device moves, input validation, and PyIQA's loss reduction.
    wrapped = pyiqa.create_metric(name, device="cpu", metric=official, variant="full")
    assert wrapped.lower_better and wrapped.metric_mode == "FR"
    ref = load_image(EXAMPLES / "parrots.png")
    target = load_image(EXAMPLES / "parrots_jpeg_q8.png")
    torch.testing.assert_close(wrapped(target, ref), official(ref, target), rtol=1e-5, atol=1e-6)
    assert not wrapped(target, ref).requires_grad
    loss_metric = pyiqa.create_metric(name, as_loss=True, device="cpu", metric=official, variant="full")
    small_ref, small_target = ref[..., :42, :56], target[..., :42, :56].clone().requires_grad_()
    loss_metric(small_target, small_ref).backward()
    assert small_target.grad is not None and small_target.grad.abs().sum() > 0
    with pytest.raises(ValueError, match="RGB"):
        wrapped(torch.rand(1, 1, 42, 56), torch.rand(1, 1, 42, 56))


def test_pretrained_torchmetrics_parity(pretrained_adapter):
    pytest.importorskip("torchmetrics")
    from jld.integrations.torchmetrics import JacobianLensDistance

    official, _ = pretrained_adapter
    metric = JacobianLensDistance(metric=official)
    ref, target = pairs()
    torch.testing.assert_close(metric(target, ref), official(ref, target).double().mean(), rtol=1e-5, atol=1e-6)


def test_torchmetrics_unequal_batches_reset_and_gradients():
    pytest.importorskip("torchmetrics")
    from jld.integrations.torchmetrics import JacobianLensDistance

    official = tiny_metric()
    metric = JacobianLensDistance(metric=official)
    ref, target = pairs()
    expected = official(ref, target).double().mean()
    metric.update(target[:1], ref[:1])
    metric.update(target[1:], ref[1:])
    torch.testing.assert_close(metric.compute(), expected)
    assert metric.pair_count == 3
    assert metric._reductions["score_sum"] is not None
    assert metric._reductions["pair_count"] is not None
    metric.reset()
    assert metric.pair_count == 0 and metric.score_sum == 0
    differentiable_target = target.clone().requires_grad_()
    value = metric(differentiable_target, ref)
    torch.testing.assert_close(value.detach(), expected)
    value.backward()
    assert differentiable_target.grad is not None and differentiable_target.grad.abs().sum() > 0
    assert metric.pair_count == 3
    assert "scorer.U" in metric.state_dict()
    metric.to("cpu")
    assert metric.scorer.encoder.device == metric.scorer.U.device == metric.score_sum.device


def _distributed_worker(rank, rendezvous, output):
    from jld.integrations.torchmetrics import JacobianLensDistance

    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2)
    try:
        metric = JacobianLensDistance(metric=tiny_metric())
        ref, target = pairs()
        selection = slice(0, 1) if rank == 0 else slice(1, 3)
        metric.update(target[selection], ref[selection])
        value = metric.compute()
        torch.save(value, Path(output) / f"rank-{rank}.pt")
    finally:
        dist.destroy_process_group()


def test_torchmetrics_distributed_unequal_ranks(tmp_path):
    pytest.importorskip("torchmetrics")
    if not dist.is_available() or not dist.is_gloo_available():
        pytest.skip("Gloo unavailable")
    mp.spawn(_distributed_worker, args=(str(tmp_path / "rendezvous"), str(tmp_path)), nprocs=2, join=True)
    ref, target = pairs()
    expected = tiny_metric()(ref, target).double().mean()
    for rank in range(2):
        actual = torch.load(tmp_path / f"rank-{rank}.pt", weights_only=True)
        torch.testing.assert_close(actual, expected)


def test_pyiqa_registration_preserves_existing_config(monkeypatch):
    pytest.importorskip("pyiqa")
    from pyiqa.default_model_configs import DEFAULT_CONFIGS
    from jld.integrations.pyiqa import register

    register()
    replacement = {"metric_opts": {"type": "AnotherJLD"}}
    monkeypatch.setitem(DEFAULT_CONFIGS, "jld", replacement)
    with pytest.raises(RuntimeError, match="different jld configuration"):
        register()
    assert DEFAULT_CONFIGS["jld"] is replacement


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_module_device_move():
    module = JLDModule(metric=tiny_metric()).cuda()
    ref, target = pairs()
    assert module.U.device.type == module.encoder.device.type == "cuda"
    actual = module(ref.cuda(), target.cuda())
    torch.testing.assert_close(actual.cpu(), tiny_metric()(ref, target), rtol=1e-5, atol=1e-5)

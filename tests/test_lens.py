"""The shipped lens, lens fitting, and the image utilities."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from conftest import EXAMPLES
from jld import JLD, Lens, ViTEncoder, fit_lens, load_image, load_pair
from jld.frontend import chroma_lowpass, to_ycbcr
from jld.images import center_crop_to_multiple
from jld.lens import list_images, subspace_overlap, top_eigenvectors
from jld.metric import PRETRAINED_LENS


def test_shipped_lens_loads_with_expected_shape():
    assert PRETRAINED_LENS.is_file()
    lens = Lens.load(PRETRAINED_LENS)
    assert lens.U.shape == (384, 64)
    assert lens.M.shape == (384, 384)
    assert lens.eigenvalues.shape == (384,)
    assert lens.k == 64
    assert lens.meta["encoder"] == "vit_small_patch14_dinov2.lvd142m"
    assert (lens.meta["layer"], lens.meta["k"], lens.meta["n_images"]) == (1, 64, 100)


def test_shipped_lens_is_the_top_eigenbasis_of_its_operator():
    lens = Lens.load(PRETRAINED_LENS)
    assert np.allclose(lens.U.T @ lens.U, np.eye(64), atol=1e-4)
    assert np.allclose(lens.M, lens.M.T, atol=1e-6)
    assert np.all(np.diff(lens.eigenvalues) <= 1e-6)  # descending
    values, vectors = top_eigenvectors(lens.M, 64)
    assert subspace_overlap(vectors, lens.U) == pytest.approx(1.0, abs=1e-3)
    assert np.allclose(values, lens.eigenvalues, rtol=1e-3, atol=1e-5 * float(values[0]))


def test_lens_round_trip(tmp_path):
    lens = Lens.load(PRETRAINED_LENS)
    path = lens.save(tmp_path / "copy")
    assert path.suffix == ".npz"
    again = Lens.load(path)
    assert np.array_equal(again.U, lens.U) and np.array_equal(again.M, lens.M)
    assert again.meta == lens.meta


@pytest.fixture(scope="module")
def encoder(metrics) -> ViTEncoder:
    return metrics["full"].encoder


def _fit(encoder, seed: int = 0) -> Lens:
    paths = list_images(EXAMPLES)[:2]
    return fit_lens(encoder, paths, k=8, n_crops=1, crop=112, n_probes=2, seed=seed, batch=2, verbose=False)


def test_fit_lens_on_two_images(encoder, tmp_path):
    lens = _fit(encoder)
    assert lens.U.shape == (384, 8) and lens.M.shape == (384, 384)
    assert np.allclose(lens.U.T @ lens.U, np.eye(8), atol=1e-4)
    assert np.allclose(lens.M, lens.M.T)
    assert lens.eigenvalues[0] > 0 and lens.eigenvalues.min() > -1e-4 * lens.eigenvalues[0]  # M is PSD
    assert lens.meta["n_images"] == 2 and lens.meta["layer"] == 1
    assert np.array_equal(_fit(encoder).M, lens.M)          # same seed, same lens
    assert not np.array_equal(_fit(encoder, seed=1).M, lens.M)

    refit = JLD.from_lens(lens.save(tmp_path / "tiny.npz"), device="cpu", cls_weight=0.0)
    assert refit.U.shape == (384, 8)
    assert float(refit(EXAMPLES / "bikes.png", EXAMPLES / "bikes_blur.png")) > 0


def test_fitting_leaves_the_encoder_frozen(encoder):
    _fit(encoder)
    assert not any(p.requires_grad for p in encoder.parameters())
    assert not encoder.training


def test_subspace_overlap():
    basis = np.linalg.qr(np.random.default_rng(0).normal(size=(16, 16)))[0]
    assert subspace_overlap(basis[:, :4], basis[:, :4]) == pytest.approx(1.0)
    assert subspace_overlap(basis[:, :4], basis[:, 4:8]) == pytest.approx(0.0, abs=1e-12)


def test_center_crop_and_pair_loading():
    x = torch.arange(3 * 30 * 45, dtype=torch.float32).reshape(1, 3, 30, 45)
    assert torch.equal(center_crop_to_multiple(x, 14), x[..., 1:29, 1:43])
    with pytest.raises(ValueError):
        center_crop_to_multiple(torch.zeros(1, 3, 13, 40), 14)
    ref, dist = load_pair(EXAMPLES / "parrots.png", EXAMPLES / "parrots_jpeg_q8.png")
    assert ref.shape == dist.shape == (1, 3, 294, 392)
    assert 0 <= float(ref.min()) and float(ref.max()) <= 1
    assert load_image(EXAMPLES / "parrots.png", patch=1).shape == (1, 3, 294, 392)


def test_chroma_prefilter_keeps_luminance():
    x = load_image(EXAMPLES / "parrots.png")
    y = chroma_lowpass(x, 2.0)
    assert y.shape == x.shape and not torch.equal(x, y)
    interior = (x > 0.05).all(1, keepdim=True) & (x < 0.95).all(1, keepdim=True) & (y > 0).all(1, keepdim=True) & (y < 1).all(1, keepdim=True)
    luma_change = (to_ycbcr(x)[:, :1] - to_ycbcr(y)[:, :1]).abs()[interior]
    assert float(luma_change.max()) < 5e-3  # only clamped pixels can change luminance
    gray = x.mean(1, keepdim=True).expand_as(x)
    assert torch.allclose(chroma_lowpass(gray, 2.0), gray, atol=2e-3)
    assert chroma_lowpass(x, 0.0) is x

"""Properties of the distance on the bundled examples and on random images."""
from __future__ import annotations

import io

import numpy as np
import pytest
import torch
from PIL import Image, ImageFilter

from conftest import EXAMPLES, EXPECTED, reference_of
from jld import JLD, VARIANTS, load_image

IMAGES = sorted(EXAMPLES.glob("*.png"))


@pytest.mark.parametrize("variant", sorted(EXPECTED))
def test_matches_expected_values(metrics, variant):
    for name, value in EXPECTED[variant].items():
        got = float(metrics[variant](reference_of(name), EXAMPLES / name))
        assert got == pytest.approx(value, rel=1e-3, abs=1e-4), name


@pytest.mark.parametrize("variant", sorted(EXPECTED))
def test_identical_images_score_zero(metrics, variant):
    for name in ("parrots.png", "hats_noise.png"):
        assert abs(float(metrics[variant](EXAMPLES / name, EXAMPLES / name))) < 1e-4


@pytest.mark.parametrize("variant", ["full", "fast"])
def test_symmetric(metrics, variant):
    ref, dist = EXAMPLES / "hats.png", EXAMPLES / "hats_noise.png"
    forward, backward = float(metrics[variant](ref, dist)), float(metrics[variant](dist, ref))
    assert forward > 0
    assert forward == pytest.approx(backward, rel=1e-4)


@pytest.mark.parametrize("variant", ["lens", "fast"])
def test_lens_term_triangle_inequality(metrics, variant):
    """The lens term is an L2 norm of projected feature differences, so it is a pseudometric."""
    lens_term = metrics[variant]
    rng = np.random.default_rng(0)
    generator = torch.Generator().manual_seed(0)
    triples = [[load_image(IMAGES[i]) for i in rng.choice(len(IMAGES), 3, replace=False)] for _ in range(8)]
    triples += [list(torch.rand(3, 1, 3, 70, 84, generator=generator)) for _ in range(8)]
    for a, b, c in triples:
        ab, bc, ac = (float(lens_term(x, y)) for x, y in ((a, b), (b, c), (a, c)))
        assert ac <= ab + bc + 1e-5
        assert ab <= ac + bc + 1e-5
        assert bc <= ab + ac + 1e-5


def test_mild_jpeg_closer_than_strong_jpeg(metric, fast):
    ref = EXAMPLES / "parrots.png"
    for m in (metric, fast):
        assert float(m(ref, EXAMPLES / "parrots_jpeg_q30.png")) < float(m(ref, EXAMPLES / "parrots_jpeg_q8.png"))


def _jpeg(image: Image.Image, quality: int) -> Image.Image:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    return Image.open(buffer).convert("RGB")


def _noise(image: Image.Image, sigma: float) -> Image.Image:
    array = np.asarray(image, dtype=np.float64)
    noisy = array + np.random.default_rng(0).normal(0.0, 1.0, array.shape) * sigma
    return Image.fromarray(np.clip(np.round(noisy), 0, 255).astype(np.uint8), "RGB")


LADDERS = {
    "blur": [lambda im, r=r: im.filter(ImageFilter.GaussianBlur(radius=r)) for r in (0.5, 1, 2, 4)],
    "noise": [lambda im, s=s: _noise(im, s) for s in (4, 8, 16, 32)],
    "jpeg": [lambda im, q=q: _jpeg(im, q) for q in (80, 40, 15, 5)],
}


@pytest.mark.parametrize("variant", ["full", "fast"])
@pytest.mark.parametrize("distortion", sorted(LADDERS))
@pytest.mark.parametrize("reference", ["parrots", "hats", "bikes"])
def test_monotonic_in_distortion_strength(metrics, variant, distortion, reference):
    image = Image.open(EXAMPLES / f"{reference}.png").convert("RGB")
    levels = [float(metrics[variant](image, apply(image))) for apply in LADDERS[distortion]]
    assert levels[0] > 0
    assert all(low < high for low, high in zip(levels, levels[1:])), levels


def test_input_types_agree(fast):
    ref, dist = EXAMPLES / "bikes.png", EXAMPLES / "bikes_blur.png"
    expected = float(fast(ref, dist))
    pil = (Image.open(ref), Image.open(dist))
    arrays = tuple(np.asarray(im.convert("RGB")) for im in pil)
    floats = tuple(a.astype(np.float32) / 255 for a in arrays)
    tensors = (load_image(ref)[0], load_image(dist))  # [3, H, W] against [1, 3, H, W]
    for a, b in (pil, arrays, floats, tensors, (str(ref), str(dist))):
        assert float(fast(a, b)) == pytest.approx(expected, rel=1e-4)


def test_batch_matches_single_pairs(metric):
    names = ["parrots_jpeg_q30.png", "parrots_jpeg_q8.png", "parrots_color_shift.png"]
    ref = load_image(EXAMPLES / "parrots.png")
    batched = metric(ref, torch.cat([load_image(EXAMPLES / n) for n in names]))
    singles = torch.stack([metric(ref, load_image(EXAMPLES / n))[0] for n in names])
    assert batched.shape == (len(names),)
    assert torch.allclose(batched, singles, rtol=1e-3, atol=1e-5)


def test_differentiable_path(metric):
    ref = load_image(EXAMPLES / "bikes.png")
    dist = load_image(EXAMPLES / "bikes_blur.png").requires_grad_(True)
    loss = metric.distance_tensor(ref, dist)
    loss.backward()
    assert dist.grad is not None and dist.grad.abs().sum() > 0
    assert float(loss.detach()) == pytest.approx(float(metric(ref, dist.detach())), rel=1e-4)
    assert not metric(ref, dist).requires_grad  # plain scoring builds no graph


def test_map_pools_to_the_fast_distance(fast):
    ref, dist = EXAMPLES / "parrots.png", EXAMPLES / "parrots_jpeg_q8.png"
    local = fast.map(ref, dist)
    assert local.shape == (1, 294 // 14, 392 // 14)
    assert (local >= 0).all()
    assert float(local.square().mean().sqrt()) == pytest.approx(float(fast(ref, dist)), rel=1e-4)


def test_full_is_fast_plus_weighted_cls_term(metric, fast):
    ref, dist = EXAMPLES / "hats.png", EXAMPLES / "hats_posterize.png"
    cls_term = (float(metric(ref, dist)) - float(fast(ref, dist))) / metric.cls_weight
    assert metric.cls_weight == 0.5 and fast.cls_weight == 0.0
    assert 0 < cls_term <= 2  # one minus a cosine


def test_fast_stops_after_block_one(fast, monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("JLD-fast ran a block after block 1")

    monkeypatch.setattr(fast.encoder.model.blocks[1], "forward", fail)
    assert float(fast(EXAMPLES / "bikes.png", EXAMPLES / "bikes_blur.png")) > 0
    hidden, cls = fast.encoder(load_image(EXAMPLES / "bikes.png"), need_cls=False)
    assert cls is None and hidden.shape == (1, 1 + 21 * 28, 384)


def test_crops_to_the_patch_grid(fast):
    generator = torch.Generator().manual_seed(1)
    ref, dist = torch.rand(2, 1, 3, 75, 90, generator=generator)
    cropped = float(fast(ref[..., 2:72, 3:87], dist[..., 2:72, 3:87]))  # centre 70 x 84
    assert float(fast(ref, dist)) == pytest.approx(cropped, rel=1e-5)
    assert fast.map(ref, dist).shape == (1, 5, 6)


def test_rejects_bad_inputs(fast):
    small = torch.rand(1, 3, 10, 10)
    with pytest.raises(ValueError):
        fast(small, small)
    with pytest.raises(ValueError):
        fast(torch.rand(1, 3, 56, 56), torch.rand(1, 3, 56, 70))
    with pytest.raises(ValueError):
        fast(torch.rand(2, 3, 56, 56), torch.rand(3, 3, 56, 56))
    with pytest.raises(TypeError):
        fast([1, 2, 3], [1, 2, 3])
    with pytest.raises(ValueError):
        JLD.pretrained("unknown")


def test_variants_table():
    assert set(VARIANTS) == {"full", "fast", "lens", "raw"}
    assert VARIANTS["full"] == (True, 2.0, 0.5)
    assert VARIANTS["fast"] == (True, 2.0, 0.0)

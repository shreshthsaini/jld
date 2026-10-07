"""Public-demo input checks and native-resolution rendering behavior."""
from pathlib import Path

import gradio as gr
import numpy as np
from PIL import Image
import pytest

import app


def image_file(tmp_path: Path, size: tuple[int, int], name: str = "image.png") -> str:
    path = tmp_path / name
    Image.new("RGB", size, color=(60, 80, 100)).save(path)
    return str(path)


def test_missing_upload():
    with pytest.raises(gr.Error, match="Choose both"):
        app.load_image(None)


@pytest.mark.parametrize("size", [(13, 100), (100, 13), (8193, 14)])
def test_image_resource_limits(tmp_path, size):
    with pytest.raises(gr.Error):
        app.load_image(image_file(tmp_path, size))


def test_upload_pixel_limit_checked_before_decoding(tmp_path, monkeypatch):
    monkeypatch.setattr(app, "MAX_UPLOAD_PIXELS", 100)
    with pytest.raises(gr.Error, match="24 megapixels"):
        app.load_image(image_file(tmp_path, (14, 14)))


def test_gradio_preprocessing_accepts_large_custom_upload(tmp_path):
    component = gr.Image(type="filepath")
    payload = component.data_model(path=image_file(tmp_path, (1400, 1120)))
    path = component.preprocess(payload)
    assert app.load_image(path).size == (1400, 1120)


def test_large_aligned_pair_reduced_with_identical_filter():
    image = Image.new("RGB", (1400, 1120), color=(60, 80, 100))
    reference, distorted = app.prepare_pair(image, image.copy(), True)
    assert reference.size == distorted.size
    assert reference.width * reference.height <= app.MAX_PIXELS
    assert max(reference.size) <= app.MAX_SIDE
    assert abs(reference.width / reference.height - 1400 / 1120) < 0.01
    assert np.array_equal(np.asarray(reference), np.asarray(distorted))


def test_native_scoring_rejects_large_pair_before_model_load(tmp_path, monkeypatch):
    def unexpected_score(*args):
        raise AssertionError("Do not score a pair that exceeds the hosted resolution limit")
    monkeypatch.setattr(app, "compute_distance_and_map", unexpected_score)
    path = image_file(tmp_path, (1400, 1120))
    with pytest.raises(gr.Error, match="Enable.*Resize large pairs"):
        app.score_pair(path, path, "fast", False)


def test_small_pair_keeps_original_pixels():
    image = Image.new("RGB", (31, 43))
    reference, distorted = app.prepare_pair(image, image, True)
    assert reference is image and distorted is image


def test_resized_result_reports_actual_scoring_grid(tmp_path, monkeypatch):
    def score(reference, distorted, variant):
        assert reference.size == distorted.size
        return 0.25, np.zeros((reference.height // app.PATCH, reference.width // app.PATCH))
    monkeypatch.setattr(app, "compute_distance_and_map", score)
    path = image_file(tmp_path, (1400, 1120))
    value, overlay, details = app.score_pair(path, path, "fast", True)
    assert value == 0.25
    assert "Demo resize" in details and "1400 × 1120" in details
    assert "resized pair" in details
    assert overlay.width * overlay.height <= app.MAX_PIXELS


def test_mismatched_native_sizes_rejected_before_metric_load(tmp_path, monkeypatch):
    def unexpected_model():
        raise AssertionError("Do not load the encoder for an invalid pair")
    monkeypatch.setattr(app, "metrics", unexpected_model)
    with pytest.raises(gr.Error, match="same size"):
        app.score_pair(image_file(tmp_path, (28, 28), "ref.png"),
                       image_file(tmp_path, (29, 28), "dist.png"), "fast")


def test_overlay_uses_native_center_crop():
    reference = Image.new("RGB", (31, 43))
    overlay = app.map_overlay(reference, np.zeros((3, 2), dtype=np.float32))
    assert overlay.size == (28, 42)


def test_supported_format_conversion_retains_dimensions(tmp_path):
    path = tmp_path / "gray.png"
    Image.new("L", (31, 43), color=80).save(path)
    image = app.load_image(str(path))
    assert image.mode == "RGB"
    assert image.size == (31, 43)

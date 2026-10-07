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


@pytest.mark.parametrize("size", [(13, 100), (100, 13), (513, 513), (1025, 14)])
def test_image_resource_limits(tmp_path, size):
    with pytest.raises(gr.Error):
        app.load_image(image_file(tmp_path, size))


def test_mismatched_native_sizes_rejected_before_metric_load(tmp_path, monkeypatch):
    def unexpected_model():
        raise AssertionError("Do not load the encoder for an invalid pair")
    monkeypatch.setattr(app, "metrics", unexpected_model)
    with pytest.raises(gr.Error, match="same native size"):
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

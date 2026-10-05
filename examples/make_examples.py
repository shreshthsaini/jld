#!/usr/bin/env python3
"""Regenerate the example pairs in this folder.

References are three images from the Kodak Lossless True Color Image Suite,
hosted by Rich Franzen; consult the source site for reuse terms: kodim23 (parrots), kodim03
(hats) and kodim05 (bikes). Each is resized to 294 px height and centre-cropped
to 392 x 294. Distorted versions are named <reference>_<distortion>.png and are
generated with the fixed parameters below. Files are PNG without metadata.
"""
from __future__ import annotations

import io
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

HERE = Path(__file__).resolve().parent
URL = "https://r0k.us/graphics/kodak/kodak/{}.png"
WIDTH, HEIGHT = 392, 294
SOURCES = {"parrots": "kodim23", "hats": "kodim03", "bikes": "kodim05"}


def jpeg(image: Image.Image, quality: int) -> Image.Image:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    return Image.open(buffer).convert("RGB")


def from_array(array: np.ndarray) -> Image.Image:
    return Image.fromarray(np.clip(np.round(array), 0, 255).astype(np.uint8), "RGB")


def noise(image: Image.Image, sigma: float) -> Image.Image:
    rng = np.random.default_rng(0)
    array = np.asarray(image, dtype=np.float64)
    return from_array(array + rng.normal(0.0, sigma, array.shape))


def resample(image: Image.Image, factor: int) -> Image.Image:
    small = image.resize((WIDTH // factor, round(HEIGHT / factor)), Image.BICUBIC)
    return small.resize((WIDTH, HEIGHT), Image.BICUBIC)


DISTORTIONS = {
    "parrots": {
        "jpeg_q30": lambda im: jpeg(im, 30),
        "jpeg_q8": lambda im: jpeg(im, 8),
        "color_shift": lambda im: from_array(np.asarray(im, np.float64) * [1.10, 1.00, 0.85]),
    },
    "hats": {
        "noise": lambda im: noise(im, 12.0),
        "contrast": lambda im: ImageEnhance.Contrast(im).enhance(0.6),
        "posterize": lambda im: ImageOps.posterize(im, 3),
    },
    "bikes": {
        "blur": lambda im: im.filter(ImageFilter.GaussianBlur(radius=2)),
        "downsample_x4": lambda im: resample(im, 4),
    },
}


def load_reference(kodak_name: str) -> Image.Image:
    with urllib.request.urlopen(URL.format(kodak_name)) as response:
        image = Image.open(io.BytesIO(response.read())).convert("RGB")
    width = round(image.width * HEIGHT / image.height)
    image = image.resize((width, HEIGHT), Image.LANCZOS)
    left = (width - WIDTH) // 2
    return image.crop((left, 0, left + WIDTH, HEIGHT))


def main() -> None:
    count = 0
    for name, kodak_name in SOURCES.items():
        reference = load_reference(kodak_name)
        reference.save(HERE / f"{name}.png")
        for distortion, apply in DISTORTIONS[name].items():
            apply(reference).convert("RGB").save(HERE / f"{name}_{distortion}.png")
            count += 1
    print(f"wrote {len(SOURCES)} references and {count} distorted images ({WIDTH}x{HEIGHT}) to {HERE}")


if __name__ == "__main__":
    main()

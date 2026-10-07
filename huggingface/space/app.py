"""An author-maintained JLD Space, using the public package without score surrogates."""
from __future__ import annotations

from functools import lru_cache
import math
import os
from pathlib import Path
import threading
import warnings

# Import before torch so the hosted ZeroGPU runtime can enable CUDA emulation.
import spaces
import gradio as gr
import numpy as np
from PIL import Image
import torch

from jld import JLD
from jld.cli import RAMP
from jld.lens import Lens
from jld.metric import PRETRAINED_LENS

HERE = Path(__file__).resolve().parent
EXAMPLES = HERE / "examples"
MAX_PIXELS = 262_144
MAX_SIDE = 1024
MAX_UPLOAD_PIXELS = 24_000_000
MAX_UPLOAD_SIDE = 8192
MAX_FILE_BYTES = 10 * 1024 * 1024
PATCH = 14
# Gradio opens images before load_image runs. Its decoding limit must allow
# uploads larger than the bounded scoring resolution.
Image.MAX_IMAGE_PIXELS = MAX_UPLOAD_PIXELS
warnings.filterwarnings("error", category=Image.DecompressionBombWarning)
torch.set_num_threads(min(4, max(1, int(os.environ.get("JLD_CPU_THREADS", "2")))))
MODEL_LOCK = threading.Lock()
HOSTED = bool(os.environ.get("SPACE_ID"))

PAPER = "https://arxiv.org/abs/2610.05967"
CODE = "https://github.com/shreshthsaini/jld"
WEBSITE = "https://shreshthsaini.github.io/jld/"
INSTALL = (
    'uv add "jacobian-lens-distance @ '
    'https://github.com/shreshthsaini/jld/releases/download/v1.1.0/'
    'jacobian_lens_distance-1.1.0-py3-none-any.whl"'
)
BIBTEX = """@article{saini2026jld,
  title={JLD: Perceptual Distance Through A Jacobian Lens},
  author={Saini, Shreshth and Adsumilli, Balu and Bovik, Alan C.},
  journal={arXiv preprint arXiv:2610.05967},
  year={2026}
}"""


def load_image(path: str | None) -> Image.Image:
    """Check resource limits before pixel decoding, then retain native RGB pixels."""
    if not path:
        raise gr.Error("Choose both a reference image and a distorted image.")
    try:
        if Path(path).stat().st_size > MAX_FILE_BYTES:
            raise gr.Error("Each image must be at most 10 MB. Use the local package for larger files.")
        with Image.open(path) as image:
            width, height = image.size
            if width < PATCH or height < PATCH:
                raise gr.Error("Each image must be at least 14 pixels wide and high.")
            if width * height > MAX_UPLOAD_PIXELS or max(width, height) > MAX_UPLOAD_SIDE:
                raise gr.Error(
                    "This demo accepts uploads up to 24 megapixels and a longest side of 8192 pixels. "
                    "Use the local package to score larger images at their native resolution."
                )
            if getattr(image, "n_frames", 1) > 1:
                raise gr.Error("Please upload a single still image.")
            return image.convert("RGB")
    except gr.Error:
        raise
    except (Image.DecompressionBombWarning, Image.DecompressionBombError):
        raise gr.Error("Upload exceeds the 24-megapixel image limit.") from None
    except (OSError, ValueError):
        raise gr.Error("Could not read the image. Please use a PNG, JPEG, or WebP image.") from None


@lru_cache(maxsize=1)
def metrics() -> dict[str, JLD]:
    """Share one frozen encoder between variants to bound resident memory."""
    device = "cuda" if HOSTED else os.environ.get("JLD_DEVICE", "cpu")
    full = JLD.pretrained("full", device=device)
    # Read the shipped lens on CPU without copying a fake-CUDA tensor back to CPU.
    fast = JLD(full.encoder, Lens.load(PRETRAINED_LENS).U, full.chroma_sigma, 0.0)
    return {"full": full, "fast": fast}


if HOSTED:
    # ZeroGPU optimizes models placed on emulated CUDA at module startup.
    metrics()


@spaces.GPU(duration=15)
def compute_distance_and_map(reference: Image.Image, distorted: Image.Image, variant: str):
    models = metrics()
    value = float(models[variant](reference, distorted)[0])
    # Maps use the same lens and chroma prefilter; they never include the CLS term.
    patch_map = models["fast"].map(reference, distorted)[0].numpy()
    return value, patch_map


def map_overlay(reference: Image.Image, patch_map: np.ndarray) -> Image.Image:
    """Use the public CLI's colour ramp and per-image maximum normalization."""
    height, width = patch_map.shape[0] * PATCH, patch_map.shape[1] * PATCH
    left, top = (reference.width - width) // 2, (reference.height - height) // 2
    cropped = reference.crop((left, top, left + width, top + height))
    scaled = patch_map / max(float(patch_map.max()), 1e-12)
    positions = np.linspace(0, 1, len(RAMP))
    colors = np.stack([np.interp(scaled, positions, RAMP[:, c]) for c in range(3)], axis=-1)
    heat = Image.fromarray(colors.astype(np.uint8)).resize((width, height), Image.Resampling.NEAREST)
    gray = Image.fromarray(np.asarray(cropped).mean(axis=-1).round().astype(np.uint8)).convert("RGB")
    return Image.blend(gray, heat, 0.6)


def prepare_pair(reference: Image.Image, distorted: Image.Image, resize_large: bool):
    """Apply the same explicit demo reduction to an aligned image pair."""
    if reference.size != distorted.size:
        raise gr.Error(
            f"Images must have exactly the same size: reference {reference.size}, "
            f"distorted {distorted.size}. Upload an aligned pair; resizing does not align different images."
        )
    width, height = reference.size
    scale = min(1.0, math.sqrt(MAX_PIXELS / (width * height)), MAX_SIDE / width, MAX_SIDE / height)
    if scale == 1.0:
        return reference, distorted
    if not resize_large:
        raise gr.Error(
            "This pair is too large for native-resolution scoring in the hosted demo. "
            "Enable 'Resize large pairs for this demo' or use the local package at native resolution."
        )
    size = (int(width * scale), int(height * scale))
    if min(size) < PATCH:
        raise gr.Error("This aspect ratio is too narrow for the demo. Crop both images to the same region.")
    return (reference.resize(size, Image.Resampling.LANCZOS),
            distorted.resize(size, Image.Resampling.LANCZOS))


def score_pair(reference_path: str | None, distorted_path: str | None, variant: str,
               resize_large: bool = True):
    if variant not in ("full", "fast"):
        raise gr.Error("Choose full or fast.")
    reference, distorted = load_image(reference_path), load_image(distorted_path)
    native_size = reference.size
    reference, distorted = prepare_pair(reference, distorted, resize_large)
    # Serialize calls even through direct API usage, including first model initialization.
    with MODEL_LOCK:
        value, patch_map = compute_distance_and_map(reference, distorted, variant)
    height, width = patch_map.shape[0] * PATCH, patch_map.shape[1] * PATCH
    name = "JLD" if variant == "full" else "JLD-fast"
    if reference.size == native_size:
        resolution = f"Native image size: {native_size[0]} × {native_size[1]}; no resizing."
    else:
        resolution = (
            f"**Demo resize:** both images reduced from {native_size[0]} × {native_size[1]} "
            f"to {reference.width} × {reference.height} with the same Lanczos filter. "
            "This distance is measured on the resized pair. Use the local package for native-resolution scores."
        )
    details = (
        f"**{name}: {value:.6f}**. Lower means more similar to the reference. "
        f"Scored RGB crop: **{width} × {height} pixels**. {resolution}\n\n"
        "The overlay shows **local projected patch displacement**, after the chroma prefilter. "
        "It excludes the global CLS term used by full JLD. Brighter colours mark larger local "
        "displacements. Colours are normalized to this pair's maximum, so do not compare colour "
        "intensity across different pairs. An identical pair has zero displacement."
    )
    return value, map_overlay(reference, patch_map), details


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="JLD: Jacobian Lens Distance", analytics_enabled=False) as demo:
        gr.Markdown(
            "# JLD: Perceptual Distance Through a Jacobian Lens\n"
            "Compare a reference image with its distorted version using our frozen DINOv2 lens. "
            "Try a release example below, or upload your own pair.\n\n"
            f"[Read the paper]({PAPER}) · [Code and installation]({CODE}) · [Research website]({WEBSITE})"
        )
        with gr.Row():
            reference = gr.Image(value=str(EXAMPLES / "parrots.png"), label="Reference image",
                                 type="filepath", sources=["upload"], height=280, interactive=True)
            distorted = gr.Image(value=str(EXAMPLES / "parrots_jpeg_q8.png"), label="Distorted image",
                                 type="filepath", sources=["upload"], height=280, interactive=True)
        variant = gr.Radio(
            choices=[("JLD: lens + global CLS term", "full"), ("JLD-fast: lens term only", "fast")],
            value="fast", label="Variant",
        )
        gr.Markdown(
            "Upload **aligned images of the same size**, up to 24 megapixels, 8192 pixels on the longest "
            "side, and 10 MB each. Large pairs can be resized together for this demo; the result shows "
            "the actual scoring resolution. Smaller pairs keep their native resolution. "
            "Scoring converts to RGB and center-crops to multiples of 14 pixels. "
            "The hosted demo uses free shared GPU access; queueing and daily usage limits may apply."
        )
        resize_large = gr.Checkbox(value=True, label="Resize large pairs for this demo",
                                  info="Uses the same scale for both images, up to 262,144 pixels and a longest side of 1024.")
        run = gr.Button("Compute distance and local map", variant="primary")
        with gr.Row():
            distance = gr.Number(value=None, label="Distance (lower means more similar)", precision=6,
                                 placeholder="Compute a pair to see its distance", interactive=False)
            response = gr.Image(label="Local lens response overlay (excludes CLS)", type="pil", height=300)
        explanation = gr.Markdown()
        run.click(score_pair, [reference, distorted, variant, resize_large], [distance, response, explanation],
                  api_name="score_pair", concurrency_limit=1, concurrency_id="jld")
        gr.Examples(
            examples=[
                [str(EXAMPLES / "parrots.png"), str(EXAMPLES / "parrots_jpeg_q8.png"), "fast"],
                [str(EXAMPLES / "parrots.png"), str(EXAMPLES / "parrots_jpeg_q30.png"), "full"],
                [str(EXAMPLES / "bikes.png"), str(EXAMPLES / "bikes_blur.png"), "fast"],
                [str(EXAMPLES / "hats.png"), str(EXAMPLES / "hats_noise.png"), "full"],
                [str(EXAMPLES / "parrots.png"), str(EXAMPLES / "parrots.png"), "fast"],
            ], inputs=[reference, distorted, variant], label="Authentic release examples",
        )
        gr.Markdown(
            "Examples come from the public JLD release: Kodak reference photographs hosted by "
            "[Rich Franzen](https://r0k.us/graphics/kodak/), with fixed JPEG, blur, and noise transformations. "
            "They are demonstrations, not a benchmark evaluation. See [example provenance](https://github.com/"
            "shreshthsaini/jld/blob/main/huggingface/space/examples/PROVENANCE.md)."
        )
        with gr.Accordion("Use JLD in your research", open=False):
            gr.Markdown(
                "Install the versioned release in your project with uv:\n\n"
                f"```bash\n{INSTALL}\n```\n\n"
                "```python\nfrom jld import JLD\nmetric = JLD.pretrained('full')\n"
                "distance = metric('reference.png', 'distorted.png')\n```\n\n"
                "If you use JLD, please cite the paper."
            )
            gr.Code(BIBTEX, language=None, label="BibTeX citation")
    return demo.queue(max_size=8, default_concurrency_limit=1)


demo = build_demo()

if __name__ == "__main__":
    if os.environ.get("SPACE_ID"):
        # Hugging Face supplies its managed runtime launch configuration.
        demo.launch(max_file_size="10mb", show_error=False)
    else:
        demo.launch(server_name="127.0.0.1", max_file_size="10mb", show_error=False)

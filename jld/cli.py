"""Command-line scoring: ``jld REFERENCE DISTORTED`` or ``python -m jld REFERENCE DISTORTED``.

    jld examples/parrots.png examples/parrots_jpeg_q8.png
    jld examples/parrots.png examples/parrots_jpeg_q8.png --variant fast
    jld examples/bikes.png examples/bikes_blur.png --map blur_map.png

Prints the distance (lower means more similar). ``--map`` also writes the local
map, the per-patch projected displacement, as an overlay on the reference.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from jld.images import load_pair
from jld.metric import JLD, VARIANTS

# Piecewise-linear dark-to-bright colour ramp for the map.
RAMP = np.array([[0, 0, 4], [87, 16, 110], [188, 55, 84], [249, 142, 9], [252, 255, 164]], float)


def save_map(metric: JLD, ref: str | Path, dist: str | Path, out: str | Path, blend: float = 0.6) -> np.ndarray:
    """Write the local map of a pair as a PNG overlay and return the [H/p, W/p] array.

    The map is scaled to its own maximum, so colours are comparable within one
    image but not across images. ``blend=1`` writes the map without the image.
    """
    patch_map = metric.map(ref, dist)[0].numpy()
    x1, _ = load_pair(ref, dist, metric.encoder.patch)
    h, w = x1.shape[-2:]
    scaled = patch_map / max(float(patch_map.max()), 1e-12)
    colors = np.stack([np.interp(scaled, np.linspace(0, 1, len(RAMP)), RAMP[:, c]) for c in range(3)], -1)
    heat = Image.fromarray(colors.astype(np.uint8)).resize((w, h), Image.NEAREST)
    gray = Image.fromarray((x1[0].mean(0).numpy() * 255).round().astype(np.uint8)).convert("RGB")
    Image.blend(gray, heat, blend).save(out)
    return patch_map


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="jld", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("reference", help="reference image")
    parser.add_argument("distorted", help="distorted image of the same size")
    parser.add_argument("--variant", default="full", choices=sorted(VARIANTS),
                        help="full = JLD (default), fast = JLD-fast; lens and raw are ablations")
    parser.add_argument("--lens", type=Path, help="lens .npz written by scripts/fit_lens.py (default: the shipped lens)")
    parser.add_argument("--map", type=Path, dest="map_path", metavar="PNG", help="write the local map overlay to this PNG")
    parser.add_argument("--device", default=None, help="cuda or cpu (default: cuda if available)")
    args = parser.parse_args(argv)

    metric = JLD.pretrained(args.variant, device=args.device, lens_path=args.lens)
    print(f"JLD ({args.variant}) = {float(metric(args.reference, args.distorted)):.4f}")
    if args.map_path:
        patch_map = save_map(metric, args.reference, args.distorted, args.map_path)
        print(f"local map ({patch_map.shape[0]}x{patch_map.shape[1]} patches) written to {args.map_path}")


if __name__ == "__main__":
    main()

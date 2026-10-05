#!/usr/bin/env python3
"""Fit a Jacobian lens for a frozen ViT from a folder of unlabelled images.

Paper lens (DINOv2-S/14, block 1, k = 64) from the 100 DIV2K validation images:

    python scripts/fit_lens.py --div2k data/div2k --out lens/refit.npz --compare shipped

Any image folder works (the first --n-images files in sorted order are used):

    python scripts/fit_lens.py --images path/to/images --out lens/mine.npz

To adopt JLD for another encoder, pass any timm ViT name and a block index:

    python scripts/fit_lens.py --images path/to/images --encoder vit_base_patch14_dinov2.lvd142m \
        --layer 1 --out lens/dinov2_b14_block1_k64.npz

and then score with ``JLD.from_lens("lens/dinov2_b14_block1_k64.npz")`` or
``jld REF DIST --lens lens/mine.npz``. The fit needs no human labels: it uses
only the encoder's own gradients on the unlabelled images. ``--compare shipped``
compares the result with the lens bundled in the package.
"""
from __future__ import annotations

import argparse
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import torch

from jld.encoder import DEFAULT_ENCODER, ViTEncoder
from jld.lens import Lens, fit_lens, list_images, subspace_overlap
from jld.metric import PRETRAINED_LENS

DIV2K_URL = "https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip"


def download_div2k(root: Path) -> Path:
    """Download and unpack the 100 DIV2K validation HR images below ``root``."""
    folder = root / "DIV2K_valid_HR"
    if len(list_images(folder) if folder.is_dir() else []) >= 100:
        return folder
    root.mkdir(parents=True, exist_ok=True)
    archive = root / "DIV2K_valid_HR.zip"
    if not archive.is_file():
        print(f"downloading {DIV2K_URL}", flush=True)
        partial = archive.with_suffix(".part")
        urllib.request.urlretrieve(DIV2K_URL, partial)
        partial.replace(archive)
    with zipfile.ZipFile(archive) as handle:
        handle.extractall(root)
    return folder


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--images", type=Path, help="folder of unlabelled images")
    source.add_argument("--div2k", type=Path, help="download DIV2K validation HR into this folder and use it")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--encoder", default=DEFAULT_ENCODER, help="timm model name")
    parser.add_argument("--layer", type=int, default=1, help="block whose output tokens are measured")
    parser.add_argument("--k", type=int, default=64)
    parser.add_argument("--n-images", type=int, default=100)
    parser.add_argument("--n-crops", type=int, default=4)
    parser.add_argument("--crop", type=int, default=224)
    parser.add_argument("--n-probes", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default=None)
    parser.add_argument("--compare", type=Path,
                        help="report subspace overlap with an existing lens file, or 'shipped' for the bundled lens")
    args = parser.parse_args()

    folder = download_div2k(args.div2k) if args.div2k else args.images
    paths = list_images(folder)[: args.n_images]
    if len(paths) < args.n_images:
        print(f"warning: found {len(paths)} images in {folder}, fewer than --n-images {args.n_images}")
    encoder = ViTEncoder(args.encoder, args.layer, args.device)
    started = time.perf_counter()
    lens = fit_lens(encoder, paths, args.k, args.n_crops, args.crop, args.n_probes, args.seed, args.batch)
    if encoder.device.type == "cuda":
        torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    lens.meta["fit_seconds"] = round(seconds, 1)
    out = lens.save(args.out)
    share = float(lens.eigenvalues[: args.k].sum() / lens.eigenvalues.clip(min=0).sum())
    print(f"lens: {args.encoder} block {args.layer}, k = {args.k}, {len(paths)} images, {seconds:.1f} s")
    print(f"top-{args.k} share of trace(M): {share:.4f}")
    print(f"saved {out}")
    if args.compare:
        if str(args.compare) == "shipped":
            args.compare = PRETRAINED_LENS
        other = Lens.load(args.compare)
        overlap = subspace_overlap(lens.U, other.U)
        rel = float(np.linalg.norm(lens.M - other.M) / np.linalg.norm(other.M))
        print(f"subspace overlap with {args.compare.name}: {overlap:.4f} (1 = identical); "
              f"relative difference of M: {rel:.2e}")


if __name__ == "__main__":
    main()

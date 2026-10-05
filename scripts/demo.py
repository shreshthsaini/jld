#!/usr/bin/env python3
"""Score the example pairs in examples/ and rank them by JLD.

    python scripts/demo.py                  # JLD
    python scripts/demo.py --variant fast   # JLD-fast
    python scripts/demo.py --maps out/      # also write one local map per pair

In examples/, a file named <reference>_<distortion>.png is a distorted version of
<reference>.png. Lower JLD means more similar to the reference. To score your
own pair, use the ``jld`` command (see ``jld --help``).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from jld import JLD, VARIANTS, load_pair
from jld.cli import save_map

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def example_pairs(folder: Path) -> list[tuple[Path, Path]]:
    pairs = []
    for path in sorted(folder.glob("*_*.png")):
        reference = folder / (path.stem.split("_", 1)[0] + ".png")
        if reference.is_file():
            pairs.append((reference, path))
    return pairs


def psnr(ref: Path, dist: Path) -> float:
    x1, x2 = load_pair(ref, dist, patch=1)
    mse = float(((x1 - x2) ** 2).mean())
    return 10 * np.log10(1.0 / max(mse, 1e-12))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--examples", type=Path, default=EXAMPLES, help="folder of example pairs")
    parser.add_argument("--variant", default="full", choices=sorted(VARIANTS))
    parser.add_argument("--maps", type=Path, help="write a local map overlay per pair into this folder")
    parser.add_argument("--device", default=None, help="cuda or cpu (default: cuda if available)")
    args = parser.parse_args()

    pairs = example_pairs(args.examples)
    if not pairs:
        parser.error(f"no <reference>_<distortion>.png pairs in {args.examples}")
    metric = JLD.pretrained(args.variant, device=args.device)
    rows = [(float(metric(ref, dist)), psnr(ref, dist), ref.stem, dist.stem.split("_", 1)[1])
            for ref, dist in pairs]
    print(f"JLD ({args.variant}) on {len(rows)} example pairs, most similar first\n")
    print(f"{'rank':>4}  {'reference':<10} {'distortion':<14} {'JLD':>7} {'PSNR (dB)':>10}")
    for rank, (value, db, ref, distortion) in enumerate(sorted(rows), start=1):
        print(f"{rank:>4}  {ref:<10} {distortion:<14} {value:7.4f} {db:10.2f}")
    if args.maps:
        args.maps.mkdir(parents=True, exist_ok=True)
        for ref, dist in pairs:
            save_map(metric, ref, dist, args.maps / f"{dist.stem}_map.png")
        print(f"\n{len(pairs)} local maps written to {args.maps}")


if __name__ == "__main__":
    main()

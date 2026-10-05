#!/usr/bin/env python3
"""SRCC / PLCC / KRCC of JLD and baselines on full-reference IQA datasets.

Reproduces the JLD and JLD-fast rows of the paper's main table:

    python scripts/evaluate.py --data-root data

Needs pandas and scipy (``pip install -e ".[eval]"``); the optional baselines
(``--metrics psnr ssim lpips lpips_vgg dists``) also need pyiqa.

Datasets. Each dataset is looked up below ``--data-root``; if it is not there it
is downloaded from the Hugging Face mirrors that pyiqa uses (pass ``--offline``
to forbid downloads). The expected layout, which is what the archives unpack to:

    data/meta/meta_info_<Name>Dataset.csv      pyiqa metainfo (pair list and scores)
    data/tid2013/**/reference_images/I01.BMP   data/tid2013/**/distorted_images/i01_01_1.bmp
    data/csiq/**/src_imgs/1600.png             data/csiq/**/dst_imgs/1600.AWGN.1.png
    data/live/**/refimgs/bikes.bmp             data/live/**/jp2k/img1.bmp (and the other four folders)
    data/kadid10k/**/I01.png                   data/kadid10k/**/I01_01_01.png
    data/pipal/**/Val_Ref/A0005.bmp            data/pipal/**/Val_Dist/A0005_10_00.bmp

Folders are matched by suffix, so any nesting below ``data/<name>/`` works.

Protocol, as in the paper:
  * JLD variants: native resolution, both images centre-cropped to multiples of 14.
  * Baselines (pyiqa defaults): native resolution without cropping; DISTS gets both
    images resized so the shorter side is 256 (bicubic, antialiased).
  * kadid10k_test: the 65 KADID-10k references listed in splits/kadid10k_split.json
    (the other 16 were used for development). pipal_val: the 1,000 pairs of the
    official PIPAL validation split.
Correlations are absolute values, so distances and similarity scores are comparable.
PLCC is computed after the standard 5-parameter logistic mapping.

Per-pair scores are written to ``--out/<dataset>_scores.csv`` and reused on the
next run, so adding a metric does not recompute the others.
"""
from __future__ import annotations

import argparse
import json
import tarfile
import time
import warnings
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.optimize import curve_fit
from scipy.stats import kendalltau, pearsonr, spearmanr

from jld import JLD, VARIANTS, ViTEncoder, load_pair
from jld.lens import Lens
from jld.metric import PRETRAINED_LENS

ROOT = Path(__file__).resolve().parents[1]
PYIQA_DATA = "chaofengc/IQA-PyTorch-Datasets"
PYIQA_META = "chaofengc/IQA-PyTorch-Datasets-metainfo"


class Spec(NamedTuple):
    archive: str        # file in the pyiqa dataset mirror
    meta: str           # pyiqa metainfo CSV
    ref_column: str
    ref_dir: str        # prefix added to the reference names of the metainfo
    dist_dir: str
    score: str          # metainfo column with the human score
    rows: int           # pairs in the evaluated split
    members: tuple[str, ...] = ()   # archive folders to unpack (default: everything)


DATASETS = {
    "tid2013": Spec("tid2013.tgz", "meta_info_TID2013Dataset.csv", "ref_name", "reference_images/", "distorted_images/", "mos", 3000),
    "csiq": Spec("csiq.tgz", "meta_info_CSIQDataset.csv", "ref_name", "src_imgs/", "dst_imgs/", "dmos", 866),
    "live": Spec("live.tgz", "meta_info_LIVEIQADataset.csv", "ref_name", "", "", "mos", 779),
    "kadid10k": Spec("kadid10k.tgz", "meta_info_KADID10kDataset.csv", "ref_name", "", "", "dmos", 10125),
    "pipal": Spec("pipal.tar", "meta_info_PIPALDataset.csv", "hq_name", "Val_Ref/", "Val_Dist/", "elo_score", 1000,
                  ("Val_Ref", "Val_Dist")),
}
# evaluation name: (dataset on disk, subset)
SPLITS = {
    "tid2013": ("tid2013", None),
    "csiq": ("csiq", None),
    "live": ("live", None),
    "kadid10k": ("kadid10k", None),
    "kadid10k_test": ("kadid10k", "test"),
    "kadid10k_dev": ("kadid10k", "dev"),
    "pipal_val": ("pipal", "val"),
}
MAIN_TABLE = ["tid2013", "csiq", "live", "kadid10k_test", "pipal_val"]
MEAN_OVER = ["tid2013", "csiq", "live", "kadid10k_test"]   # the paper's "Mean, first four"
# SRCC reported in the paper's main table, for the comparison printed at the end.
PAPER_SRCC = {
    "jld": {"tid2013": 0.877, "csiq": 0.971, "live": 0.964, "kadid10k_test": 0.892, "pipal_val": 0.624},
    "jld_fast": {"tid2013": 0.875, "csiq": 0.956, "live": 0.945, "kadid10k_test": 0.870, "pipal_val": 0.578},
}
JLD_METRICS = {"jld": "full", "jld_fast": "fast", "jld_lens": "lens", "jld_raw": "raw"}
PYIQA_METRICS = {"psnr": "psnr", "ssim": "ssim", "lpips": "lpips", "lpips_vgg": "lpips-vgg", "dists": "dists"}


# ------------------------------------------------------------------ datasets
def _locate(root: Path, relative: str) -> Path | None:
    """The folder below ``root`` that contains ``relative`` (case-insensitive suffix match), or None."""
    if not root.is_dir():
        return None
    suffix = relative.lower()
    matches = sorted(p for p in root.rglob(Path(relative).name) if p.as_posix().lower().endswith(suffix))
    if not matches:
        return None
    base = matches[0]
    for _ in Path(relative).parts:
        base = base.parent
    return base


def _meta(spec: Spec, data_root: Path, offline: bool) -> pd.DataFrame:
    path = data_root / "meta" / spec.meta
    if not path.is_file():
        if offline:
            raise FileNotFoundError(f"{path} is missing and --offline is set")
        from huggingface_hub import hf_hub_download

        path = Path(hf_hub_download(PYIQA_META, spec.meta, repo_type="dataset", local_dir=data_root / "meta"))
    return pd.read_csv(path)


def _download(name: str, spec: Spec, target: Path) -> None:
    from huggingface_hub import hf_hub_download

    print(f"downloading {name} ({spec.archive}) from {PYIQA_DATA}", flush=True)
    path = Path(hf_hub_download(PYIQA_DATA, spec.archive, repo_type="dataset", local_dir=target))
    with tarfile.open(path) as handle:
        members = None
        if spec.members:
            members = [m for m in handle if set(Path(m.name).parts) & set(spec.members)]
        try:
            handle.extractall(target, members=members, filter="data")
        except TypeError:  # Python without tarfile extraction filters
            handle.extractall(target, members=members)


def load_dataset(name: str, data_root: Path, offline: bool = False) -> pd.DataFrame:
    """Return a frame with columns ref, dist (paths relative to ``data_root``) and mos."""
    if name not in SPLITS:
        raise ValueError(f"unknown dataset {name!r}")
    physical, subset = SPLITS[name]
    spec = DATASETS[physical]
    meta = _meta(spec, data_root, offline)
    if physical == "pipal":
        meta = meta[meta.official_split.eq(subset)].reset_index(drop=True)
    if len(meta) != spec.rows:
        raise ValueError(f"{physical}: expected {spec.rows} pairs, metainfo has {len(meta)}")
    refs = spec.ref_dir + meta[spec.ref_column].astype(str)
    dists = spec.dist_dir + meta.dist_name.astype(str)
    root = data_root / physical
    if _locate(root, refs.iloc[0]) is None or _locate(root, dists.iloc[0]) is None:
        if offline:
            raise FileNotFoundError(f"{name}: no images below {root} and --offline is set")
        _download(physical, spec, root)
    ref_base, dist_base = _locate(root, refs.iloc[0]), _locate(root, dists.iloc[0])
    if ref_base is None or dist_base is None:
        raise FileNotFoundError(f"{name}: could not find {refs.iloc[0]} and {dists.iloc[0]} below {root}")
    frame = pd.DataFrame({
        "ref": [(ref_base / r).relative_to(data_root).as_posix() for r in refs],
        "dist": [(dist_base / d).relative_to(data_root).as_posix() for d in dists],
        "mos": meta[spec.score].astype(float),
    })
    if physical == "kadid10k" and subset:
        split = json.loads((ROOT / "splits" / "kadid10k_split.json").read_text())
        keep = set(split[subset])
        frame = frame[frame.ref.map(lambda p: Path(p).name).isin(keep)].reset_index(drop=True)
    missing = [p for p in frame.ref.tolist() + frame.dist.tolist() if not (data_root / p).is_file()]
    if missing:
        raise FileNotFoundError(f"{name}: {len(missing)} image files missing, e.g. {data_root / missing[0]}")
    return frame


class _Pairs(torch.utils.data.Dataset):
    def __init__(self, frame: pd.DataFrame, data_root: Path, patch: int) -> None:
        self.refs = [data_root / p for p in frame.ref]
        self.dists = [data_root / p for p in frame.dist]
        self.patch = patch

    def __len__(self) -> int:
        return len(self.refs)

    def __getitem__(self, index: int):
        return load_pair(self.refs[index], self.dists[index], self.patch)


def _pairs(frame: pd.DataFrame, data_root: Path, patch: int, workers: int):
    loader = torch.utils.data.DataLoader(
        _Pairs(frame, data_root, patch), batch_size=None, num_workers=workers, pin_memory=torch.cuda.is_available()
    )
    yield from loader


# ------------------------------------------------------------------- metrics
def _resize_shorter(x: torch.Tensor, size: int = 256) -> torch.Tensor:
    h, w = x.shape[-2:]
    scale = size / min(h, w)
    out = (max(1, round(h * scale)), max(1, round(w * scale)))
    if out == (h, w):
        return x
    return F.interpolate(x, size=out, mode="bicubic", align_corners=False, antialias=True).clamp(0, 1)


def make_scorer(name: str, device: str, encoder: ViTEncoder | None, lens: Lens):
    """Return (callable(ref, dist) -> float, patch size of the loading protocol)."""
    if name in JLD_METRICS:
        use_lens, sigma, weight = VARIANTS[JLD_METRICS[name]]
        metric = JLD(encoder, lens.U if use_lens else None, sigma, weight)
        return (lambda x1, x2: float(metric(x1, x2)[0])), encoder.patch
    import pyiqa

    model = pyiqa.create_metric(PYIQA_METRICS[name], device=torch.device(device))

    def score(x1: torch.Tensor, x2: torch.Tensor) -> float:
        x1, x2 = x1.to(device), x2.to(device)
        if name == "dists":
            x1, x2 = _resize_shorter(x1), _resize_shorter(x2)
        with torch.inference_mode():
            return float(model(x2, x1).float().mean())  # pyiqa takes (distorted, reference)

    return score, 1


# -------------------------------------------------------------- correlations
def _logistic5(x, b1, b2, b3, b4, b5):
    return b1 * (0.5 - 1.0 / (1 + np.exp(b2 * (x - b3)))) + b4 * x + b5


def plcc_logistic(pred, mos) -> float:
    """PLCC after a 5-parameter logistic fit (several starts; linear PLCC is the floor)."""
    pred, mos = np.asarray(pred, float), np.asarray(mos, float)
    x = (pred - pred.mean()) / (pred.std() + 1e-12)
    best = abs(pearsonr(x, mos)[0])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # failed starts overflow or leave the covariance undefined
        for sign in (1.0, -1.0):
            for b2 in (0.5, 1.0, 2.0, 4.0):
                try:
                    p0 = [sign * (mos.max() - mos.min()), b2, 0.0, 0.0, float(mos.mean())]
                    popt, _ = curve_fit(_logistic5, x, mos, p0=p0, maxfev=20000)
                    r = abs(pearsonr(_logistic5(x, *popt), mos)[0])
                    if np.isfinite(r):
                        best = max(best, r)
                except Exception:
                    pass
    return float(best)


def correlations(values, mos) -> dict:
    values, mos = np.asarray(values, float), np.asarray(mos, float)
    return {
        "SRCC": abs(spearmanr(values, mos)[0]),
        "PLCC": plcc_logistic(values, mos),
        "KRCC": abs(kendalltau(values, mos)[0]),
        "n": len(values),
    }


def srcc_table(summary: pd.DataFrame, metrics: list[str], datasets: list[str], complete: bool) -> str:
    """SRCC by metric and dataset, with the paper's values for the JLD rows when they apply."""
    table = summary.pivot(index="metric", columns="dataset", values="SRCC").reindex(index=metrics, columns=datasets)
    with_mean = all(d in datasets for d in MEAN_OVER)
    if with_mean:
        table["mean_first_four"] = table[MEAN_OVER].mean(axis=1)
    if complete:  # the paper's numbers are for whole datasets only
        for name in [m for m in metrics if m in PAPER_SRCC]:
            row = {d: PAPER_SRCC[name].get(d, np.nan) for d in datasets}
            if with_mean:
                row["mean_first_four"] = float(np.mean([PAPER_SRCC[name][d] for d in MEAN_OVER]))
            table.loc[f"{name} (paper)"] = row
    return table.to_string(float_format=lambda v: f"{v:.3f}")


# ---------------------------------------------------------------------- main
def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", nargs="+", default=MAIN_TABLE, choices=sorted(SPLITS),
                        help="default: the five columns of the paper's main table")
    parser.add_argument("--metrics", nargs="+", default=["jld", "jld_fast"], choices=[*JLD_METRICS, *PYIQA_METRICS],
                        help="jld_lens and jld_raw are ablations; the others are pyiqa baselines")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data", help="folder holding the datasets")
    parser.add_argument("--out", type=Path, default=ROOT / "results", help="folder for per-pair scores and summary.csv")
    parser.add_argument("--lens", type=Path, default=PRETRAINED_LENS, help="lens .npz (default: the shipped lens)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--workers", type=int, default=8, help="image-loading worker processes")
    parser.add_argument("--limit", type=int, default=0,
                        help="score only N evenly spaced pairs per dataset (quick check; not the paper's numbers)")
    parser.add_argument("--offline", action="store_true", help="never download; fail if a dataset is missing")
    args = parser.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    lens = Lens.load(args.lens)
    encoder = None
    if any(m in JLD_METRICS for m in args.metrics):
        encoder = ViTEncoder(lens.meta["encoder"], lens.meta["layer"], args.device)
    rows = []
    for dataset in args.datasets:
        frame = load_dataset(dataset, args.data_root, args.offline)
        if args.limit and args.limit < len(frame):
            keep = np.linspace(0, len(frame) - 1, args.limit).round().astype(int)
            frame = frame.iloc[keep].reset_index(drop=True)
        scores_path = args.out / f"{dataset}_scores.csv"
        if scores_path.is_file():
            cached = pd.read_csv(scores_path)
            if cached.dist.tolist() == frame.dist.tolist():
                frame = cached
        for name in args.metrics:
            if name not in frame or frame[name].isna().any():
                scorer, patch = make_scorer(name, args.device, encoder, lens)
                started = time.perf_counter()
                frame[name] = [scorer(x1, x2) for x1, x2 in _pairs(frame, args.data_root, patch, args.workers)]
                seconds = time.perf_counter() - started
                print(f"{dataset:<14} {name:<10} {len(frame)} pairs in {seconds:.0f} s", flush=True)
                frame.to_csv(scores_path, index=False)
            rows.append({"dataset": dataset, "metric": name, **correlations(frame[name], frame.mos)})
            print(f"{dataset:<14} {name:<10} SRCC {rows[-1]['SRCC']:.3f}  PLCC {rows[-1]['PLCC']:.3f}  "
                  f"KRCC {rows[-1]['KRCC']:.3f}  n {rows[-1]['n']}", flush=True)
    summary = pd.DataFrame(rows)
    summary.to_csv(args.out / "summary.csv", index=False)
    note = f" on {args.limit} pairs per dataset (a subset, not comparable with the paper)" if args.limit else ""
    print(f"\nSRCC{note}\n" + srcc_table(summary, args.metrics, args.datasets, complete=not args.limit))


if __name__ == "__main__":
    main()

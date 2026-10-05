"""Fitting, saving and loading a Jacobian lens.

For tokens ``h`` at one encoder block and the encoder's final CLS output ``c``,
the lens is the top-k eigenbasis of

    M = E[J^T J],   J = d c / d h_t,

averaged over images and patch positions t. ``M`` is estimated without labels
with Gaussian output probes: for v ~ N(0, I), g = J^T v satisfies
E[g g^T] = J^T J, and one backward pass gives g at every token at once.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import torch
from PIL import Image
from torch import Tensor

from jld.images import load_image

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass
class Lens:
    """A fitted lens: the operator ``M`` and its top-k eigenvectors ``U`` [dim, k]."""

    U: np.ndarray
    M: np.ndarray
    eigenvalues: np.ndarray
    meta: dict = field(default_factory=dict)

    @property
    def k(self) -> int:
        return int(self.U.shape[1])

    def save(self, path: str | Path) -> Path:
        path = Path(path).with_suffix(".npz")
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            U=np.asarray(self.U, dtype=np.float32),
            M=np.asarray(self.M, dtype=np.float32),
            eigenvalues=np.asarray(self.eigenvalues, dtype=np.float32),
            meta=np.asarray(json.dumps(self.meta, sort_keys=True)),
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> "Lens":
        with np.load(Path(path), allow_pickle=False) as data:
            return cls(
                U=data["U"].astype(np.float32),
                M=data["M"].astype(np.float32),
                eigenvalues=data["eigenvalues"].astype(np.float32),
                meta=json.loads(str(data["meta"])),
            )


def top_eigenvectors(M: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Eigenvalues (descending) and the top-k eigenvectors of a symmetric matrix."""
    M = np.asarray(M, dtype=np.float64)
    values, vectors = np.linalg.eigh((M + M.T) * 0.5)
    order = np.argsort(values)[::-1]
    return values[order], vectors[:, order[:k]]


def subspace_overlap(U1: np.ndarray, U2: np.ndarray) -> float:
    """||U1^T U2||_F^2 / k for orthonormal bases: 1 means the same subspace."""
    U1, U2 = np.asarray(U1, np.float64), np.asarray(U2, np.float64)
    return float(np.sum((U1.T @ U2) ** 2) / min(U1.shape[1], U2.shape[1]))


def list_images(folder: str | Path) -> list[Path]:
    return sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)


def _crop_specs(paths: Sequence[Path], patch: int, n_crops: int, crop: int, rng) -> list[tuple]:
    """Random crop positions, drawn image by image in path order."""
    if crop % patch:
        raise ValueError(f"crop {crop} must be divisible by the patch size {patch}")
    specs = []
    for path in paths:
        with Image.open(path) as image:
            width, height = image.size
        height, width = height // patch * patch, width // patch * patch
        if height < crop or width < crop:
            raise ValueError(f"image {path} ({height}x{width}) is smaller than the crop {crop}")
        for _ in range(n_crops):
            top = int(rng.integers(0, height - crop + 1))
            left = int(rng.integers(0, width - crop + 1))
            specs.append((path, top, left))
    return specs


def _crop_batches(specs, patch: int, crop: int, batch: int, device) -> Iterator[Tensor]:
    crops, current, image = [], None, None
    for path, top, left in specs:
        if path != current:
            image, current = load_image(path, patch=patch), path
        crops.append(image[:, :, top : top + crop, left : left + crop])
        if len(crops) == batch:
            yield torch.cat(crops).to(device)
            crops = []
    if crops:
        yield torch.cat(crops).to(device)


def fit_lens(
    encoder,
    image_paths: Sequence[str | Path],
    k: int = 64,
    n_crops: int = 4,
    crop: int = 224,
    n_probes: int = 8,
    seed: int = 0,
    batch: int = 8,
    verbose: bool = True,
) -> Lens:
    """Estimate ``M`` for ``encoder.layer`` on unlabelled images and return its top-k lens.

    Paper defaults: 100 images, 4 random 224 x 224 crops each, 8 Gaussian probes
    per crop batch, seed 0, batch 8.
    """
    paths = [Path(p) for p in image_paths]
    if not paths:
        raise ValueError("no images given")
    device = encoder.device
    specs = _crop_specs(paths, encoder.patch, n_crops, crop, np.random.default_rng(seed))
    generator = torch.Generator(device=device)
    generator.manual_seed(seed + 2)
    M = np.zeros((encoder.dim, encoder.dim), dtype=np.float64)
    done = 0
    for crops in _crop_batches(specs, encoder.patch, crop, batch, device):
        crops.requires_grad_(True)
        hidden, cls = encoder(crops, need_cls=True)
        for _ in range(n_probes):
            probe = torch.randn(cls.shape, generator=generator, device=device, dtype=cls.dtype)
            (grad,) = torch.autograd.grad((cls * probe).sum(), hidden, retain_graph=True)
            g = grad[:, encoder.n_prefix :].float()  # [B, N, dim], patch tokens only
            M += (g.reshape(-1, g.shape[-1]).T @ g.reshape(-1, g.shape[-1]) / g.shape[1]).double().cpu().numpy()
        done += crops.shape[0]
        del hidden, cls, crops
        if verbose:
            print(f"  fit: {done}/{len(specs)} crops", flush=True)
    M /= len(specs) * n_probes
    M = ((M + M.T) * 0.5).astype(np.float32)
    values, U = top_eigenvectors(M, k)
    meta = {
        "encoder": getattr(encoder, "model_name", type(encoder).__name__),
        "layer": int(encoder.layer),
        "k": int(k),
        "n_images": len(paths),
        "n_crops": int(n_crops),
        "crop": int(crop),
        "n_probes": int(n_probes),
        "probe": "isotropic Gaussian on the final CLS token",
        "seed": int(seed),
        "batch": int(batch),
    }
    return Lens(U=U.astype(np.float32), M=M, eigenvalues=values.astype(np.float32), meta=meta)


__all__ = ["Lens", "fit_lens", "list_images", "subspace_overlap", "top_eigenvectors"]

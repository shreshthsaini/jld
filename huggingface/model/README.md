---
license: mit
language:
  - en
library_name: jld
tags:
  - image-quality-assessment
  - perceptual-distance
  - full-reference
  - dinov2
  - pytorch
  - arxiv:2610.05967
---

# JLD: Perceptual Distance Through A Jacobian Lens

Official fitted Jacobian lens for **JLD** and **JLD-fast**, by Shreshth Saini,
Balu Adsumilli, and Alan C. Bovik.

[Paper](https://arxiv.org/abs/2610.05967) ·
[Code](https://github.com/shreshthsaini/jld) ·
[Project and visual examples](https://shreshthsaini.github.io/jld/) ·
[Hugging Face paper page](https://huggingface.co/papers/2610.05967) ·
[Interactive demo](https://huggingface.co/spaces/shreshthsaini/JLD-demo)

JLD measures the perceptual distance between an aligned reference image and a
distorted image. Lower scores mean more similar images. It projects early
DINOv2 patch features onto directions selected by the encoder's output
sensitivity. The lens is fitted from unlabeled images without human quality
ratings.

## Use with uv

```bash
uv add "jacobian-lens-distance @ https://github.com/shreshthsaini/jld/releases/download/v1.1.0/jacobian_lens_distance-1.1.0-py3-none-any.whl"
```

This installs the versioned release wheel and records it in `uv.lock`.
The distribution is named `jacobian-lens-distance`; the Python import remains `jld`.

```python
from jld import JLD

metric = JLD.pretrained("full")  # use "fast" for JLD-fast
distance = metric("reference.png", "distorted.png")
print(distance.item())

response = metric.map("reference.png", "distorted.png")
```

The package already includes this lens. The encoder weights are downloaded
separately through `timm` on first use. Python 3.10 or newer is required; CPU
and CUDA are supported.

To download the lens from this repository explicitly:

```python
from huggingface_hub import hf_hub_download
from jld import JLD

lens_path = hf_hub_download(
    repo_id="shreshthsaini/JLD",
    filename="jld_dinov2_s14_block1_k64.npz",
)
metric = JLD.pretrained("fast", lens_path=lens_path)
```

## Artifact and provenance

This repository contains the fitted projection, its estimated sensitivity
matrix, eigenvalues, and fitting metadata. The encoder remains frozen; its
weights are provided by
[timm/vit_small_patch14_dinov2.lvd142m](https://huggingface.co/timm/vit_small_patch14_dinov2.lvd142m),
under that repository's Apache 2.0 license.

| Setting | Released lens |
| --- | --- |
| Encoder | DINOv2-S/14, feature width 384 |
| Captured features | Patch tokens after block 1 |
| Projection rank | 64 |
| Fitting images | 100 DIV2K validation images |
| Fitting crops | Four random 224 × 224 crops per image |
| Gaussian probes | Eight per crop |
| Seed | 0 |

SHA-256 of `jld_dinov2_s14_block1_k64.npz`:

```text
0ae11bc65fb590e6d7b23e3840258746910efd4da94e0e147014c28afbc9c26c
```

This is the same lens shipped in the official Python package. Fitting code and
evaluation scripts are in the [source repository](https://github.com/shreshthsaini/jld).

## Evaluation and scope

The [paper](https://arxiv.org/abs/2610.05967) reports image-quality correlations
on TID2013, CSIQ, LIVE, a held-out KADID-10k reference split, and PIPAL
validation. The repository contains commands to reproduce those evaluations.
This Hub release does not add a new benchmark result.

Use native-resolution RGB inputs, with values in [0, 1]. Images are
center-cropped to multiples of 14 pixels without resizing, and each pair must
have matching spatial dimensions. Changes to resizing, cropping, or color
preprocessing can change scores and comparisons.

JLD requires a reference image. It is intended for aligned image comparisons,
such as restoration and compression evaluation. The patch response map shows
the lens term only. Full JLD adds a final CLS cosine term, so the full score
need not satisfy the triangle inequality. The lens term is a pseudometric.

The video's temporal encoder and fitted lens in the paper are separate from
this image lens. The image lens alone does not reproduce the paper's temporal
video results. See the project page for the evaluated video protocols.

## Citation

If you use JLD, its fitted lens, or its integrations in research, please cite:

```bibtex
@misc{saini2026jld,
  title={JLD: Perceptual Distance Through A Jacobian Lens},
  author={Shreshth Saini and Balu Adsumilli and Alan C. Bovik},
  year={2026},
  eprint={2610.05967},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2610.05967}
}
```

JLD code and this fitted lens are released under the MIT license. See
[LICENSE](LICENSE).

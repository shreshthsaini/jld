---
title: JLD Perceptual Distance
emoji: 🔍
colorFrom: blue
colorTo: yellow
sdk: gradio
sdk_version: 6.29.1
python_version: "3.12"
app_file: app.py
pinned: false
license: mit
tags:
  - image-quality-assessment
  - perceptual-distance
  - full-reference
  - dinov2
  - arxiv:2610.05967
---

An author-maintained interactive demo of **JLD: Perceptual Distance Through a Jacobian Lens**.

[Paper](https://arxiv.org/abs/2610.05967) · [Code](https://github.com/shreshthsaini/jld) · [Research website](https://shreshthsaini.github.io/jld/)

Upload an aligned reference/distorted pair, select JLD or JLD-fast, and obtain
a scalar distance plus a local lens response overlay. Lower means more similar.
The overlay excludes the global CLS term and is normalized separately for each pair.

Both images must have the same native size. The demo converts to RGB and
center-crops to multiples of 14 pixels without resizing. Public demo resource
limits are 262,144 pixels, a longest side of 1024 pixels, and 10 MB per image.
Use the local package for larger images. The pretrained encoder loads on hosted
startup, while the fitted lens is included in the package. The hosted Space uses
free shared ZeroGPU access; queueing and daily usage limits may apply.

## Local use with uv

```bash
uv add "jacobian-lens-distance @ https://github.com/shreshthsaini/jld/releases/download/v1.1.0/jacobian_lens_distance-1.1.0-py3-none-any.whl"
```

```python
from jld import JLD

metric = JLD.pretrained("full")
distance = metric("reference.png", "distorted.png")
```

To run this Space checkout locally:

```bash
uv run --python 3.12 --with-requirements requirements.txt app.py
```

The app runs on CPU and listens on localhost for local runs. Hugging Face manages
external access and dynamic GPU allocation when deployed as a ZeroGPU Space.

## Citation

```bibtex
@article{saini2026jld,
  title={JLD: Perceptual Distance Through A Jacobian Lens},
  author={Saini, Shreshth and Adsumilli, Balu and Bovik, Alan C.},
  journal={arXiv preprint arXiv:2610.05967},
  year={2026}
}
```

Code is MIT licensed. Bundled images have their own provenance and source reuse
statement in [examples/PROVENANCE.md](examples/PROVENANCE.md). Encoder weights
retain their upstream license. No uploaded images or scores are deliberately
archived by the app; Gradio and Hugging Face may keep temporary upload files
under their platform policies.

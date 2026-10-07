# Example image provenance

These PNGs are copied unchanged from the public JLD repository's `examples/`
directory. They use reference photographs from the Kodak Lossless True Color
Image Suite hosted by Rich Franzen at https://r0k.us/graphics/kodak/:

| Reference | Kodak source | Release preprocessing |
|---|---|---|
| `parrots.png` | `kodim23.png` | Resize to 294-pixel height, center-crop to 392 × 294 |
| `hats.png` | `kodim03.png` | Resize to 294-pixel height, center-crop to 392 × 294 |
| `bikes.png` | `kodim05.png` | Resize to 294-pixel height, center-crop to 392 × 294 |

The source site's reuse statement says Rich Franzen understands the Eastman
Kodak Company released the photographs for unrestricted usage. This is the
source's statement, not a claim that JLD authors own these photographs or
license them under MIT. Checked 2026-10-07.

Distortions follow `examples/make_examples.py` in the JLD release:

- `parrots_jpeg_q8.png` and `parrots_jpeg_q30.png`: Pillow JPEG round trip at quality 8 and 30.
- `bikes_blur.png`: Pillow Gaussian blur, radius 2.
- `hats_noise.png`: Gaussian noise with standard deviation 12 on 8-bit RGB, NumPy RNG seed 0.

These are reproducible demonstration pairs. The Space computes their scores
live with the released model and does not present them as a benchmark result.

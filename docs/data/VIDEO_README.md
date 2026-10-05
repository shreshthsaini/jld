# Video examples

Open any MP4 from `../assets/videos/` directly in a video player.
The project page at `../index.html` also works offline.
The eight original files are muted comparison excerpts copied byte-for-byte
from recorded outputs. Version 1.1 adds four actual map movies under
`../assets/video-maps/`, with original MP4 and browser-compatible WebM versions.
No new model inference or score computation was performed.

## Waterloo: temporal VideoMAE lens

waterloo-14.mp4 (railway), waterloo-08.mp4 (football), waterloo-16.mp4 (dance)
and waterloo-11.mp4 (game scene) show synchronized reference/A/B excerpts at
the original cadence. The fixed orange rectangle locates an identical crop.
MOS prefers A in all four. The lens agrees for the first three; VMAF agrees
for the game scene. These cases were selected for disagreements, not sampled
to estimate accuracy. Dance's VMAF gap is only 0.051237 points and the game
scene's lens gap only 0.0251947 feature units. Visible artifacts are observations
of pixels, not token attribution or proof of why a whole-video score changes.

## AVT: separate spatial image-JLD pilot

avt-bigbuck_bunny_8bit.mp4, avt-cutting_orange_tuil.mp4,
avt-vegetables_tuil.mp4 and avt-water_netflix.mp4 show comparison images and
score cards for four recorded examples from a separate study. The actual
local map movies are `../assets/video-maps/<scene>/maps.{mp4,webm}`.
This study applies the image-JLD spatial
component frame by frame with no temporal encoder or global term. It is not
the manuscript's VideoMAE held-out score. Twelve full 448 x 252 frames score
each complete clip; the original human ratings were collected at 4K.

One example per source minimizes the PSNR difference subject to separated
published MOS intervals and a 0.75-point MOS gap. Selection does not use JLD,
raw-DINO or LPIPS outcomes. All four methods disagree with MOS in the orange
example. Absolute human MOS values are not direct pairwise votes, and there
are no spatial human labels. The display movies use every second source
frame at half the source rate, preserving elapsed speed.

The interactive explorer shows twelve evenly spaced display frames per scene,
with the reference, both distortions and both JLD maps visible together.
The maps use square roots of the archived squared projected-displacement
arrays, a linear magma color scale fixed across both variants and all 90
display frames, nearest-neighbor rendering, and no clipping. Raw arrays,
hashes, source-frame indices and exact timestamps are in `video-maps.json`
and `video-map-arrays/`. The slider does not recompute the clip scores.

## Paper's per-frame profile

`../assets/video-profile.png` is the unchanged final-paper figure with two
200-frame distance curves and high/low-response maps. This separate protocol
scores every third whole 1080p frame. Full-image curves include the global
term; displayed maps are local responses. These figure overlays share a
99.5th-percentile display limit. Details are in `video-profile.json`.

## Evidence

video_manifest.json records the eight files, original hashes, dimensions and
cadence. video/ retains exact source identities, source hashes, source-frame
indices, scores, selection rules and protocol. Machine-specific paths have
been replaced by portable descriptive roots. The original supplementary pixel archive is not included in this compact release.
Sources and clips are not interchangeable across the two study protocols.

Dataset attribution: Waterloo IVC 4K compression database;
AVT-VQDB-UHD-1, Rao et al., IEEE ISM 2019. Source-content credits and licenses
remain in the datasets' official documentation.

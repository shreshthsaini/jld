# JLD integration handover

Updated: 2026-10-07.

The official implementation is public at https://github.com/shreshthsaini/jld.
Paper: https://arxiv.org/abs/2610.05967, an October 2026 preprint.

## Evaluation tool integration

Opened and verified https://github.com/chaofengc/IQA-PyTorch/issues/304 under
`shreshthsaini`: "Integration proposal: JLD and JLD-fast full-reference perceptual distances".
The issue proposes `jld` and `jld_fast`, documents the native-resolution crop
protocol and differentiable adapter requirements, and offers implementation,
tests, documentation, and plugin maintenance.

PyIQA's current architecture registry supports external entry points, but
`create_metric()` and `list_models()` consult `DEFAULT_CONFIGS`. The issue asks
for the recommended plugin configuration/discovery path and whether maintainers
prefer a plugin or an upstream PR. Treat this as an observed integration question,
not a statement about upstream acceptance. The implementation and validation
below were completed after the proposal.

The issue explicitly distinguishes executable metric support from the existing
paper-list contribution at
https://github.com/chaofengc/Awesome-Image-Quality-Assessment/pull/48.

Opened and verified https://github.com/Lightning-AI/torchmetrics/issues/3561 under
`shreshthsaini`: "Integration proposal: JLD and JLD-fast image perceptual distances".
The proposal follows TorchMetrics' feature-request structure and covers a
`JacobianLensDistance` module, full/fast variants, sample-weighted distributed
accumulation, registered encoder/lens state, and validation. It asks whether
maintainers prefer a native implementation or an optional package dependency,
and whether to include a functional interface. An author-maintained compatible
wrapper is offered as an initial alternative.

On 2026-10-07 the user removed PIQ from the integration plan. The two targets are
PyIQA and TorchMetrics. The PyIQA issue was still open with no comments when
checked during the TorchMetrics proposal step.

## Adoption release, 7 Oct 2026

Implemented author-maintained integrations in `jld/integrations/`: PyIQA
`register()` plus `jld`/`jld_fast`, differentiable `JLDModule`, and
TorchMetrics `JacobianLensDistance` with distributed sample-weighted mean.
The original scorer is unchanged. PyIQA 0.1.16 requires explicit registration
before `create_metric()` or `list_models()`.

Renamed the Python distribution to `jacobian-lens-distance` and bumped to 1.1.0.
The unrelated PyPI package `jld` belongs to another author. The Python import
and CLI remain `jld`. README now uses uv and documents both optional adapters.
Builds produce a roughly 663 KB wheel with the bundled lens and no website assets.
GitHub Actions build, metadata checking, tests, and PyPI Trusted Publishing are
configured in `.github/workflows/`.

Validation on the existing Vista CPU workspace: 73 tests passed, one CUDA test
skipped; eight Space tests passed. A fresh noneditable wheel install from outside
the checkout verified entry points, full/fast scores through both integrations,
and CLI map output. Two-process Gloo verifies unequal-rank sample weighting.
No new benchmark or paper result was produced. No new compute jobs were launched.

Hugging Face paper https://huggingface.co/papers/2610.05967 is indexed and was
submitted to October 7 Daily Papers. The public model repository
https://huggingface.co/shreshthsaini/JLD contains the unchanged fitted lens,
configuration, MIT code/artifact license, and model card, and is linked from
the paper automatically.

The Gradio demo in `huggingface/space/` supports uploaded pairs, released example
images, both variants, and local response overlays. Hosted CPU creation returned
HTTP 402 because HF now requires PRO for CPU Gradio hosting. Free ZeroGPU creation
succeeded for https://huggingface.co/spaces/shreshthsaini/JLD-demo. The app has
been adapted to preload the shared encoder and use a short GPU scoring function.
The Space reached RUNNING and anonymous live API calls verified both variants
and response maps against the released examples. Desktop/mobile browser checks
passed; mobile has no horizontal overflow. The default pair now loads on arrival.

Two account steps need the user: configure the pending PyPI Trusted Publisher
listed in `.agent/notes/pypi-publishing.txt`; claim authorship on the HF paper
page, since the available token returned HTTP 403 for that account action.
The working versioned wheel installation is independent of PyPI readiness.
GitHub release https://github.com/shreshthsaini/jld/releases/tag/v1.1.0 is live
with wheel, source distribution, and SHA256SUMS. The public wheel was installed
cleanly and its download hash matched the validated local final build. GitHub
CI passed on Python 3.10 and 3.12. Build/publish run 37580454484 passed the build
but PyPI rejected OIDC with invalid-publisher because no matching publisher
was configured. After the user configures it, rerun that failed publish job.

HF collection groups the paper, lens, and Space:
https://huggingface.co/collections/shreshthsaini/jld-perceptual-distance-through-a-jacobian-lens-6ac5e3613e5371d47f962c05
Both artifacts are independently verified in the paper API's linked lists.
HF lens bytes match the bundled lens. No local browser or server remains running.
The real `uv init`, `uv add` from the public wheel, and `uv run` flow succeeded
with both optional adapters. Integration issues #304 and #3561 and IQA
paper-list PR #48 now link the shipped package, adapter instructions, and demo.

Submitted issue texts are preserved in `.agent/notes/pyiqa-proposal.txt` and
`.agent/notes/torchmetrics-proposal.txt`.

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
not evidence that a plugin has been implemented or tested.

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

Next: check both maintainer responses and implement the agreed adapters, starting
with PyIQA. Validate parity, batching, device moves, gradients, discovery for
PyIQA, and update/compute/reset and distributed reductions for TorchMetrics.
No adapter or new benchmark was produced in these proposal steps. No compute
jobs were launched.

Submitted issue texts are preserved in `.agent/notes/pyiqa-proposal.txt` and
`.agent/notes/torchmetrics-proposal.txt`.

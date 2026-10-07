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

Next: check the maintainer response, then implement the agreed PyIQA adapter and
validate parity, batching, device moves, gradients, and API/CLI/benchmark discovery.
PIQ and a TorchMetrics-compatible wrapper follow PyIQA. No adapter, new benchmark,
PIQ submission, or TorchMetrics wrapper was produced in this proposal step.
No compute jobs were launched.

The submitted issue text is preserved in `.agent/notes/pyiqa-proposal.txt`.

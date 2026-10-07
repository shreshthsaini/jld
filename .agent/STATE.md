## Goal
Publish JLD with uv installation, HF paper/model/Space, and PyIQA/TorchMetrics integrations.

## Status and next action
Validated: 73 tests pass, one CUDA skip, eight demo tests pass, clean installed-wheel smoke passes.
GitHub v1.1.0 wheel live and cleanly installed; CI passes Python 3.10/3.12.
HF paper indexed/Daily-submitted; model, ZeroGPU Space and collection live and linked.
Anonymous hosted full/fast scoring and desktop/mobile browser checks passed.
Actual uv init/add/run from the public wheel succeeded with both adapters.
Integration proposals and IQA paper-list PR now link the shipped package/demo.
NEXT: after pending PyPI publisher is configured, rerun failed publish job 37580454484.

## Map
| Note | Read it when |
| --- | --- |
| HANDOVER.md | Resuming adoption release |
| .agent/notes/pypi-publishing.txt | Configuring initial PyPI Trusted Publisher |
| huggingface/DISSEMINATION.md | Reviewing HF paper actions and account steps |
| huggingface/paper_receipt.json | Checking sanitized submission evidence |
| .agent/notes/pyiqa-proposal.txt | Reviewing PyIQA issue #304 |
| .agent/notes/torchmetrics-proposal.txt | Reviewing TorchMetrics issue #3561 |

## Decisions
Use distribution jacobian-lens-distance; retain Python import and CLI jld.
Target PyIQA and TorchMetrics only; user removed PIQ.
PyIQA requires explicit register() because its factory checks configurations first.
Keep core scoring unchanged; adapter encoder is a child module and lens a buffer.
TorchMetrics uses per-pair sums/counts for sample-weighted distributed mean.
Use free HF ZeroGPU after CPU hosting returned HTTP 402 PRO requirement.
Release wheel provides uv installation while PyPI publisher is configured.

## Open questions
User must configure pending PyPI publisher; asynchronous request has exact values.
HF authorship claim requires user browser authentication (token returned HTTP 403).
PyPI publish was attempted and rejected invalid-publisher; package is not on PyPI yet.
No new compute allocation; completed test processes exited.

## Goal
Publish JLD with uv installation, HF paper/model/Space, and PyIQA/TorchMetrics integrations.

## Status and next action
Validated: 73 tests pass, one CUDA skip, eight demo tests pass, clean installed-wheel smoke passes.
HF paper indexed and Daily-submitted; model published; free ZeroGPU Space created.
NEXT: publish GitHub v1.1.0 wheel, upload Space, verify hosted scoring and artifact links.

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
PyIQA requires explicit register() due factory validation order.
Keep core scoring unchanged; adapter encoder is a child module and lens a buffer.
TorchMetrics uses per-pair sums/counts for sample-weighted distributed mean.
Use free HF ZeroGPU after CPU hosting returned402 PRO requirement.
Release wheel provides uv installation while PyPI publisher is configured.

## Open questions
User must configure pending PyPI publisher; asynchronous request has exact values.
HF authorship claim requires user browser authentication (token returned403).
Hosted ZeroGPU scoring and GitHub CI still need verification.
No new compute allocation; completed test processes exited.

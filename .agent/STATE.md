## Goal
Introduce JLD into PyIQA and TorchMetrics, starting with integration proposals.

## Status and next action
Proposals complete: PyIQA #304 and TorchMetrics #3561 are open and verified under shreshthsaini.
NEXT: check both issues for maintainer guidance, then implement the agreed adapters.

## Map
| Note | Read it when |
| --- | --- |
| HANDOVER.md | Resuming evaluation-tool integration |
| .agent/notes/pyiqa-proposal.txt | Reviewing the exact submitted issue text |
| .agent/notes/torchmetrics-proposal.txt | Reviewing the TorchMetrics proposal |

## Decisions
Propose both full and fast image distances with FR mode and lower-better scores.
Preserve the official preprocessing and differentiable tensor API in an adapter.
Ask about DEFAULT_CONFIGS discovery because architecture entry points alone appear insufficient.
Keep paper-list outreach separate from executable metric integration.
Target PyIQA and TorchMetrics only; the user removed PIQ on 2026-10-07.
Use summed per-pair scores and counts for a sample-weighted distributed mean.

## Open questions
Do PyIQA maintainers prefer a plugin or upstream PR?
What is their recommended external default-configuration registration path?
Do TorchMetrics maintainers prefer a native implementation or optional package dependency?
Should the TorchMetrics contribution include both module and functional interfaces?
No adapter or new benchmark was run; no compute jobs are in flight.

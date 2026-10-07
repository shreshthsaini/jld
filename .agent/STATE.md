## Goal
Introduce JLD into researchers' evaluation tools, starting with a PyIQA proposal.

## Status and next action
Proposal step complete: PyIQA issue #304 is open and verified under shreshthsaini.
NEXT: check https://github.com/chaofengc/IQA-PyTorch/issues/304 for maintainer guidance.

## Map
| Note | Read it when |
| --- | --- |
| HANDOVER.md | Resuming evaluation-tool integration |
| .agent/notes/pyiqa-proposal.txt | Reviewing the exact submitted issue text |

## Decisions
Propose both full and fast image distances with FR mode and lower-better scores.
Preserve the official preprocessing and differentiable tensor API in an adapter.
Ask about DEFAULT_CONFIGS discovery because architecture entry points alone appear insufficient.
Keep paper-list outreach separate from executable metric integration.
Follow PyIQA with PIQ and a TorchMetrics-compatible wrapper.

## Open questions
Do PyIQA maintainers prefer a plugin or upstream PR?
What is their recommended external default-configuration registration path?
No adapter or new benchmark was run; no compute jobs are in flight.

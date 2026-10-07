# JLD Hugging Face publication receipts

Updated: 2026-10-07.

## Paper listing

Indexed https://huggingface.co/papers/2610.05967 under `shreshthsaini`.
`POST /api/papers/index` with `{"arxivId":"2610.05967"}` returned HTTP 200.
A separate public metadata read confirmed the title and three authors.

Submitted JLD to Daily Papers for October 7, 2026. The production endpoint
uses `POST /api/papers/submit` with `{"paperId":"2610.05967"}`. The server
redirected to the paper page, and independent public reads confirmed
`submittedOnDailyAt: 2026-10-07T00:00:00.000Z`, submitter `shreshthsaini`, and
a matching entry in the October 7 Daily Papers feed. Do not submit again.

ArXiv's first submission was October 5, 2026. The current official HF papers
skill states a 14-day Daily Papers window. An older SDK issue says seven
days and gives a snake_case payload that no longer matches production.

## Authorship and direct paper links

Attempted the documented authorship claim for Shreshth Saini using the existing
fine-grained token. `POST /api/settings/papers/claim` returned HTTP 403,
`Authorization error.` No authorship verification was completed.

Attempted to set the official GitHub and project URLs using
`POST /api/papers/2610.05967/links`. It returned HTTP 403,
`You are not allowed to update this paper.` Those direct fields remain unset.
The existing token can create and update repositories in `shreshthsaini`,
and it successfully indexed and submitted the paper, but it did not permit
these two account/paper management actions.

The remaining author action is to sign in to
https://huggingface.co/papers/2610.05967, click **Shreshth Saini**, choose
**Claim authorship**, and confirm in https://huggingface.co/settings/papers.
Hugging Face staff validate the claim. Once verified, add
https://github.com/shreshthsaini/jld and https://shreshthsaini.github.io/jld/
through the paper page's link editor if those fields are still empty.

Model and Space cards referencing https://arxiv.org/abs/2610.05967 can link
their artifacts to the paper independently of this authorship step. Verify
the public paper API's `linkedModels` and `linkedSpaces` after upload.

## Release preparation

`model/README.md` is the prepared lens card. Upload it with the MIT license
and the unchanged `jld/data/jld_dinov2_s14_block1_k64.npz`. The card includes
the paper link, explicit `arxiv:2610.05967` tag, native-resolution protocol,
uv installation, artifact provenance, limitations, and citation.

Target artifact repositories are https://huggingface.co/shreshthsaini/JLD
and https://huggingface.co/spaces/shreshthsaini/JLD-demo. The model card uses
`uv add` from the official GitHub source until a PyPI upload is verified.
Replace that with `uv add jacobian-lens-distance` only after publication.

Sanitized API receipts are preserved in `paper_receipt.json`. Tokens, email
addresses, and authorization headers are excluded.

The account reports `isPro=false` and `canPay=false`. Use the free CPU Space;
do not request a paid hardware upgrade as part of this release.

Existing paper-list submissions should be updated with released integration
links when relevant; avoid submitting the same paper twice to the same list.
No email, Slack message, or private direct outreach was sent in this task.

## Official references

- Paper pages and author verification: https://huggingface.co/docs/hub/paper-pages
- Current index/claim/link endpoints: https://github.com/huggingface/skills/blob/main/skills/huggingface-papers/SKILL.md
- Older submit endpoint proposal: https://github.com/huggingface/huggingface_hub/issues/2745
- Model card and arXiv tagging: https://huggingface.co/docs/hub/model-cards

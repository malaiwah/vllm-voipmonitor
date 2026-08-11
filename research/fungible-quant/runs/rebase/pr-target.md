# Where the PR goes, and what to rebase on — decided 2026-08-11

Short answer, so nobody has to re-derive it:

| question | answer |
|---|---|
| PR base repo | `local-inference-lab/vllm` |
| PR base branch | **`dev/gilded-gnosis`** @ `e2666d9a65f41fc376607531453cbd57c4c71016` |
| PR head | `malaiwah/vllm-voipmonitor` branch `fq/m1-stats-collector`, opened cross-fork |
| rebase needed | **none** — we are already a clean linear descendant of that commit |

## The trap: GitHub pre-fills the WRONG base

`malaiwah/vllm-voipmonitor` forks **`vllm-project/vllm` directly**, not
local-inference-lab. So the compare view pre-fills the base as *upstream
vLLM's* `main`. Opening the PR without retargeting puts it in front of the
upstream vLLM maintainers rather than the GG maintainers — with a diff against
a tree 416+ commits divergent. **Retarget the base by hand.** Roughly 20
existing open PRs from `malaiwah:` and `voipmonitor:` are cross-fork into
`dev/gilded-gnosis`; follow that pattern.

## What "GG v20 r34" actually is

r34 is the latest release (highest tag on Docker Hub, HEAD of
`blackwell-llm-docker`). But aligning to it costs nothing: **r31, r33 and r34
all sit on the same vLLM base we are already on.**

r34 = base `e2666d9a` + **21 unmerged vLLM PR heads** composed into an isolated
tree `4d006a4` (including #280 @ `8e7be4d5` and #281 @ `126039af`). That tree
404s on GitHub — it exists only as a 708 KB `integration.patch` inside
`blackwell-llm-docker`. There are **zero git tags and zero GitHub releases in
the entire org**; release identity is the image digest.

Consequences:

- Base the PR on the **branch**, not on the composed tree — reviewers diff
  against the branch. The 0-conflict replay onto #280 is *evidence to cite*,
  not something to fold in.
- `disposition: "merged"` in the release lockfiles does **not** mean merged.
  All 21 PRs are still open, and #145 is *closed and unmerged* yet ships in
  the image (rtx6kpro issue #33 says so explicitly).
- **Do not couple our PR to #280.** Its merge chain is blocked at the bottom:
  B12X #144 (required first) currently has merge conflicts, and `b12x:master`
  has drifted 3 commits past r34's pin.

## Contribution requirements that bind us

`AGENTS.md` in `local-inference-lab/vllm` is **byte-identical to upstream's**
(same blob SHA `a53b8187`). So the PR must satisfy, non-negotiably:

- no pure code-agent PR — a human submitter understands and defends it;
- an explicit duplicate-work check (see `report.md` §2 — EPLB and #280);
- an explicit statement that AI assistance was used;
- test commands run, and their results.

PR template is the stock upstream one: **Purpose / Test Plan / Test Result**.
CI is `pre-commit` plus CodeRabbit — **no GPU CI**, so every correctness and
performance claim must come with our own evidence.

## Dependency map (for the PR's context section)

- `sparkinfer` **is** `b12x` — one repo mid-rename (identical commit/tree
  across both label namespaces; GG HEAD's message is `refactor: complete b12x
  rename`).
- `exllamav3` comes from **`brandonmmusic-max/exllamav3`** branch
  `a1-retile-sm120` — a personal fork outside both LIL and `turboderp-org`.

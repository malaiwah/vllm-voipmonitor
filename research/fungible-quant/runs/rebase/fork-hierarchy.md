# vLLM fork hierarchy, GG v20 r34 provenance, and where our PR goes

Investigated 2026-08-11. Every claim below was verified against the GitHub REST API,
the `rtx6kpro` docs repo, the `blackwell-llm-docker` release locks, and `skopeo inspect`
of the published r34 image manifest. Nothing here is inferred from convention.

## TL;DR — the actionable answer

| Question | Answer |
|---|---|
| PR against which repo/branch? | `local-inference-lab/vllm`, base branch **`dev/gilded-gnosis`** |
| Rebase onto what? | **`dev/gilded-gnosis@e2666d9a65f41fc376607531453cbd57c4c71016`** — i.e. *exactly what we are already on*. No rebase is required. |
| Does r34 move the vLLM base? | **No.** r34's canonical vLLM base is `e2666d9a`, unchanged since 2026-08-07. |
| Is r34 a branch we can rebase onto? | **No.** It is a composed integration tree `4d006a4…` that exists in no branch of any repo. |

**The operator's belief is half right.** "GG v20 r34" *is* the latest release (confirmed:
newest tag on Docker Hub, newest commit in `blackwell-llm-docker`). But "align with r34"
does **not** imply moving our base. r34 is built from the same `dev/gilded-gnosis@e2666d9a`
that our branch was cut from. We are already aligned. What r34 adds is 21 *unmerged PR heads*
composed on top — a release artifact, not a branch state.

## 1. Fork hierarchy map

### vLLM line

```
vllm-project/vllm                    (true upstream, not a fork, default: main)
  │
  ├── local-inference-lab/vllm       fork; parent=source=vllm-project/vllm
  │     ├── main                 e12b91b0  STALE MIRROR of upstream
  │     └── dev/gilded-gnosis    e2666d9a  <-- INTEGRATION POINT
  │
  ├── malaiwah/vllm-voipmonitor      fork; parent=source=vllm-project/vllm  (our `work` remote)
  └── voipmonitor/vllm               fork; parent=source=vllm-project/vllm
```

Verified via the `parent` / `source` fields on `GET /repos/{owner}/{repo}`.

**Surprising and load-bearing:** `malaiwah/vllm-voipmonitor` was forked from
`vllm-project/vllm`, **not** from `local-inference-lab/vllm`. GitHub therefore defaults a
new PR's base to `vllm-project/vllm:main`. That default is wrong for us and must be changed
by hand on every PR, or the PR lands in front of upstream vLLM maintainers.

`main` vs `dev/gilded-gnosis` (`GET /compare/main...dev/gilded-gnosis`):
`status=ahead, ahead_by=416, behind_by=0`. `main` is a strict ancestor — 416 commits behind.
Confirms the prior finding: **never rebase onto `main`**; it would delete the EXL3 backend
and the b12x MoE path.

`dev/gilded-gnosis` is **not** a protected branch, and `local-inference-lab/vllm` has
**zero releases and zero tags**. There is no git-level release marker anywhere in the org;
releases exist only as container images plus lock files.

Also present, both derived from `dev/gilded-gnosis`: `dev/gilded-gnosis-rebase` (9c8e5c3e),
`dev/gilded-gnosis-old-base` (80a725fc). Neither is the integration point. 122 branches total.

### Dependency repos

| Component | Repo | Owner / nature | Branch used by r34 |
|---|---|---|---|
| B12X | `local-inference-lab/b12x` | LIL original (not a fork) | `master` @ `7cecbb2c` |
| **sparkinfer** | *same repo as B12X* | — | identical commit/tree/patch |
| QSRT | `local-inference-lab/qsrt` | LIL original | (ABI contract via B12X #144) |
| LMCache | `local-inference-lab/LMCache` | fork of `LMCache/LMCache` | `release/v0.5.2-glm52-dcp-base` @ `9cebd405` |
| exllamav3 | **`brandonmmusic-max/exllamav3`** | third-party personal fork | `a1-retile-sm120` @ `704aefd7` |
| InstantTensor | `voipmonitor/InstantTensor` | — | `49b4010a` |
| FlashInfer | (pinned source) | — | `1ac69427` |
| NCCL | `local-inference-lab/nccl-canonical` | fork | `canonical/cu132-nccl2304-amd-noxml` |

**`sparkinfer` and `b12x` are the same project.** The r34 image labels emit both
`local-inference.sparkinfer.*` and `local-inference.b12x.*` with byte-identical commit,
integration tree, PR list, and patch SHA256. Image tags renamed the field `si…` → `b12x…`
at r29, and the current `dev/gilded-gnosis` HEAD commit message is literally
`refactor: complete b12x rename`. There is no separate `local-inference-lab/sparkinfer` repo (404).

**exllamav3 is not owned by the org.** It is `brandonmmusic-max/exllamav3`, a personal fork
on a bespoke branch. Upstream is `turboderp-org/exllamav3`. Anything we touch in the EXL3
path has a dependency outside LIL's control.

### The integration point

**`local-inference-lab/vllm:dev/gilded-gnosis` is where GG releases are cut from.**
Evidence: every release lock names it as `base.ref`; the image label
`local-inference.vllm.integration.base_commit` points at its HEAD; and 40+ of the ~50 open
PRs target it. It is the only branch that functions as a trunk.

## 2. What "GG v20 r34" concretely is

### Artifact

```text
Image:  voipmonitor/vllm:gilded-gnosis-v20-vllm4d006a4-b12xcd3ce19-fi1ac6942-cu132-20260810-r34
Digest: sha256:820181fbbc975cd5291c411cda9771d58fecee1636d916f508f47230df20592b
Created: 2026-08-10T18:44:25Z   Size: 24,993,088,498 bytes (11.67 GB compressed, 36 layers)
```

Confirmed newest: `skopeo list-tags docker://docker.io/voipmonitor/vllm` shows r34 as the
highest `r`-suffix (…r30, r31, r33, r34 — no r32 or r35 published). `blackwell-llm-docker`
HEAD is `98224d130` "release(gg): qualify GLM-5.2 R7 mixed-Trellis serving" (2026-08-10T19:45Z),
the r34 release commit itself.

r34 is the **GLM-5.2 R7** release. r33 (`vllmfa13d33-b12x06db0f4`) is the **DeepSeek-V4-Flash**
release. Different models, same v20 line — r34 does not supersede r33 for DS4.

### Source composition (from image labels, cross-checked against the lock file)

| Component | Canonical base | r34 composed tree |
|---|---|---|
| vLLM | `dev/gilded-gnosis@e2666d9a65f41fc376607531453cbd57c4c71016` | `4d006a43928cdee01306691a766542c1e9bebb59` |
| B12X / sparkinfer | `master@7cecbb2c4819636ae7f05f8b116f2c45ee2cff7b` | `cd3ce190f0f1917402cdfd5773724267cc9a63f8` |
| LMCache | `release/v0.5.2-glm52-dcp-base@9cebd405` | `9a05c8818bae48d15b79c7e876418bb813c08cd0` |

Image label `local-inference.cache.fingerprint = vllme2666d9a65-b12x7cecbb2c48-…` independently
confirms the vLLM base.

### The composed tree does not exist in any repo

`GET /repos/local-inference-lab/vllm/commits/4d006a4` → **404**. Same for r33's `fa13d33`.
The tag's `vllm4d006a4` is the SHA of an *isolated integration tree* built by the release
composer, never pushed. It is reconstructible only by replaying:

```
patches/releases/gilded-gnosis-v20-r34/vllm/integration.patch      (708,207 bytes)
patches/releases/gilded-gnosis-v20-r34/vllm/integration.lock.json  (5,392 bytes)
```
in `local-inference-lab/blackwell-llm-docker` @ `98224d1303c1497eec26c7d92f34a6fa9a58fa82`,
or via `VLLM_RELEASE_COMPOSITION=reproduce-r34 ./build-gilded-gnosis-v20-final-cu132.sh`.

### r34 = base + 21 unmerged PR heads

From `local-inference.vllm.integration.prs`, applied in this order:

```
145@99bf3f13  256@48e9d0f7  188@aa75b01c  229@1a2b01f4  213@5f144274
214@8d989f18  217@bde9a133  218@3b89c7a1  230@9401f9d8  234@eaf24cc1
235@54349f9c  245@9dbd6d00  248@85bc9770  251@c66ce732  253@1772842a
252@e3317de2  254@72c94709  255@d1b7bbeb  258@63b77c80  280@8e7be4d5
281@126039af
```

The lock file marks each `"disposition": "merged"`. **That word is misleading and the
project says so explicitly** — from `rtx6kpro` issue #33:

> The release lock field `disposition: merged` means that the composer applied a PR head to
> an isolated integration tree. It does not mean that GitHub reports the PR as merged.

Verified: **all 21 are still open on GitHub.** #145 is *closed, unmerged* and is explicitly
flagged "included in the r34 image but not requested for merge".

The canonical ordered merge contract lives in
[rtx6kpro issue #33](https://github.com/local-inference-lab/rtx6kpro/issues/33)
("[GG r34] Canonical source merge contract", open, every checkbox unchecked). It states:

> - Release artifact: **qualified**
> - Source integration: **implemented in the archived r34 composition**
> - Canonical branch integration: **not complete**

## 3. The actionable answer

### a. PR target

**`local-inference-lab/vllm`, base `dev/gilded-gnosis`.**

Push the head to `malaiwah/vllm-voipmonitor` (our `work` remote) and open the PR
cross-fork. Precedent is overwhelming: open PRs #271, #268, #266, #265, #258, #249, #240,
#186 all use `head=malaiwah:…` → `base=dev/gilded-gnosis`, and #253/#252/#251/#248/#245/#235/
#230/#229/#218/#217 use `head=voipmonitor:…` the same way.

**Watch the base.** Because our fork's parent is `vllm-project/vllm`, GitHub pre-fills the
base as `vllm-project/vllm:main`. Change it explicitly.

### b. Rebase target

**`e2666d9a65f41fc376607531453cbd57c4c71016` — no action needed.**

Verified locally (read-only, no working tree touched):

```
git merge-base fq/m1-stats-collector e2666d9a…  ->  e2666d9a…   (exact ancestor)
git rev-list --count e2666d9a…..fq/m1-stats-collector  ->  13
```

Our branch is a clean linear descendant of the r34 vLLM base with 13 commits
(was 11 at the time of the original brief; live work added 2). `dev/gilded-gnosis` has not
advanced since 2026-08-07T15:59:54Z, so there is nothing to catch up to.

### c. Unmerged PRs in r34 — what this changes

r34 contains **21 unmerged vLLM PR heads** composed on top of our base (full list above).
This means "rebase onto r34" is not a thing you can do against a branch. Two distinct targets:

- **For the PR (do this):** base `dev/gilded-gnosis@e2666d9a`. Reviewers diff against the
  branch, so the PR must apply to the *branch*, not to the composed tree. We are already there.
- **For runtime validation (separate concern):** the r34 composed tree. Our 11 commits
  already replay onto **#280 @ `8e7be4d5c97fb86d983bd5f83c825153452efaec`** with **0 conflicts**,
  which is the meaningful signal — #280 is by far the largest overlap with our area
  (27 commits, 14 files, +4750/-199, the EXL3 R7 mixed-K3/K4/K5 runtime).

Do **not** fold #280 into our PR. Keep the PR minimal against the branch and note the
#280 compatibility in the description.

Relevant PR states as of 2026-08-11:

| PR | State | Mergeable | Note |
|---|---|---|---|
| #280 EXL3 R7 mixed K3/K4/K5 | open, not merged | `true` / `clean` | depends on B12X #144 |
| #281 InstantTensor borrowed-buffer | open, not merged | `true` / `unstable` | pre-commit queued |
| B12X #144 | open, not merged | **`false` (conflicts)** | blocker for #280 |

**The chain is stalled at the bottom.** Issue #33 requires B12X #144 to land before vLLM #280,
and #144 currently does not merge cleanly into `b12x:master`. `b12x:master` has also moved
3 commits ahead of r34's pinned `7cecbb2c`. Expect #280 to sit unmerged for a while — another
reason not to couple our PR to it.

### d. Contribution requirements in the target repo

`local-inference-lab/vllm@dev/gilded-gnosis` carries `AGENTS.md`, `CONTRIBUTING.md`,
`CLAUDE.md`, and `.github/PULL_REQUEST_TEMPLATE.md`.

**`AGENTS.md` is byte-identical to upstream's** — blob SHA `a53b81873cf0d168c23fa25eef505065cbdba3a0`,
5654 bytes, on *both* `local-inference-lab/vllm@dev/gilded-gnosis` and `vllm-project/vllm@main`.
It was inherited verbatim, not customized, and its text says it governs contributions to
`vllm-project/vllm`. It has not been adapted or waived for the fork, so treat it as binding.
Our PR description must include:

- Why this is **not duplicating** an existing PR (~50 are open — check the EXL3/quant area,
  especially #280, #279, #277, #270, #249, #240).
- **Test commands run and results.**
- **Model evaluation results**, since our change affects output/accuracy/serving.
- An explicit **statement that AI assistance was used**.
- A human who understands and can defend the change end-to-end — *"Pure code-agent PRs are
  not allowed."*

Also from `AGENTS.md`, and consistent with our own standing practice: never use system
`python3` or bare `pip`; everything through `uv` and `.venv/bin/python`.

The duplicate-work check is specified as `gh` commands. **There is no `gh` on this box** —
run the equivalent REST queries with curl and paste those results instead.

`.github/PULL_REQUEST_TEMPLATE.md` is the stock upstream template: `## Purpose`,
`## Test Plan`, `## Test Result`, plus the essential-elements checklist. Fill all three.

**CI** — 6 active workflows: `pre-commit`, `New PR Bot`, `Add label on auto-merge enabled`,
`Label issues based on keywords`, `macOS Apple Silicon Smoke Test`, `Close inactive issues and PRs`.
Observed on live PR #281: `pre-commit` (the real gate), `reminder-comment`,
`update-description`, `pre-run-check`, plus a **CodeRabbit** commit status. No GPU CI and no
Buildkite in this fork — correctness evidence must be supplied by us in the PR body, which is
exactly what `AGENTS.md` demands. Run `pre-commit` locally before pushing.

`.github/CODEOWNERS` is upstream's, listing `vllm-project` maintainers who have no role here;
it will not produce useful auto-reviewers. `dev/gilded-gnosis` is unprotected, so merging is
by maintainer judgement rather than an enforced check set.

## Surprises worth flagging

1. **Our fork's parent is `vllm-project/vllm`, not `local-inference-lab/vllm`** — every PR
   defaults to the wrong base and will be aimed at upstream vLLM unless corrected.
2. **`disposition: "merged"` in the release locks does not mean merged.** All 21 vLLM PR heads
   in r34 are still open; #145 is closed and unmerged yet shipped in the image.
3. **r34 does not advance the vLLM base at all.** r31, r33 and r34 all sit on `e2666d9a`.
   "Align with r34" costs us nothing on the vLLM side.
4. **`sparkinfer` == `b12x`** — one repo, two label namespaces, mid-rename.
5. **exllamav3 comes from `brandonmmusic-max/exllamav3`**, a personal fork on branch
   `a1-retile-sm120`, not from LIL and not from `turboderp-org`.
6. **No git tags or GitHub releases exist in the whole org.** Release identity is the image
   digest plus the lock files — the composed tree SHA in the image tag resolves to 404 on GitHub.
7. **The r34 merge chain is blocked at B12X #144**, which currently has merge conflicts against
   `b12x:master`, and `b12x:master` has drifted 3 commits past the r34 pin.

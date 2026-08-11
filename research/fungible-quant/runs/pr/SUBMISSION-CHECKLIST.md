# Submission checklist — do these in order, by hand

Prepared 2026-08-11. **The PR is not opened.** Everything below is for the
human who opens it.

State at the time of writing:

| | |
|---|---|
| head | `malaiwah/vllm-voipmonitor` branch `fq/m1-stats-collector` @ `161536085362d5c717fa5f8a94ce70b873bfd1ff` |
| head pushed? | yes — local `HEAD` == `work/fq/m1-stats-collector`, 0 unpushed commits, clean tree |
| base repo | `local-inference-lab/vllm` |
| base branch | `dev/gilded-gnosis` |
| base branch head | `fa033bd4e1b16d9d729ad94be2d87da5a13210ce` (2026-08-11T02:28:24Z) |
| merge base | `e2666d9a65f41fc376607531453cbd57c4c71016` |
| our commits atop the merge base | 24 |
| base commits we do not have | 19 |
| merge conflicts | **none** (verified, §3) |

---

## 0. Read this first: the base branch moved after `pr-target.md` was written

`runs/rebase/pr-target.md` pinned the base at `e2666d9a`. `dev/gilded-gnosis`
has since advanced **19 commits** to `fa033bd4e`. Two consequences:

1. **PR #281 is now merged** (`5d2079094`, "[GG] loader: add explicit
   InstantTensor borrowed-buffer mode"). That makes separate report (a) in
   `SEPARATE-REPORTS.md` live against the base branch rather than
   hypothetical — the borrowed-buffer mode now exists on `dev/gilded-gnosis`
   and EXL3 is not safe under it.
2. Everything else in `pr-target.md` still holds. #277, #279 and #280 are all
   still **open**, so the coexistence analysis in the PR body is current.

Re-check both at submit time with the commands in §2 — these change daily.

---

## 1. Retarget the base branch BY HAND. This is the step that goes wrong.

`malaiwah/vllm-voipmonitor` is a fork of **`vllm-project/vllm` directly**, not
of `local-inference-lab/vllm`. So when you open a PR from this branch, GitHub
pre-fills:

```
base repository: vllm-project/vllm     base: main        <-- WRONG, both fields
head repository: malaiwah/vllm-voipmonitor    compare: fq/m1-stats-collector
```

Submitting that puts a 21,820-line diff in front of the **upstream vLLM**
maintainers, against a tree that is hundreds of commits divergent from what
this work was built on. It would be closed, and correctly.

**Do this instead:**

1. Go to <https://github.com/local-inference-lab/vllm/compare> — start from the
   *base* repo, not from the fork.
2. Click **compare across forks**.
3. Set the four fields explicitly, in this order (setting the base repo first
   prevents GitHub resetting the others):
   - base repository: **`local-inference-lab/vllm`**
   - base: **`dev/gilded-gnosis`**  ← *not* `main`
   - head repository: **`malaiwah/vllm-voipmonitor`**
   - compare: **`fq/m1-stats-collector`**
4. **Before clicking "Create pull request", re-read the base dropdown.**
   Confirm it says `local-inference-lab/vllm` and `dev/gilded-gnosis`.
5. After creating, check the PR header line reads
   `local-inference-lab:dev/gilded-gnosis ← malaiwah:fq/m1-stats-collector`
   and that the file count is ~46, not ~2000. A file count in the thousands
   means the base is still `main`.

`main` is a trap in its own right: `local-inference-lab/vllm`'s
`default_branch` is `main`, but `main` (`e12b91b03`) is a strict **ancestor**
of `dev/gilded-gnosis`, 416 commits behind, and is the stale vanilla-vLLM
mirror. It has no EXL3 backend, no b12x fused-MoE path, and no
`MoERunner`/`BaseRouter` classes for this branch to bind to.

There are ~20 existing open PRs from `malaiwah:` and `voipmonitor:` opened
cross-fork into `dev/gilded-gnosis`. Follow that pattern; it is the norm in
this repo, not an exception.

---

## 2. Re-run the duplicate-work check at submit time

`AGENTS.md` in `local-inference-lab/vllm` is byte-identical to upstream's
(blob `a53b8187`), so this is mandatory, not optional. Results go in the PR
body — the "Why this is not duplicating an existing PR" section is already
written; update it if any of these return something new.

With `gh` (the form `AGENTS.md` prescribes):

```bash
gh pr list --repo local-inference-lab/vllm --state open --limit 100
gh pr list --repo local-inference-lab/vllm --state open --search "exl3"
gh pr list --repo local-inference-lab/vllm --state open --search "mixed trellis"
gh pr view 277 --repo local-inference-lab/vllm
gh pr view 279 --repo local-inference-lab/vllm
gh pr view 280 --repo local-inference-lab/vllm
gh pr view 281 --repo local-inference-lab/vllm

gh pr list --repo vllm-project/vllm --state open --search "expert precision"
gh pr list --repo vllm-project/vllm --state open --search "runtime requantization"
gh pr list --repo vllm-project/vllm --state open --search "expert bitrate"
gh pr list --repo vllm-project/vllm --state open --search "eplb"
```

Without `gh` (REST; `$GH_TOKEN` from `~/.fq_env`, never echo it):

```bash
source ~/.fq_env
curl -s -H "Authorization: Bearer $GH_TOKEN" \
  "https://api.github.com/repos/local-inference-lab/vllm/pulls?state=open&per_page=100" \
  | python3 -c 'import sys,json;[print(p["number"],"|",p["base"]["ref"],"|",p["title"]) for p in json.load(sys.stdin)]'

for n in 277 279 280 281; do
  curl -s -H "Authorization: Bearer $GH_TOKEN" \
    "https://api.github.com/repos/local-inference-lab/vllm/pulls/$n" \
    | python3 -c 'import sys,json;d=json.load(sys.stdin);print(d["number"],d["state"],"merged=",d["merged"],d["title"])'
done

curl -s -H "Authorization: Bearer $GH_TOKEN" \
  "https://api.github.com/search/issues?q=repo:vllm-project/vllm+is:pr+is:open+expert+precision&per_page=10" \
  | python3 -c 'import sys,json;d=json.load(sys.stdin);print("total",d["total_count"]);[print(" ",i["number"],i["title"]) for i in d["items"]]'
```

What the check returned on 2026-08-11 (32 open PRs in
`local-inference-lab/vllm`):

- **#280** open, base `dev/gilded-gnosis` — "[GG] EXL3 R7 native mixed
  K3/K4/K5 runtime". Static load + dispatch; no stats, no re-tiering. Parallel
  contract, 0-conflict replay verified.
- **#279** open, base `dev/gilded-gnosis` — R7 per-(expert, projection)
  checkpoint loading. Load-time only.
- **#277** open, base `feat/gg-r20-exl3-consolidated-20260802` — direct-to-slab
  load. Changes how slabs are *populated*, not the runtime dict shape;
  obsoletes nothing here and needs no adaptation.
- **#281** **closed, merged** into `dev/gilded-gnosis`.
- Upstream `vllm-project/vllm`: nothing doing runtime expert-precision
  re-tiering. The EPLB PRs that come back (#51568, #44987, #37656, #50647,
  #49956, #47588, #49700) are all placement/rebalancing or metrics work — the
  distinction is argued in the PR body and a reviewer will still raise it, so
  be ready to defend it verbally.

If any search returns a genuinely overlapping PR, **stop and reassess** —
`AGENTS.md` says fail closed.

---

## 3. Verify the branch state before opening

Run these and confirm each output.

```bash
cd /home/mbelleau/src/gg-vllm

# a) everything is pushed
git status --short                      # see the note below before you act on this
git rev-parse HEAD                      # expect: 161536085362d5c717fa5f8a94ce70b873bfd1ff
git rev-parse work/fq/m1-stats-collector    # expect: identical to HEAD
git log --oneline work/fq/m1-stats-collector..HEAD   # expect: empty

# b) the base has not moved again
git fetch origin dev/gilded-gnosis
git rev-parse origin/dev/gilded-gnosis
git merge-base origin/dev/gilded-gnosis HEAD    # expect: e2666d9a65f41f...
git rev-list --left-right --count origin/dev/gilded-gnosis...HEAD   # was: 19  24

# c) the merge is still clean (read-only; writes nothing)
git merge-tree $(git merge-base origin/dev/gilded-gnosis HEAD) \
    origin/dev/gilded-gnosis HEAD | grep -c '<<<<<<<'    # expect: 0

# d) the tests still pass
CUDA_VISIBLE_DEVICES="" \
  /home/mbelleau/protensors-work/vllm-voipmonitor/research/fungible-quant/runs/gg-env/gg-run.sh \
  python -m pytest tests/exl3_fungible/ -q --noconftest | tail -1
# expect: 493 passed, 10 skipped, 1 warning in <N>s
```

### Uncommitted work in the tree — decide before you submit

At the time of writing `git status` in `gg-vllm` showed **one uncommitted
modification** that another agent was working on and that is therefore **not
part of the PR**:

```
 M vllm/model_executor/layers/quantization/exl3_fungible/loop.py   (+20 −1)
```

It restricts the loop's decision domain to layers the collector actually
instruments, because GLM-5.2's layer 78 is an MTP layer that the EXL3 loader
*requires* a bitrate entry for but that is never bound as a main-model
`MoERunner` — so the two requirements were unsatisfiable at once and the loop
refused to start.

Decide deliberately: either commit it (and re-run the tests, and mention it in
the body) or leave it out. Do not let it merge in by accident. The verbatim
test result quoted in `PR-BODY.md` was taken **both** on the working tree and
on a detached worktree at the exact head commit; both gave 493 passed, so this
change adds no tests and breaks none.

Only one file is touched by both sides — `vllm/v1/worker/gpu_model_runner.py`,
where the base rewrote 240 lines and we add 14. Git resolves it: our two hunks
anchor on `self.eplb_step()` and `if not skip_eplb:`, both of which still exist
on `fa033bd4e` (now at lines 4768 and 6255). **Do not rebase.** The branch is a
clean linear descendant of the merge base, GitHub will merge it as is, and a
rebase would rewrite 24 commits for no benefit — the reviewer diffs against
the branch, and the merge is already clean.

If you want the #280 coexistence claim re-confirmed at the current commit
count (the recorded replay was run at 11 commits, the branch now has 24):

```bash
git worktree add /tmp/pr280-replay pr280
cd /tmp/pr280-replay && git rebase --onto pr280 \
  e2666d9a65f41fc376607531453cbd57c4c71016 fq/m1-stats-collector
# expect: no conflicts; then run the CPU suite; then
git worktree remove --force /tmp/pr280-replay
```

Do that in a **worktree**, never on the real branch.

If (c) ever returns non-zero, resolve on a scratch branch and re-run (d) before
touching `fq/m1-stats-collector`.

### Then read the diff

`AGENTS.md`: *"The submitting human must review every changed line."* 21,820
insertions is a lot, so use `COMMITS.md` — it groups the 24 commits into seven
themes and marks which files are new versus pre-existing. The six pre-existing
files total **+152 / −6 lines** and are the ones that actually need
line-by-line scrutiny:

```bash
git diff e2666d9a65f41fc376607531453cbd57c4c71016..HEAD -- \
  vllm/config/load.py \
  vllm/model_executor/model_loader/__init__.py \
  vllm/model_executor/layers/fused_moe/router/base_router.py \
  vllm/v1/worker/gpu_model_runner.py \
  vllm/v1/worker/gpu_worker.py \
  vllm/entrypoints/serve/__init__.py
```

---

## 4. Lint

CI here is `pre-commit` plus CodeRabbit; there is **no GPU CI**, which is why
the PR body carries our own measurements.

```bash
cd /home/mbelleau/src/gg-vllm
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -r requirements/lint.txt
pre-commit install
pre-commit run --files $(git diff --name-only e2666d9a65f41fc376607531453cbd57c4c71016..HEAD)
```

Note `AGENTS.md` §2: never bare `pip`, never system `python3` — everything
through `uv` and `.venv/bin/python`. (The `gg-run.sh` invocation used for the
test suite is a separate rootfs environment used for GPU-adjacent work and is
not a substitute for the `uv` venv when running `pre-commit`.)

---

## 5. Assemble the PR body

1. Paste `PR-BODY.md` in full. It is already shaped to the stock upstream
   template — **Purpose / Test Plan / Test Result** — with the
   `AGENTS.md`-mandated sections folded in where a reviewer will look for
   them. Do not paste anything below the template's `---` separator; GitHub
   Actions strips it.
2. **Attach the flagship figure.** GitHub does not render an SVG linked from a
   raw URL inside a PR body, so use the PNG:
   `research/fungible-quant/runs/m5-serve/heatmap/renders/svg-flagship-4axis-1600px.png`.
   Drag-and-drop it into the comment box so GitHub hosts it, then replace the
   "Figures" bullet path with the resulting `user-images.githubusercontent.com`
   URL. (The source SVG is `results/axes/flagship-4axis.svg`; its numeric
   sidecar `flagship-4axis.json` is what the on-figure numbers are read from.)
   - `runs/m5-serve/heatmap/renders/` now exists — 24 light/dark PNG pairs of
     the `/fq/heatmap` page plus `canvas-stats.json`. These are supporting
     material; attach a couple only if a reviewer asks what the page looks
     like. Do not lead with them.
   - Do **not** attach `heatmap/axis-panels.SYNTHETIC.svg` or
     `renders/svg-axis-panels.SYNTHETIC-1600px.png` without the "synthetic"
     label. They are renderer-validation output, not data.
   - Do **not** attach `renders/light-13-DEFECT-stale-dimming-comparison.png`.
     It is a render of a defect, kept for the review trail.
3. Replace the evidence-index paths with permalinks if the evidence repo is
   reachable to reviewers. Pin to a commit SHA rather than the branch name so
   the links do not rot:
   `https://github.com/malaiwah/vllm-voipmonitor/blob/<SHA>/research/fungible-quant/runs/...`
4. File the two separate reports from `SEPARATE-REPORTS.md` **first**, as
   issues in `local-inference-lab/vllm`, then replace the two
   `SEPARATE-REPORTS.md` references in the PR body with the resulting issue
   numbers. Report (b) should be cross-referenced from #280, since it blocks
   #280's K5 support on SM120.

---

## 6. The AI-assistance statement

`AGENTS.md` requires "a clear statement that AI assistance was used". The
wording below is already in `PR-BODY.md` under "AI assistance and
accountability". Use it verbatim, or edit it to be *more* specific — never
less. Do not soften it, and do not move it into a collapsed `<details>` block.

> **AI assistance was used to produce this change.** Claude-based coding
> agents wrote most of the code, the tests and the measurement harnesses, and
> drafted this description, under direction and review.
>
> This is not a pure code-agent PR. A human submitter has reviewed the changed
> lines, ran the tests, operated the serves that produced every runtime number
> here, and will defend the change in review.

The second paragraph is a factual claim about you. **Make it true before you
post it** — §3 "Then read the diff" is that work. `AGENTS.md` states that
pure code-agent PRs are not allowed and that breaching the guidelines "can
result in automatic banning".

---

## 7. Be ready for these four review questions

1. *"Isn't this EPLB?"* — the table in the PR body is the answer. The short
   verbal form: EPLB moves **where** an expert lives and adds copies of it;
   this changes **how many bits** it is stored in, in place, under a fixed
   budget. Also: EPLB counts physical experts post-replication; a bitrate
   policy needs logical identity, which is why the hook is taken before
   `_apply_eplb_mapping`.
2. *"Does this conflict with #280?"* — no. Disjoint file sets, 0-conflict
   replay verified, both quantization contracts coexist in #280's own
   `override_quantization_method`. v1 is scoped to the rank-sliced
   expert-granular path; R7-native swap is declared future work in the body.
3. *"Why should this land when the decode-overhead gate is missed?"* — the
   honest answer, which is in the body: the miss is measured on a proxy whose
   MoE GEMMs are ~150× smaller than the target model's, against a fixed
   per-layer cost, at cc1/cc4 rather than the specified cc8; the GLM-5.2
   measurement has not been taken, so no pass is claimed. Everything is off by
   default, so the cost is not imposed on anyone who does not opt in. If a
   maintainer wants the gate closed before merge, that is a reasonable ask —
   it needs one cc8 A/B on the big model.
4. *"Why aren't your env vars in `envs.py`?"* — they should be, and the PR
   body says so. Roughly 50 `VLLM_FQ_*` variables are read directly from
   `os.environ`, so the engine logs `Unknown vLLM environment variable
   detected` for each at boot. #255 and #186 are the precedent in this repo.
   Offer to add the registrations in this PR or a follow-up; it is mechanical.
   Expect this one — it is the most likely first review comment because it
   shows up in any boot log a maintainer looks at.

---

## 8. Do not do these

- Do not open against `vllm-project/vllm`. Ever. §1.
- Do not open against `local-inference-lab/vllm:main`. §1.
- Do not rebase or force-push `fq/m1-stats-collector`. §3.
- Do not fold #280 into this PR, and do not declare a dependency on it. The
  0-conflict replay is *evidence to cite*, not something to merge.
- Do not delete the retraction banner from
  `runs/m5-serve/results/k3-fq/CONVERGENCE-RESULTS.md` or from the PR body.
- Do not quote GSM8K strict-match (0.116) without flexible-extract (0.892), or
  the reverse.
- Do not present `axis-panels.SYNTHETIC.svg` as measured data.
- Do not claim K5 support on SM120.

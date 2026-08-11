# Demo 1 — does live routing rediscover a human's expert-precision map?

Status as of 2026-08-11 02:50 UTC: **the scoring leg is runnable today; the
live-promotion leg is blocked on one missing artifact** (a mixed K3/K4
checkpoint) and, even once unblocked, is currently limited to 8 of 76 layers
whose only available K4 weights were cut from the reference itself. Read
[What this demo cannot show](#what-this-demo-cannot-show) before quoting any
number from it.

## The claim

Boot GLM-5.2 at a flat 3.0 bpw (all 19,456 routed experts at K3), give the
loader every cached segment plus HF access, cap it at a fixed memory budget,
throw the reference's own calibration corpus and GSM8K at it, and watch the
loop promote experts to K4. The hypothesis: live routing promotes roughly the
same experts a human chose when building
`willfalco/GLM-5.2-EXL3-TR3-3.42bpw` (the "Coder" variant).

If it does, the ~35 GiB of extra bytes people download to get a 3.4 bpw quant
were mostly derivable at runtime from a 3.0 bpw base plus routing observation.

Design, envelope arithmetic and baselines: [`convergence-demo-plan.md`](convergence-demo-plan.md).
Reference map: [`reference-coder-quant.json`](reference-coder-quant.json) —
all 19,456 experts resolved, 11,414 K3 + 8,042 K4, envelope 73.0187 GiB/rank,
chance floor 0.267, human-human ceiling 0.657.

## Commands

```bash
cd /home/mbelleau/protensors-work/vllm-voipmonitor/research/fungible-quant/runs/m5-serve

# 0. Validate every input without touching a GPU. Always do this first.
FQ_DRY_RUN=1 ./run-demo1.sh /home/mbelleau/glm52-mixed-k3k4 demo1

# 1. The real run (needs GPUs 0-3 free).
./run-demo1.sh /home/mbelleau/glm52-mixed-k3k4 demo1

# Variants
FQ_AXIS=axis3_code_agentic ./run-demo1.sh <ckpt> demo1-code   # sharper coder test
FQ_LIMIT=2000  ./run-demo1.sh <ckpt> demo1-short              # shorter replay
FQ_SKIP_EVAL=1 ./run-demo1.sh <ckpt> demo1-noeval             # skip GSM8K
FQ_K4_DIRS=/home/mbelleau/fq-primed/segments-342/expanded:/home/mbelleau/fq-primed/segments-336 \
               ./run-demo1.sh <ckpt> demo1-wider              # larger, less circular K4 pool
```

Everything lands in `results/<tag>/`. The script is `set -euo pipefail` and
verifies each artifact exists before continuing; it refuses to report success
on an empty result directory.

### Knobs

| env | default | what it does |
|---|---|---|
| `FQ_DRY_RUN` | 0 | validate inputs, stop before boot |
| `FQ_PORT` | 8000 | serve port; the run aborts if it is already answering |
| `FQ_K4_DIRS` | `…/fq-primed/segments-342/expanded` | colon-separated dirs searched for K4 fragments |
| `FQ_FILL` | 0.5 | fraction of each layer's K4 pool occupied at boot |
| `FQ_MAX_GIB` | 8.835 | per-rank memory cap above uniform K3 (= the reference's own headroom) |
| `FQ_INTERVAL` / `FQ_DWELL` | 200 / 250 | loop decision cadence, in engine steps |
| `FQ_SIGNAL` | mass | ranking signal for the scorer (`mass` or `count`) |
| `FQ_AXIS`, `FQ_LIMIT`, `FQ_CC` | — | corpus axis filter, prompt cap, replay concurrency |
| `FQ_ALLOW_BUSY_GPU` | 0 | skip the GPU-free check (do not) |
| `FQ_KEEP_STORE` | 0 | reuse a committed policy instead of moving it aside |

## What each step produces, and what it proves

| # | Step | Artifact | What it proves |
|---|---|---|---|
| 0 | preflight | `run.log` | port free, GPUs 0-3 free, signing key present, disk ≥ 20 GB |
| 1 | seeded policy | `policy-demo1.json`, `k4-coverage.json` | every declared K4 slot is backed by a fragment that exists on disk; the budget honours both the reference cardinality and the memory cap |
| 2 | checkpoint gate | `checkpoint-match.json` | the checkpoint physically carries the K4 slabs the policy declares — **the gate that makes promotion possible at all** |
| 3 | store hygiene | `run.log` | no stale committed policy silently overrides the boot policy |
| 4 | boot | `serve.log` | the loop armed (`FQ loop: armed — mode=atomic …`) rather than degrading to collector-only |
| 5 | probe | `probe.json` | the model generates coherent text, i.e. the tier layout is not mis-decoding |
| 6 | startup composition | `composition-start.txt`, `metrics-start.txt`, `policy-boot.json` | the recorded t0 state to diff the end state against |
| 7 | corpus replay | `replay.json`, `timeline.jsonl` | the exact `reap_recall_calib.jsonl` bytes (sha256 `cf247acc…`) were replayed, with counts |
| 8 | GSM8K | `eval-gsm8k/` | the re-tiered model still answers correctly (250-item subsample by default — **report it as a subsample**) |
| 9 | snapshot | `stats.jsonl`, `fq-artifacts/`, `fq-lines.log`, `composition-final.txt`, `policy-final.json`, `metrics-final.txt` | per-interval per-expert routing, the decision log, and the policy the loop actually committed |
| 10 | scoring | `convergence.json`, `SCOPE.md` | mean/pooled Jaccard against the human map, with chance floor and human-human ceiling; `SCOPE.md` states which layers are honest |
| 11 | charts | `swap-timeline.svg` | throughput and tier occupancy over the run, with phase boundaries |

The run ends with a printed verdict line: corpus replayed, mean Jaccard,
lift over chance, fraction of human-human agreement.

## Expected runtime

| Step | Time | Basis |
|---|---|---|
| dry run (steps 0-3) | < 1 s | measured |
| boot to healthy | ~10 min | measured 02:04:38 → 02:14:59 on the K3 checkpoint, warm page cache; a cold 295 GiB boot is longer. The script waits up to 100 min. |
| probe | seconds | measured |
| corpus replay, full 12,228 prompts | ~20-45 min | estimated. Measured: 3,057 code-axis prompts in 155 s, but on truncated stubs (see the replay bug below); full-length rows carry 7.37 M prompt tokens total. |
| GSM8K, 250 items | ~10-25 min | estimated; `MAX_GEN=3072`, concurrency 16 |
| scoring + charts | seconds | measured |
| **total** | **~45-90 min** | dominated by replay + eval |

Set `FQ_LIMIT=2000` for a ~15 min end-to-end smoke run.

## Readiness checklist — verified against the filesystem, not assumed

### Present and working

| Item | State | Evidence |
|---|---|---|
| Reference map | **exists** | `reference-coder-quant.json`, 76 layers, 8,042 K4, self-consistency asserted by `test_score_convergence.py` |
| Calibration corpus | **exists** | `reap_recall_calib.jsonl`, sha256 `cf247acc…` **verified OK**, 12,228 rows, 4 × 3,057 axes, inside the brandonmusic 3.0bpw snapshot — no download |
| Corpus loader | **exists** | `harness/load_mtp78_corpus.py`, runs in 0.14 s |
| Replay driver | **exists, was broken, now fixed** | `replay_mtp78.py` scraped the loader CLI's `--show` output, which prefixes each row with `[line_no] axis source` and truncates to 160 chars, and also swept the summary block in as prompts. It now imports the loader. Regression tests: `test_replay_mtp78.py`. **The earlier `replay-code.json` (3,063 prompts for a 3,057-row axis) was affected — that run's convergence numbers were measured on mangled prompts.** |
| Scorer | **exists** | `score_convergence.py`; dry run on a synthetic 55 %-agreement signal returns 0.4965 vs a 0.2652 floor |
| Policy builder | **exists, extended** | `make_scenario1_policy.py` now discovers K4 coverage per (layer, expert) from segment indexes; 24 tests |
| Charts | **exists** | `make_charts.py`, rendered 44 samples to SVG in the dry run |
| GSM8K harness | **exists** | `harness/eval_gsm8k.sh`, `lm_eval` at `/home/mbelleau/venvs/lmeval/bin/lm_eval`, `openai/gsm8k` in the HF cache |
| Serve script | **exists** | `serve-glm52.sh`; signing pubkey `~/.fq_keys/fq_signing.pub` present, so it will not boot with trust filtering silently off |
| Runner | **exists** | `run-demo1.sh` (this document's subject); both failure and success paths dry-run verified |
| Disk | **178 GB free** | `df /home` |

### Partial

| Item | State | Detail |
|---|---|---|
| K4 fragment coverage | **8 of 76 layers** | Layers 3-10 only, and only as human-derived pools: `/home/mbelleau/fq-primed/segments-342/expanded` (50 experts for L3, 108 for L4-L10) and `/home/mbelleau/fq-primed/segments-336` (50 / 96). |
| Our own K4 encode | **not published** | `/home/mbelleau/fq-0c/work-k4-tr3` holds `layer-003…012.done.json` + TR3 tails, but no publishable `layer-NNN.k4.safetensors` + `index-k4.json`. The campaign supervisor's tier order is `2 5 4` and it is presently on **K2 window 67-74**, so K4 is the *last* tier it will reach. |
| K5 | **24 of 76 layers, and unusable** | Serving K5 as a mixed tier exceeds the SM120 shared-memory limit (109,568 > 101,376). See [`k5-shared-memory-limit.md`](k5-shared-memory-limit.md). K3+K4 is the viable ladder. |

### Absent — these block the live-promotion leg

| Item | State | What it takes |
|---|---|---|
| **Mixed K3/K4 checkpoint** | **absent — the blocker** | `/home/mbelleau/glm52-k3-assembled` is uniform K3 (`tier_bitmap.json`: 19,456 experts, all 3). A promotion is a 1-for-1 trade inside a fixed-capacity mixed slab (`swap.py` `MixedLayerState` requires `tier_bits == (3, 4)` and `_validate_layer` requires occupancy == capacity), so **a uniform-K3 layer has no K4 slab and can never promote**. `run-demo1.sh` step 2 refuses to boot without one. Build it with `fq_assemble.py --segments <k3+k4 family> --source <base> --policy results/<tag>/policy-demo1.json --out /home/mbelleau/glm52-mixed-k3k4 --reflink`, batching like [`assemble-mixed.sh`](assemble-mixed.sh) does. Cost: only the 8 mixed shards are new bytes (~4.6 GB each, ≈ 37 GB); every other shard reflinks from the source at zero cost. |
| Combined K3+K4 segment family | absent | One dir for `fq_assemble --segments`. `make-mixed-inputs.py` does this for K3+K5 by symlinking; the K4 equivalent needs the same treatment over `/home/mbelleau/fq-segments/GLM-5.2-EXL3-FQ` (K3, 76 layers, `rank_sliced_tp4`) and `/home/mbelleau/fq-primed/segments-342/expanded` (K4, 8 layers, `rank_sliced_tp4`). Note `segments-336`'s manifest declares `num_experts: 96`, which is its K4 count, not the model's 256 — `make-mixed-inputs.py`'s field-equality check will reject it as written. |
| K4 in the published segment repo | absent | `malaiwah/GLM-5.2-EXL3-FQ-segments` (454 files) carries K2 × 64, K3 × 76, K5 × 24 layers and **zero K4**. `VLLM_FQ_SOURCES` cannot supply a K4 fragment today. |
| Real gate mass | **landed, unverified on GPU** | Every earlier run ranked on hit frequency: the collector aliased `mass` to `count` because `topk_weights_getter` was never bound, and the live dump had the two arrays byte-identical across 75 × 256 cells. Fixed in gg-vllm `6e08f683d` ("record REAL gate mass, not a copy of the hit count"), which has not yet run on the real model. **Check `mass_is_real` in `stats.jsonl` before quoting a `--signal mass` score**; if it is false you are scoring counts under a mass label. Scoring both ways and reporting the difference is the honest move for the first run after the fix. |

## What this demo cannot show

Three limits. The first two are scope; the third is the one that could
produce a spectacular fake result if it goes unstated.

### 1. Measurement and execution have different coverage

Convergence measurement needs routing counts and the reference bitmap. It does
**not** need K4 weights. So:

> **The overlap metric is computable on all 75 scorable layers (3-77) from a
> plain K3 serve.** Segment coverage does not limit the headline claim.

Materializing and *serving* the rediscovered quant is what is coverage-limited.
So the honest phrasing is:

- *Convergence claim* (does the loop pick the same experts?) — **75 layers, full strength.**
- *Realization claim* (does the rediscovered quant serve and score?) — **8 layers, partial.**

`run-demo1.sh` writes this split into `results/<tag>/SCOPE.md` from the run's
own policy, so the caveat travels with the numbers rather than living only here.

### 2. Live promotion is demonstrated on 8 layers, not 76

Layers 11-77 stay uniform K3 with a zero budget. They contribute routing to the
convergence score and nothing to the promotion evidence. Closing the gap means
encoding K4 for layers 11-77 (≈ 7,236 experts, ~5 GPU-hours single-GPU) — which
is queued behind K2 and K5 in the current campaign.

### 3. The available K4 pool is the reference's own answer — this is circular

This is the sharpest limitation and it is not obvious from the coverage table.

The only K4 fragments on this box were **primed from the human quants
themselves**, so a primed segment contains only the experts that human chose to
put at K4:

| Pool | L3 | L4-L10 | Subset of the reference's K4 set? |
|---|---|---|---|
| `segments-342/expanded` (from the 3.42 reference) | 50 | 108 | **Yes, exactly equal — Jaccard 1.000 on all 8 layers** |
| `segments-336` (from the 3.36 sibling) | 50 | 96 | L3-L5 yes; L6-L10 partially (Jaccard 0.47-0.65) |

Consequences, in order of severity:

1. **With the 342 pool alone, any overlap score on layers 3-10 is 1.0 by
   construction.** The loop can only promote experts the human already picked,
   because no other expert has K4 bytes to promote *into*. That is not
   agreement, it is the absence of an alternative.
2. **At `FQ_FILL=1.0` those layers cannot swap at all.** Budget = pool size
   means zero legal promotion targets, so the loop shows zero swaps forever and
   looks broken. The default `FQ_FILL=0.5` exists to leave trade room.
3. Adding the 336 pool reduces circular layers from 8 to 3 (L6-L10 gain 16-31
   off-reference candidates each), which is better but still a human-derived
   pool.

`make_scenario1_policy.py` computes this at build time, prints a `WARNING`
naming the circular layers, and records `provenance.circular_layers` plus a
`circularity_warning` in the policy. `run-demo1.sh` propagates it into
`SCOPE.md`.

> **Rule: never quote a per-layer Jaccard from a circular layer as evidence of
> convergence.** Promotion there demonstrates the *mechanism* — that experts are
> re-tiered live, under load, through the guards, with the model still answering
> correctly. That is worth showing. It is not evidence that the loop agrees with
> the human.
>
> The non-circular evidence is the offline score over all 75 layers, which is
> computed from routing and needs no K4 weights at all.

The fix is our own K4 encode: fragments encoded from the z.ai BF16 base for all
256 experts per layer, independent of any willfalco allocation. Until then, the
demo has a genuine convergence number (75 layers, offline) and a genuine
mechanism demonstration (8 layers, live), and those are two different claims.

### Also worth stating

- **The reference is one sample.** Four human builds sharing an identical
  calibration manifest agree with each other at J 0.58-0.67. Frame results
  against that band, never against 1.0.
- **The corpus is not code-exclusive.** The 3.25/3.36/3.40/3.42 builds share a
  byte-identical `calibration_manifest.json`; the "Coder" character lives in the
  K-allocation step, not the corpus. We replay the reference's corpus, not a
  coding corpus.
- **Prefill-only routing.** The replay generates 8 tokens per prompt, so routing
  mass is essentially prefill routing — the same regime the reference was
  calibrated in, so the comparison is consistent, but it is not decode routing.
- **GSM8K default is a 250-item subsample** (seed 1234) of 1,319. Enough to
  catch a broken serve, not enough to resolve a 1-2 pp quantization delta. Set
  `ITEMS=0` for the full set.
- **No reference artifact may reach the allocator.** `expert_rel_rt_mse`,
  `tier_bitmap.json` and every willfalco field are outputs of the reference; the
  loop sees only routing from our own serve. `score_convergence.py`'s module
  docstring records why ranking by `expert_rel_rt_mse` would be circular.

## Tests

```bash
cd /home/mbelleau/protensors-work/vllm-voipmonitor/research/fungible-quant/runs/m5-serve
/home/mbelleau/venvs/fq/bin/python -m pytest \
  test_make_scenario1_policy.py test_replay_mtp78.py test_score_convergence.py -q
# 41 passed
```

The policy tests pin the invariant that matters: every K4 slot the policy
declares is backed by a fragment that exists. An unbacked slot does not fail
loudly — `VLLM_FQ_K_FALLBACK=3` serves the K3 bytes instead, so the model keeps
answering while the occupancy table reports a promotion that never happened.

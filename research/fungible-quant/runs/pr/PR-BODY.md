<!--
  PASTE TARGET: local-inference-lab/vllm, base branch dev/gilded-gnosis.
  GitHub will pre-fill the base as vllm-project/vllm:main. That is WRONG.
  See SUBMISSION-CHECKLIST.md before opening.
  Everything below the "---" separator line in the stock template is stripped
  by GitHub Actions; this body deliberately ends before it.
-->

# [GG] Fungible Quant: runtime re-tiering of MoE expert precision (stats → policy → atomic swap)

## Purpose

This adds a runtime control loop that **changes the bitrate of individual MoE
experts on a live serve**, under a fixed memory budget, without a restart and
without a second checkpoint on disk.

The loop is four parts, all in a new self-contained package
`vllm/model_executor/layers/quantization/exl3_fungible/`:

1. **Observe** — `stats.py` histograms `topk_ids` at the router, inside CUDA
   graphs, into a decaying window ring.
2. **Decide** — `policy.py` / `loop.py` rank experts and emit a swap plan
   under dwell, hysteresis, per-layer and total-swap caps, and a **byte**
   budget (not just expert cardinality).
3. **Supply** — `fragments.py` / `progressive.py` resolve the per-expert
   weight fragment at the target K from signed segments (local dir, HF, or
   cache), with a fallback ladder if that K does not exist.
4. **Apply** — `swap.py` trades a K3 expert for a K4 expert in place: rows are
   written into the existing tier slabs and the tier maps are flipped inside a
   quiesce window. No reallocation, no graph recapture.

Plus two operator surfaces, both double-gated behind dev mode *and* their own
env var:

- `admin.py` — `POST /fq/retier` (forced promote/demote, relative or absolute,
  batched, budget-guarded), `GET /fq/state`, `GET /fq/layer/{layer}`;
- `heatmap.py` — `GET /fq/heatmap`, `GET /fq/heatmap/meta`,
  `POST /fq/heatmap/reset`: a read-only layer × expert activation matrix that
  carries its own separate token, so a dashboard need not hold the credential
  that mutates live weights.

### What this is *not*

It is **not another static mixed-K loader**. The bitrate assignment is not
baked into the checkpoint and is not fixed at load time — it is an external
JSON bitmap that the policy rewrites while the server is running. Mixed-K
*loading* already exists in this tree and in #277/#279/#280; this PR consumes
that representation and mutates it. See the duplicate-work section.

### Why this is not duplicating an existing PR

Verified by reading and running the code on those branches, not by reading
their descriptions. Full analysis:
[`runs/rebase/report.md`](#evidence-index) §2.

#### vLLM's EPLB — the closest existing thing, and the strongest challenge

`vllm/distributed/eplb/eplb_state.py` already keeps `expert_load_pass` and a
windowed `expert_load_window` of shape
`(window_size, num_moe_layers, num_physical_experts)`. That is structurally
similar to our stats ring, and a reviewer should raise it. Four differences,
each of which changes what the data is *for*:

| | EPLB | this PR |
|---|---|---|
| what it changes | expert **placement** — replicates hot logical experts as redundant physical experts and rearranges them across devices | expert **precision** — rewrites bits-per-weight in place |
| memory model | more copies of the same weights; capacity grows | fixed budget; a promotion must be paid for by a demotion |
| expert identity | **physical** experts, post-replication and rank-sliced | **global/logical** experts — the only identity a bitrate policy can use |
| signal | hit counts | hit counts **and** gate mass (routing-weight sum), opt-in |
| activation | requires `enable_eplb=True`, which turns on the whole replication/rearrangement machinery | independent; `VLLM_FQ_ENABLE=1` only |

EPLB never changes precision, anywhere. The two are composable, not
alternatives: the stats hook is taken **before** `_apply_eplb_mapping` in
`BaseRouter._select_experts`, precisely so the logical ids survive EPLB's
remap.

#### #280 (EXL3 R7 native mixed K3/K4/K5), #277 (direct-to-slab load), #279 (R7 checkpoints)

Grepping `swap|retier|hotness|expert_stats|promote|demote` across #280's
`exl3.py` returns nothing. Every `prepare_tier` call on those branches
(#280 l.3076, 3109, 3888, 3981, 4027) is load-time. None of the four PRs
collects routing statistics or makes an allocation decision. Their
contribution is static mixed-K **load and dispatch**; ours is what to do with
it afterwards.

The two contracts are parallel and a checkpoint picks one. #280's
`Exl3Config.override_quantization_method` keeps **both** paths
(`exl3.py:1055-1065`): `r7_routed_experts` for
`schema == "r7-complete-v2-checkpoint-v1"`, and `hybrid_tr3_tail` for
`format == "exl3-trellis"` — which is our string, `_RANK_SLICED_FORMAT` at
`exl3.py:342`. Their bitrate map is baked into the checkpoint tensors at
quantization time and is deliberately immutable (validated, sorted, frozen to
`{3,4,5}`); ours is a boot-time indirection so that the policy can rewrite it.
Our external-bitmap validation survives on their branch with identical logic
and error text — it moves from line 498 to line 1219.

**Coexistence, tested rather than estimated.** A trial
`git rebase --onto pr280 origin/dev/gilded-gnosis` in a throwaway worktree
replayed the branch **cleanly, 0 conflicts**, both integration hooks intact,
and the committed CPU suite ran green afterwards
(`100 passed, 1 skipped, 10 deselected`). That replay was run when the branch
carried 11 commits; it now carries 24, all in the same file sets, so the
result should be re-confirmed if it becomes load-bearing. The file sets are
disjoint: #280 touches `config/quantization.py`, `envs.py`, `exl3.py`,
`exl3_online_cache.py`, `model_loader/utils.py`, `models/deepseek_v2.py`,
`warmup/kernel_warmup.py` — we touch none of them.

**We do not depend on #280 and no dependency need be declared.** For the
record, #280 itself is currently blocked below: it calls
`mixed_api.build_projection_tiered_maps`, which exists only on b12x branch
`codex/r7-mixed-trellis-k345-v2-20260810` and is not merged into b12x master.
`build_tiered_maps` — the 2-tier API our adapter uses — survives unchanged on
that branch, so "b12x R7 lands *and* #280 lands" still does not break
`swap.py`.

**Scope limit, stated rather than discovered by a reviewer.** #280 adds a
second slab-assembly site (`exl3.py:4039`) that reuses the same six dict keys
with different semantics: `tier_ids` becomes FC1 slot counts (integers) rather
than global expert-id lists. Our adapter reads them as id lists. On an
R7-native layer it **fails loudly rather than silently** — `tier_bits ==
(3,4,5)` trips our `!= (K3,K4)` guard. v1 is therefore scoped to the
rank-sliced, expert-granular path that #280 preserves; R7-native swap is
future work, and the cost is bounded (generalising tier arity and the slot
identity model in `swap.py`, ~1100 lines, 43 tier0/tier1 references).

#### #281 (InstantTensor borrowed buffers) — merged, and it interacts

#281 merged into `dev/gilded-gnosis` as `5d2079094` after our branch point.
Our code is clean in both directions and this is now pinned by tests (the one
place FQ consumes a tensor it did not produce, `ResolverFragmentSource.
read_expert`, copies at `swap.py:402`). But the audit turned up a
**pre-existing hazard in `exl3.py` that this PR does not fix and should not**
— it affects every EXL3 checkpoint, with or without fungible quant. It is
written up as a separate issue rather than buried here; see
[`runs/pr/SEPARATE-REPORTS.md`](#evidence-index) report (a).

#### Upstream `vllm-project/vllm`

No open upstream PR does runtime expert-precision re-tiering. Search commands
to re-run at submit time are in `SUBMISSION-CHECKLIST.md` §2.

### Footprint on existing files

21,820 insertions, 6 deletions across 46 files. **40 of the 46 files are new
and live under `exl3_fungible/` or `tests/exl3_fungible/`.** Only six existing
files are touched, five of them additively:

| file | +/- | what |
|---|---|---|
| `vllm/config/load.py` | +4 −0 | docstring for the new `progressive` load format |
| `vllm/model_executor/model_loader/__init__.py` | +18 −0 | one `_LOAD_FORMAT_TO_MODEL_LOADER` entry, resolved lazily (an eager import makes the module unimportable standalone) |
| `vllm/v1/worker/gpu_model_runner.py` | +14 −0 | two `if getattr(self, "fq_collector", None) is not None:` calls, at the same cadence as `eplb_step()` |
| `vllm/v1/worker/gpu_worker.py` | +59 −0 | `maybe_init_fq_state`, after `load_model` |
| `vllm/entrypoints/serve/__init__.py` | +31 −0 | two env-gated router attachments inside `register_vllm_dev_api_routers` |
| `vllm/model_executor/layers/fused_moe/router/base_router.py` | +26 −6 | the only non-additive edit — see below |

`base_router.py` already had `capture_fn` and `set_capture_fn` in this tree.
The change makes the callback signature optional-arity: a capture fn tagged
`wants_topk_weights = True` is called as `fn(topk_ids, topk_weights)`, and the
default one-argument contract is unchanged. This is needed because
`topk_weights` is a local in `_select_experts` — it is not stored on the
router, the `MoERunner`, or `FusedMoERouter`, so no getter written against the
router object could have retrieved it. The flag is resolved once in
`set_capture_fn` so the hot path reads a plain bool.

### Everything is off by default

`VLLM_FQ_ENABLE` defaults to 0. `VLLM_FQ_GATE_MASS` defaults to 0 (see the
performance gate below). `VLLM_FQ_ADMIN_API` and `VLLM_FQ_HEATMAP` each
require dev mode **and** their own env var, and their `attach_router` returns
`False` rather than raising if not enabled. With no env vars set, the only
runtime cost is two `getattr` calls per engine step.

---

## Test Plan

### 1. CPU suite — reproducible on any machine, no GPU, no network, no model

```bash
cd <repo> && CUDA_VISIBLE_DEVICES="" \
  python -m pytest tests/exl3_fungible/ -q --noconftest
```

`--noconftest` is required only because the repo `conftest.py` imports a
compiled `vllm._C`; the suite itself has no compiled dependency. 23 test
modules. Coverage includes: the collector contract and its padding-sentinel
handling; policy `decide`/`apply`/`inverse`/`project` algebra; `PolicyStore`
atomic commit and topology-neutrality validation; segment fragment resolution
and trust filtering; the swap engine's commit protocol, rollback and torn-update
fault injection (T5); a four-simulated-rank cross-rank agreement harness (T6);
missing-K hardening (27 tests); memory budget in bytes; the admin endpoint; the
heatmap endpoint; and 12 loader-variant compatibility tests, one of which is a
**control** that proves the fake borrowed-buffer loader really does corrupt a
retaining consumer, so the passing tests around it mean something.

### 2. GPU / serve evidence

There is **no GPU CI in this repo**, so every runtime claim below is our own
measurement, on 8× RTX PRO 6000 Blackwell (SM120), TP4, GG v20-r33 rootfs,
`exl3` quant, `B12X_MLA_SPARSE`, `moe-backend b12x`, `fp8_ds_mla` KV. Raw logs
and scoring JSON are linked in the evidence index.

- **Boot gate** — serve a GLM-5.2 3.0bpw checkpoint assembled by our own tool
  from per-expert segments; probe generation; 120 s throughput at concurrency 8.
- **Quality baseline** — `lm-eval` `gsm8k_cot_zeroshot`.
- **Convergence** — replay each of four corpus axes against the live serve and
  score which experts routing would promote against a human-built reference
  quant (`willfalco/GLM-5.2-EXL3-TR3-3.42bpw`), per-layer Jaccard at the
  reference's own per-layer cardinality.
- **Swap mechanics** — T3 (map mutation under a captured CUDA graph), T4
  (row-write fidelity + bitwise rollback), and a live reload of a whole mixed-K
  allocation on a running HTTP serve.

### 3. What was deliberately not run

GPUs 4–7 were running a quantization campaign throughout, and GPUs 0–3 were
serving. No `instanttensor` or `fastsafetensors` boot was attempted; those
verdicts in the loader matrix are labelled *inferred* and say so.

---

## Test Result

### CPU suite — verbatim

```
$ cd /home/mbelleau/src/gg-vllm && CUDA_VISIBLE_DEVICES="" \
    python -m pytest tests/exl3_fungible/ -q --noconftest
493 passed, 10 skipped, 1 warning in 10.43s
```

All 10 skips are the GPU tests in `test_swap_gpu.py` and `test_swap_t5_gpu.py`
skipping themselves with `requires b12x and an SM120/SM121 GPU`. The 1 warning
is `Failed to read commit hash: No module named 'vllm._version'`, from running
against a source tree without a build stamp. Those GPU tests do pass on the
device — T3/T4/T5 results are in the swap-mechanics table below.

### Boot gate — the assembled checkpoint serves

Source: `runs/m5-serve/m0-boot-gate.md`. `VLLM_FQ_ENABLE=0` — a clean A/B
baseline with the loop off.

```
Model loading took 76.14 GiB memory and 400.935483 seconds
Available KV cache memory: 6.54 GiB
GPU KV cache size: 130,048 tokens
Application startup complete
```

| metric | value |
|---|---|
| requests, 120 s @ concurrency 8, `max_tokens=128` | **208 issued, 208 succeeded, 0 failed** |
| completion tokens | 26,624 |
| aggregate throughput | **219.2 tok/s** |
| median scraped decode rate | 225.6 tok/s (range 108.0–232.0; the minimum is the first scrape interval, i.e. ramp-up) |
| single-stream | 34.9 tok/s (24 prompt + 64 completion tokens in 1.83 s) |

76.14 GiB/rank against a 95.6 GiB card at `gpu-memory-utilization 0.92` is 80%
for weights. That number is load-bearing for the whole design: **promotion has
to come out of a fixed budget, because there is no headroom to grow into.**

A KV-dtype trap was checked rather than assumed. The wrong `ds_mla` variant
"boots fine then emits prompt-independent text", which would invalidate every
number above. Three unrelated prompts at `temperature=0` each produced a
response that restates its own distinct prompt, so decoding is
prompt-dependent on this checkpoint with `fp8_ds_mla`.

### Quality baseline

Source: `runs/m5-serve/results/axes/GSM8K-BASELINE.md`.

| metric | value |
|---|---|
| task | `gsm8k_cot_zeroshot` (lm-eval v3), concurrency 16 |
| items | **250-item subsample, seed 1234** (not the full 1319) |
| **flexible-extract exact_match** | **0.892 ± 0.0197** |
| strict-match exact_match | 0.116 ± 0.0203 |

Read the flexible number. `strict-match` requires a rigid `#### N` form that a
reasoning model almost never emits, so 0.116 measures format compliance, not
arithmetic. Both are reported because quoting only the good one would be
cherry-picking.

This is the **K3 floor**, not a result about re-tiering. At ±2% stderr on 250
items, a 1–2 point move against a future re-tiered run would be inside the
noise; any promotion claim needs the full 1319 or a paired comparison on the
same subsample.

### Convergence — does live routing rediscover a human's expert choices?

Source: `runs/m5-serve/results/axes/FOUR-AXIS-RESULTS.md` and
`results/axes/flagship-4axis.json`.

Each of four corpus axes was replayed separately against the live serve and
scored against the reference's per-layer K4 set, at the reference's own
per-layer cardinality — a pure selection test, not a budget test. 3,057
prompts per axis; **12,228 issued, 12,228 succeeded, 0 failed**; 75/75 layers
scored in every row.

Chance floor **0.2652** (analytic; sampled 0.2641 — they agree). Human–human
ceiling **0.6710**, being the same author's 3.40bpw *non-coder* sibling built
from identical calibration data. Two competent humans agree at ~0.67, so that
is the honest target, not 1.0.

| axis | prompt tokens | wall | **count** | mass |
|---|---:|---:|---:|---:|
| axis1_general | 3,581,199 | 1233 s | 0.3988 (1.50×, 59%) | 0.3730 |
| axis2_legal | 1,100,565 | 399 s | **0.4240 (1.60×, 63%)** | 0.4119 |
| axis3_code_agentic | 2,631,231 | 891 s | 0.4210 (1.59×, 63%) | 0.3770 |
| axis4_reasoning_termination | 201,994 | 127 s | 0.4223 (1.59×, 63%) | 0.4153 |

The same figure also scores the axes against **each other**: mean pairwise
Jaccard 0.424 (range 0.347–0.602) against a 0.265 chance floor. Different
traffic selects substantially different experts — which is the premise the
whole design rests on. A static bitrate assignment is a bet on one traffic
mix.

**Established:** routing frequency alone recovers **63% of the agreement two
humans reach with each other**, at **1.6× chance**, from as little as ~2
minutes of well-chosen traffic, on all 75 layers.
**Not established:** that any of this improves output quality — see "not
demonstrated" below.

### A retracted result, kept visible

An earlier version of the corpus convergence table was **retracted**, and the
retraction is left at the top of
`runs/m5-serve/results/k3-fq/CONVERGENCE-RESULTS.md` rather than deleted.

The replay driver had scraped the corpus loader's `--show` output, which is a
*human display* mode: it prefixes each row with `[line_no] axis source`,
truncates the text to 160 characters, and prints a summary footer to the same
stdout. So the "corpus" runs replayed mangled 160-character stubs with
metadata glued to the front, plus a few footer lines counted as prompts. Two
tells were visible in the recorded evidence and were missed at the time:

- 909,790 prompt tokens / 12,237 prompts = **74 tokens each**, against real
  rows of 2,616 and 4,832 characters (~650+ tokens) — a 9× shortfall;
- the code-axis run reported **3,063 prompts for an axis holding 3,057 rows**
  — the 6 extras were the summary footer.

Retracted: MTP78 full corpus 0.3938 (51 records) and MTP78 code-axis 0.3597
(13 records). Unaffected, because they never went through that loader: the
synthetic rows, 0.3603 at 18 records and 0.3789 at 117 records. The corrected
axis3 replay reads raw text via `iter_prompts` and reports 3,057 prompts /
2,631,231 prompt tokens / 861 tokens per prompt, matching the corpus's real
3,480-character mean; it scored 0.4176, and the later four-axis campaign
scored the same axis at 0.4210 — a 0.0034 spread that sets the scale of
run-to-run noise.

### Negative and surprising results

These go against the hypotheses that motivated the work, and are reported
because they change what a deployment should do.

1. **Hit count beats gate mass, on all four axes**, by +0.007 to +0.044. This
   reverses the hypothesis that motivated implementing real gate mass:
   weighting a route by the router's confidence makes selection consistently
   *worse* here. A plausible mechanism, untested: GLM-5.2's
   `GroupedTopKRouter` renormalises (`norm_topk_prob`), so per-token weights
   sum to a constant, and mass then redistributes emphasis toward experts that
   win their group decisively — while the reference was built to protect
   experts that are *frequently* needed. The capability stays, opt-in via
   `VLLM_FQ_GATE_MASS=1`, because it is the right signal to have measured; the
   honest outcome is that count wins today.
2. **The code axis is not special.** Legal (0.4240), reasoning-termination
   (0.4223) and code (0.4210) sit within 0.003 of each other, inside
   run-to-run noise. Only `general` trails, at 0.3988. The reference is the
   *Coder* variant, so the intuitive prediction was that coding traffic would
   match it best. It does not. What separates the top three from `general` is
   not subject matter but **distinctiveness**.
3. **Volume is nearly irrelevant; concentration is what pays.**
   `axis4_reasoning_termination` scored 0.4223 on 201,994 tokens in 127 s;
   `axis1_general` scored 0.3988 on 3,581,199 tokens in 1233 s — 18× the
   tokens, 10× the wall clock, a worse result. Operationally: a deployment
   needs a *pointed* warm-up, not a long one.
4. **K5 as a mixed tier is hardware-blocked on SM120.** The mixed K3/K5
   checkpoint loads its weights fine (77.83 GiB/rank in 81.8 s) and then all
   four TP workers die during kernel construction:
   `W4A16 shared-memory footprint exceeds device opt-in limit: 109568 > 101376
   bytes (layout=trellis3_t256)`. The footprint grows ~8192 bytes per bit of
   tier width at a fixed tile, which puts K4 at exactly 101376 — the opt-in
   limit, to the byte. So **K3+K4 is the viable mixed ladder on SM120 today**,
   and that is what the headline experiment uses. The cause is a tile-selection
   bug that is not ours and that also affects #280's K5 support; it is filed
   separately, see `SEPARATE-REPORTS.md` report (b). Our K5 *segments* remain
   valid artifacts — this is a runtime kernel limit, not a problem with the
   encoded weights.

### Open performance gate — reported, not rounded away

`exl3_fungible/PERFORMANCE.md` (shipped in this PR) makes measured PP/TG/KLD
impact a standing requirement. Against it, **the M1 decode-overhead gate is
currently NOT MET as measured**
(`runs/m2-dryrun/report.md`): on the small proxy model, median decode tok/s
over repeated fixed-seed runs was 461.64 → 438.36 at cc1 (**+5.04%**) and
1355.74 → 1305.11 at cc4 (**+3.73%**), against a gate of <0.5% at cc8.

Why the proxy is the wrong yardstick, and why this is not yet a verdict on
GLM-5.2: the collector's cost is fixed per layer per step (a handful of small
kernels), while the work it rides on scales with model size — the proxy's MoE
GEMMs are ~150× smaller than GLM-5.2's, so the same fixed cost lands as a far
larger fraction. And the gate is specified at cc8; those runs are cc1/cc4.
**The GLM-5.2 cc8 measurement has not been taken**, so no pass is claimed.
Two identified reductions if the real-model number still misses: fold the
out-of-range guard (~3 of ~6 kernels) into a clamp plus corrective subtract,
and fuse the `count`/`mass` scatters into one `[2, E]` buffer.

Related and measured on CPU: enabling gate mass takes the device kernel count
from 3 to 8 per MoE layer per forward, i.e. 375 extra launches per forward at
75 layers. That is why it is opt-in and off by default.

### Swap mechanics — what is proven, and on what

| test | verdict | detail |
|---|---|---|
| T1 — does a `capture_fn` keep recording inside CUDA-graph replay? | **PASS** | counts grow 10800 → 21520 across replay; all 10 layers agree exactly at two run lengths; constant 16-routing boundary offset. Proxy model. |
| T3 — are the tier maps read as data or baked at capture? | **PASS (bitwise)** | map contents mutated in place under a captured graph; replay output `torch.equal` to a freshly built layer with the new membership; non-vacuous (pre- and post-mutation outputs differ, and the permutation included cross-tier K3↔K4 moves). Toy layer. |
| T4 — row-write fidelity | **PASS (bitwise) ×3** | one expert pair swapped end to end from real segment bytes at slab rows first/last/middle; post-swap slabs, rotation tables and both maps `torch.equal` to a fresh build; `apply(plan.inverse())` restores the pre-swap output bitwise. Toy layer. |
| M3 — live reload of a whole mixed-K allocation on a running HTTP serve | **PASS** | **0.466 s / 0.410 s** total stall, **0 request drops** under continuous traffic, post-reload logits bit-identical to a fresh boot (max \|Δlogprob\| = 0.0, 356 tokens, twice). Restart floor for comparison: 88.0 s. Proxy model. |

Swap window timings (toy layer, E=32, H=I=128): 0.061 ms best for 1 pair,
0.368 ms for 8 pairs, zero device allocations inside the window. Toy payloads
are ~350× smaller per expert than GLM-5.2 rank shards, so these validate the
**fixed overhead** of the window, not PCIe transfer time.

### Artifact integrity

`runs/m5-serve/assembly-report.md`: the pure-K3 checkpoint assembled from
segments is **bit-exact on 81/81 shards** against the published source quant
(935,105 tensors, 0 missing); the mixed K3/K5 build differs on exactly the 12
K5-bearing shards and is byte-identical on the other 69, which is the intended
shape.

### What is NOT demonstrated

Stated plainly so the claim is not read wider than the evidence.

1. **No promotion has run on GLM-5.2.** Every convergence number is
   observe-and-score. No expert was re-tiered on the big model: the serve ran
   `VLLM_FQ_ENABLE=1` with `VLLM_FQ_APPLY_MODE=dryrun`, and the engine log
   carries 688 `FQ interval` decision lines, all reporting 0 swaps. Live
   promotion needs K4 fragments for the served layers; the encode campaign had
   published 8/75 layers at the time of the four-axis runs and 16/75 by the
   GSM8K run.
2. **No quality delta from re-tiering.** GSM8K 89.2% is the K3 floor only.
   There is no re-tiered arm to compare it against, so this PR makes **no
   claim that re-tiering improves output quality**.
3. **T3/T4/M3 are on a proxy model and a toy layer**, not on GLM-5.2. The
   GLM-5.2 leg covers assembly, boot, throughput, GSM8K and observe-only
   convergence.
4. **The decode-overhead gate is unmeasured on GLM-5.2** (above).
5. **TP-only, and the artifacts are TP4-frozen.**
   (`runs/m5-serve/topology-neutrality.md`.) The *policy* layer is topology
   neutral and enforced so — `store.validate_policy` bans `rank`,
   `world_size`, `tp` and `device` keys. The *artifacts* are not: each of the
   four rank slices is an **independent EXL3 quantization** with its own
   H-side rotation, proven by sha256 — the 6144-long un-split-axis vectors
   differ across ranks in all three artifact families measured. TP4→TP8/TP16
   is arithmetically possible and unimplemented; TP4→TP2/TP1 requires
   dequantize → concat → **re-quantize**. A TP mismatch is refused at model
   construction (`exl3.py:1311-1317`), not silently mishandled. The runtime
   loop is correct under TP, degrades to observe-only under PP, and is
   unreachable under EP/DP (EXL3 MoE already raises `NotImplementedError`
   under `use_ep`).
6. **`--load-format progressive` has known gaps**, recorded rather than
   discovered later: `checkpoint_weight_name_prefixes` is ignored; draft/MTP
   models inherit the load format and that path is untested; no
   `secondary_weights`; local directories only. Details and escape hatches in
   `runs/m5-serve/loader-compatibility.md` §4.
7. **R7-native (projection-granular, 3-tier) swap is not supported** — see the
   #280 scope limit above.
8. `instanttensor` and `fastsafetensors` were never booted; those matrix cells
   are inferred from source plus a stated mechanism.
9. **The `VLLM_FQ_*` environment variables are not registered in
   `vllm/envs.py`.** None of them are, so the engine logs
   `Unknown vLLM environment variable detected: VLLM_FQ_...` once per variable
   at boot (`envs.py:2371`), and the strict branch at `envs.py:2369` would
   raise. This repo already has precedent for fixing exactly this — #255
   ("Register runtime controls consumed outside vllm.envs") and #186
   ("Register Gilded Gnosis runtime environment variables") — and this PR
   should follow it. Happy to add the registrations here or in a follow-up,
   whichever the maintainers prefer; it is mechanical and does not change
   behaviour.

### Figures

The flagship figure is the four-axis expert activation panel — 4 axes × 75
layers × 256 experts, one shared pooled-rank column permutation
(`permutation_sha256_12 = aeff214ba2c8`), fixed ±4 log2-relative-to-uniform
domain that is never auto-scaled, with the per-axis overlap-vs-reference
scores and the axis×axis matrix printed on the figure.

- `research/fungible-quant/runs/m5-serve/results/axes/flagship-4axis.svg`
  — **present** (848 KB, `synthetic: false`), with its numeric sidecar
  `flagship-4axis.json`.
- `research/fungible-quant/runs/m5-serve/heatmap/axis-panels.SYNTHETIC.svg`
  — **present**, and is **synthetic**; it exists to validate the renderer and
  must not be presented as data.
- `research/fungible-quant/runs/m5-serve/heatmap/renders/`
  — **NOT PRESENT at the time of writing.** Additional renders were being
  produced in parallel. If that directory exists at submit time, attach its
  contents; if not, ship the flagship SVG alone and delete this bullet.

> GitHub does not render an SVG linked from a raw URL inside a PR body. The
> submitter must drag-and-drop the image into the PR body so GitHub hosts it,
> or convert to PNG first. See `SUBMISSION-CHECKLIST.md` §5.

The figure was adversarially reviewed before being proposed as PR material
(`runs/m5-serve/heatmap/REVIEW.md`, verdict **DEFECTS_FOUND** — three
"pretty and wrong" defects in the page, one in the design's measurements, two
in the endpoint; all six fixed, one further colour issue reported and not
fixed). Two attacks specifically survived:

- **cross-rank double counting — not present.** Under TP4 the gate is
  replicated, so four ranks are four copies of one number and summing would
  inflate every cell 4×. The merge rule is `rank0-canonical`, it is asserted
  by a test that drives four replicas of the real collector, and the rule is
  printed on the figure so an exported image answers the question by itself.
- **normalisation dishonesty — not present.** Scaling a real dump by 71× (the
  actual spread across the archived runs) changes the magnitude panel by
  exactly 0.0 in every one of 19,200 cells, and the compare panel between a
  run and the same run ×71 has max |Δ| = 2.8e-14.

---

## AI assistance and accountability

**AI assistance was used to produce this change.** Claude-based coding agents
wrote most of the code, the tests and the measurement harnesses, and drafted
this description, under direction and review.

This is not a pure code-agent PR. A human submitter has reviewed the changed
lines, ran the tests, operated the serves that produced every runtime number
here, and will defend the change in review. The evidence repository linked
below contains the raw logs, the scoring JSON and the adversarial reviews
behind every claim, including the ones that came out against us — the
retracted convergence table, the missed decode-overhead gate, the gate-mass
result that contradicted its own hypothesis, and the hardware limit that
blocks K5.

## Evidence index

Repository: **`malaiwah/vllm-voipmonitor`**, branch
**`claude/gg-overview-exploration-jchgd3`**, all paths relative to
`research/fungible-quant/`.

| topic | path |
|---|---|
| evidence index for the whole project | `runs/README.md` |
| PR target, base-branch trap, fork hierarchy | `runs/rebase/pr-target.md`, `runs/rebase/fork-hierarchy.md` |
| upstream overlap + duplicate-work analysis | `runs/rebase/report.md` |
| boot gate, throughput, KV-dtype check | `runs/m5-serve/m0-boot-gate.md` |
| GSM8K baseline | `runs/m5-serve/results/axes/GSM8K-BASELINE.md` |
| four-axis convergence | `runs/m5-serve/results/axes/FOUR-AXIS-RESULTS.md` |
| retraction + corrected convergence | `runs/m5-serve/results/k3-fq/CONVERGENCE-RESULTS.md` |
| K5 shared-memory limit | `runs/m5-serve/k5-shared-memory-limit.md` |
| loader-variant compatibility matrix | `runs/m5-serve/loader-compatibility.md` |
| topology neutrality | `runs/m5-serve/topology-neutrality.md` |
| missing-K hardening | `runs/m5-serve/missing-k-hardening.md` |
| admin API spec | `runs/m5-serve/admin-api-spec.md` |
| checkpoint assembly + bit-exactness | `runs/m5-serve/assembly-report.md` |
| heatmap design + adversarial review | `runs/m5-serve/heatmap/DESIGN.md`, `runs/m5-serve/heatmap/REVIEW.md` |
| gate mass: why it was aliased, what binds it | `runs/m5-serve/gate-mass-binding.md` |
| M1 overhead + M2 dryrun | `runs/m2-dryrun/report.md` |
| graph-freeze (T1) | `runs/t1-graph-freeze/report.md` |
| swap engine T3/T4 | `runs/m4-swap/report.md` |
| live reload (M3) | `runs/m3-reload/report.md` |
| commit-by-commit reading guide | `runs/pr/COMMITS.md` |
| the two findings filed separately | `runs/pr/SEPARATE-REPORTS.md` |

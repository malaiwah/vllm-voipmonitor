# Convergence demo — can the runtime loop rediscover a human's expert-precision map?

Date: 2026-08-11. Status: **inputs gathered, demo is feasible — with one scope
split that must be stated up front.**

## The claim under test

A human (willfalco) built a 3.42 bpw GLM-5.2 EXL3 "Coder" quant by choosing,
offline, which of 19,456 routed experts get 4 bits instead of 3. We boot a plain
all-K3 3.0 bpw GLM-5.2, hand the loader **the same memory envelope**, replay
**the same corpus**, and let the runtime fungible-quant loop pick its own K4 set
from observed routing. If our set overlaps the human's set well above chance,
then the ~35 GiB of extra bytes people downloaded to get a 3.4x bpw quant were
mostly *derivable at runtime* from a 3.0 bpw base plus routing observation.

## Inputs (both found, both verified)

### 1. Reference quant — `willfalco/GLM-5.2-EXL3-TR3-3.42bpw`

Pin: `ae68c65947efa90bea37308e15421872f124c46d`. Full machine-readable map:
[`reference-coder-quant.json`](reference-coder-quant.json).

It **is** the Coder variant — this is stated, not inferred:

- Its own README H1: *"GLM-5.2 EXL3 TR3 3.42 bpw Coder"*, *"with Coding expert
  allignments from [3.25bpw]/[NF3]"*.
- The 3.40 bpw sibling README: *"3.42bpw is Coder version, inheriting less busier
  Coder experts at higher bpw."*
- The 3.36 bpw sibling README: *"superseded by 3.42bpw Coder version."*

**Correction to the demo's framing.** The Coder character does **not** come from
a coding-focused calibration corpus. The `calibration_manifest.json` of the
3.25 / 3.36 / 3.40 / 3.42 willfalco builds *and* brandonmusic's 3.5 bpw build are
**identical**, field for field — same `corpus_sha256` (`cf247acc…`), same
`capture_fingerprint` (`2efd1027…`), same 4-axis row counts, same passes. The
corpus has a code axis (3,057 of 12,228 rows) but is not code-exclusive. The
"Coder" difference lives entirely in the **K-allocation step**, inherited from
the 3.25 bpw Coder build and `madeby561/GLM-5.2-MXFP8-NVFP4-NF3-Hybrid`. The
demo is unaffected — we still replay the corpus the reference was calibrated on
— but we must not claim we are replaying a *coding* corpus.

### 2. Corpus — `reap_recall_calib.jsonl`

sha256 `cf247acc7c5da9f0600c7d6ab3b7c2fcfc54ec30b794e3b6047559285fa44df4`,
12,228 rows, 34,002,059 B. **Already on local disk**, inside the
brandonmusic K3 snapshot — zero download.

Loader: [`harness/load_mtp78_corpus.py`](harness/load_mtp78_corpus.py), verified
(12,228 items, hash OK, 4 × 3,057 axes). It reproduces `drive_corpus.py`
semantics exactly: raw `text`, checkpoint tokenizer, truncate 4096,
`max_tokens=1`, `temperature=0`, `ignore_eos`, plus the `%128` small-final-chunk
trim.

The provenance chain closes: the 3.42 reference's `calibration_manifest.json`
cites this exact sha256, the MTP78 capture dataset was driven from this exact
file, and the 3.42 model card credits
`malaiwah/GLM-5.2-MTP78-calibration-capture` for its layer-78 routed experts.
The operator's own dataset is *inside* the reference artifact.

> Two-layer note. "MTP78 activation corpus" can mean the recorded activations
> (`malaiwah/GLM-5.2-MTP78-calibration-capture`, 7,288,310 rows of layer-78
> hidden states + ground-truth top-8 ids) or the prompts that produced them.
> The loader serves the **prompts**, because the recorded capture covers
> **layer 78 only** and convergence needs routing on layers 3–78. The capture is
> exposed via `iter_capture_shards()` for layer-78 work and is not downloaded by
> default.

## The reference expert set — what we are trying to rediscover

| Scope | K3 | K4 | Total |
|---|---|---|---|
| Layer 3 | 206 | 50 | 256 |
| Layers 4–77 (74 layers) | 148 × 74 = 11,952 | 108 × 74 = 7,992 | 18,944 |
| Layer 78 (MTP draft) | 256 | 0 | 256 |
| **Model** | **11,414** | **8,042** | **19,456** |

76 MoE layers (3–78), 256 experts each. Every layer 4–77 has the identical
*count* (148/108) but a **different membership** — that membership is the target.

Measured bpw: trellis-K mean **3.413343**; from bytes, **3.415957** (expert
payload) / **3.416079** (incl. shared-H rows). Config declares
`expert_bpw_mean = 3.418854`.

**How the map was determined — three independent agreeing methods:**

1. **Metadata (primary).** `tier_bitmap.json` → per layer, `k` = 256-entry list
   of 3|4. `config.json` declares `bits_per_expert = "tier_bitmap.json:k"`.
   Byte-identical at both the pinned and current revisions.
2. **Header-verified (independent).** Ranged reads of safetensors headers for
   layers 0, 3, 4, 5, 30, 77, 78: each expert's 12 trellis tensors have last dim
   `16*K` (48 = K3, 64 = K4), all 12 agree within an expert, and per-layer counts
   matched the bitmap **exactly**. Per-expert unit sizes reproduce header totals
   to the byte (L4: `148×3,542,028 + 108×4,721,676 = 1,034,161,152`).
3. **MSE-band (independent).** `expert_rel_rt_mse` splits into two perfectly
   separable bands — K4 `0.004539–0.008291`, K3 `0.017867–0.031967` — matching
   this project's own ε ladder (K3 0.0231 / K4 0.0060), and K4set == lowest-mse-N
   in **75/75** layers.

Confidence: **HIGH — exact per-expert identity for all 19,456 experts. Not partial.**

> Trap to avoid: `expert_rel_rt_mse` is the **achieved** error at the shipped K,
> an *output*. It is not the human's selection criterion and must never be used
> as the reference signal — doing so would make the demo circular.

## Memory envelope

Exact unit sizes (`shared_h_v1`, verified against headers):

| Tier | Per expert | **Per TP4 rank** |
|---|---|---|
| K2 | 9,449,520 B | 2,362,380 B (2.253 MiB) |
| K3 | 14,168,112 B | 3,542,028 B (3.378 MiB) |
| K4 | 18,886,704 B | 4,721,676 B (4.503 MiB) |
| K5 | 23,605,296 B | 5,901,324 B (5.628 MiB) |
| K3→K4 promotion | +4,718,592 B | **+1,179,648 B** |

**The envelope: 78,403,227,648 B = 73.0187 GiB per rank** (expert payload
78,400,425,984 B + shared-H 2.672 MiB). Reference repo total 351.6 GB / 327.42 GiB;
non-expert bytes in layer shards 31.72 GiB; embed 1.77 GiB; head 1.77 GiB — all
constant across scenarios and excluded from the contested budget.

Self-consistency check that validates the whole accounting:

```
headroom over all-K3      = 9,486,729,216 B/rank = 8.835 GiB
promotion cost            = 1,179,648 B/rank
headroom / cost           = 8042  ==  the reference's own K4 count   ✓
```

The loop gets **exactly enough budget for 8,042 promotions and not one more** —
so this is a pure selection problem, not a budgeting problem. That is the
cleanest possible experimental setup.

> **Layout caveat.** brandonmusic 3.0bpw is `per_expert_v1`: its K3 expert is
> 14,315,568 B (= 14,168,112 + 147,456 replicated H rows), giving 64.849 GiB/rank
> vs 64.183 GiB/rank for `shared_h_v1` all-K3. Compare envelopes in **one**
> layout or the 147,456 B/expert (2.67 GiB total) gets mis-attributed as headroom.

## How we measure convergence

**Metric: per-layer Jaccard** between our promoted set `Ours(L)` and the
reference `Ref(L)`, over layers 3–77 (75 layers):

```
J(L) = |Ours(L) ∩ Ref(L)| / |Ours(L) ∪ Ref(L)|
```

Report the mean over layers **and** the pooled Jaccard (sum intersections / sum
unions) — the pooled figure resists small-layer distortion. Exclude layer 78: it
is uniformly K3 and contributes a degenerate empty set. Score layer 3 separately
(budget 50, not 108).

Because the budget forces `|Ours(L)| = |Ref(L)|`, Jaccard, precision, and recall
are monotone transforms of each other; publish raw intersection counts too.

**Baseline 1 — chance.** Uniform random selection of the same cardinality:

| Layer class | E[intersection] | E[Jaccard] |
|---|---|---|
| Bulk (108 of 256) | 45.6 | **0.2673** |
| Layer 3 (50 of 256) | 9.8 | **0.1082** |

Run it empirically too (≥1,000 permutations per layer) to get a null distribution
and a p-value, not just an expectation.

**Baseline 2 — human vs human (the meaningful ceiling).** Three sibling builds
share the *identical* calibration manifest and differ only in allocation:

| Pair | Mean per-layer J | Pooled J |
|---|---|---|
| 3.42 Coder vs **3.40** (non-coder, nearest bpw) | **0.6572** (sd 0.0966, range 0.462–1.000) | 0.6497 |
| 3.42 Coder vs 3.36 | 0.6710 | 0.6635 |
| 3.42 Coder vs 3.25 Coder | 0.5761 | 0.5747 |
| *chance* | *0.2673* | *0.2673* |

This is the single most valuable number we found. **Two competent humans working
from the same calibration data at nearly the same bitrate agree at J ≈ 0.66,
against a chance floor of 0.27.** So:

- J ≈ 0.27 → the loop learned nothing.
- J ≈ 0.45 → real signal, clearly beating chance.
- **J ≈ 0.65 → the loop is indistinguishable from a human expert.** This, not
  1.0, is the honest target. Claiming we should hit 1.0 would be claiming the
  reference is uniquely correct, which the sibling spread disproves.

**Baseline 3 — a stronger straw man.** Also score a *static* heuristic that uses
no routing (e.g. rank by expert weight norm from the BF16 base). If routing
observation does not beat it, the claim is about the heuristic, not the loop.

**Sharper sub-target — the coder signature.** Against the 3.40 general sibling,
the Coder build moves **1,824 experts into K4** and **1,528 out**. Overlap
restricted to those 1,824 "coder-only" experts tests whether the loop picks up
the *coding-specific* allocation rather than generic expert importance. Per-layer
lists are in `reference-coder-quant.json → coder_vs_general_delta`. Optionally
drive only `axis3_code_agentic` (3,057 rows) and check the coder-only overlap
rises relative to a general-axis drive — that would be the strongest result
available, and it isolates the one thing the corpus-identity caveat above costs us.

## Scenarios

**Scenario 1 (primary).** Base = all-K3, tiers {K3, K4}, envelope 73.0187 GiB/rank.
Headroom = exactly 8,042 promotions. Direct comparison to the reference.

**Scenario 2.** Base = **all-K2**, tiers {K2, K3, K4, K5}, **same** envelope.
Headroom over all-K2 = **32,437,960,704 B/rank = 30.210 GiB** — far larger, and
spendable non-uniformly:

| If spent only on | Experts affordable | Share of 19,456 |
|---|---|---|
| K2→K3 | 27,498 | 141% (i.e. all, with slack) |
| K2→K4 | 13,749 | 70.7% |
| K2→K5 | 9,166 | 47.1% |

Uniform K2 = 42.806 GiB/rank and uniform K3 = 64.181 GiB/rank both fit; uniform
K4 (85.556) and K5 (106.931) do not. Scenario 2 has **no human reference** — the
reference only uses {3,4} — so score it on quality (KLD vs BF16) against the
3.42 reference at equal bytes, not on overlap. The interesting question there is
whether a K2/K5 mix beats a K3/K4 mix at identical cost.

## Risks and honest scope

**The one that matters: measurement and execution have different coverage.**

Convergence measurement needs (a) routing counts per expert per layer, and
(b) the reference bitmap. Neither needs K4 *weights*. So:

> **The overlap metric is computable on all 75 scorable layers (3–77) from a
> plain K3 serve.** Segment coverage does not limit the headline claim.

Materializing and *serving* the rediscovered quant is what is coverage-limited.
Published segment inventory (`malaiwah/GLM-5.2-EXL3-FQ-segments`, 438 files):

| Tier | Layers | Count |
|---|---|---|
| K3 | 3–78 | **76/76 complete** |
| K2 | 3–58 | 56/76 |
| K5 | 3–10, 35–50 | 24/76 |
| **K4** | **3–10** | **8/76** (from community priming: willfalco 3.36 + 3.42) |

So the **end-to-end serve leg runs on layers 3–10 only** — 806 of 8,042 reference
K4 experts (10.0%). On the other 67 layers the assembled checkpoint falls back to
K3 (`VLLM_FQ_K_FALLBACK` marks them). State the split explicitly in the writeup:

- *Convergence claim* (does the loop pick the same experts?) — layers 3–77, full strength.
- *Realization claim* (does the rediscovered quant serve and score?) — layers 3–10, partial.

Closing the gap means encoding K4 for layers 11–77 (≈7,236 experts). At the
measured 2.48 s/expert K4 encode that is ~5.0 GPU-hours single-GPU, ~40 min on
8 GPUs — cheap, but it competes with the running campaign for disk and GPUs.

**Other risks:**

- **Layout mismatch.** Reference is `shared_h_v1`; our base is `per_expert_v1`.
  Already solved and *measured*: the expansion is bit-exact (reconstruction table
  row 5, 2048/2048 experts, cos 1.00000). Do not re-litigate; do keep the
  147,456 B/expert out of the envelope arithmetic.
- **Layer 3 is special** (50 K4, not 108) and layer 78 is uniformly K3. Hard-code
  the per-layer budget from the reference rather than assuming 108.
- **Routing observation must be armed.** `t1-graph-freeze` found three prior runs
  were hollow because the collector binding is gated on
  `enable_return_routed_experts`. Verify non-zero, growing counts before trusting
  any allocation.
- **Corpus drive is prefill-only, 1 token out.** Routing mass is prefill routing.
  If the loop is meant to serve decode traffic, note the distribution difference;
  the reference was calibrated the same way, so for *this* comparison it is
  consistent.
- **The reference is one sample.** The sibling spread (J 0.58–0.67 among humans)
  shows there is no unique right answer. Frame results against the human-vs-human
  band, never against 1.0.
- **Circularity.** Do not feed `expert_rel_rt_mse`, `tier_bitmap.json`, or any
  willfalco artifact into the allocator. The loop must see only routing from our
  own K3 serve. Worth an explicit assertion in the run script.

## Verdict

**Feasible as designed, for the claim that matters.** Both inputs are in hand and
verified to the byte; the envelope arithmetic closes exactly on the reference's
own K4 count; the metric has both a chance floor and a human-competence ceiling
from four independent human builds. The one scope limit is that *serving* the
rediscovered checkpoint is currently confined to layers 3–10 — which constrains
the quality follow-up, not the rediscovery result.

Two framing corrections to carry into the demo: the reference's calibration
corpus is shared with its non-coder siblings (the Coder-ness is in the allocation
step), and success means landing in the human-vs-human band around J ≈ 0.65,
not at 1.0.

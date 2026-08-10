# 12 — Phase 0a: Expert-Routing Stability on the Real Layer-78 Capture

**Status:** measured, 2026-08-10. CPU-only, zero GPU, ~58 MB downloaded.
**Script:** `research/fungible-quant/poc/poc_0a_stability.py` (pure numpy; scipy/pandas not on box).
**Verdict in one line:** the *aggregate* top-108 allocation converges within ~2% of the corpus (τ > 0.9 after ~150k tokens, set exact after 75%) — consistent with K3 firing — but individual ~73k-token windows sit **below** the 0.9 bar (adjacent τ mean 0.830), and the rank-108 cut lands on a frequency plateau, so short-horizon set membership is intrinsically fuzzy.

## 1. Data

`hf.co/datasets/malaiwah/GLM-5.2-MTP78-calibration-capture` (revision `main`, read 2026-08-10):

| file | size | tensors | ids byte-range read |
|---|---|---|---|
| capture-00001-of-00003.safetensors | 43,036,000,664 B | `x` BF16 [3.5M, 6144]; `ids` U8 [3.5M, 8] | 43,008,000,664–43,036,000,663 (28.0 MB) |
| capture-00002-of-00003.safetensors | 43,036,000,672 B | same shapes, `row_offset=3500000` | 43,008,000,672–43,036,000,671 (28.0 MB) |
| capture-00003-of-00003.safetensors | 3,545,060,424 B | `x` [288310, 6144]; `ids` [288310, 8], `row_offset=7000000` | 3,542,753,944–3,545,060,423 (2.3 MB) |

`ids` is a **separate tensor at the tail of each shard** — no need to touch the 89.6 GB of hidden states. Acquisition = the `poc_slice.py` technique: range-read 8-byte LE header length, range-read the JSON header, then one coalesced range read per shard for `ids` only (58.3 MB total). Shards concatenated by `row_offset` → **7,288,310 tokens × top-8 = 58,306,480 routing slots** over **256 routed experts** (all 256 observed; no duplicate expert within any token's top-8; dtype U8).

Provenance (shard metadata): GLM-5.2 753B via `GLM-5.2-EXL3-TR3-3.0bpw`, layer prefix `model.layers.78.mlp`, capture `mtp78_xcapture v3 device-ring, live serving, prefill-driven full corpus + MTP draft rows, 2026-07-25`, corpus `reap_recall_calib.jsonl` (12,228 rows, **4 axes**).

## 2. Method

- Token stream split into **100 sequential windows** of ~72,883 tokens (`np.array_split`, order = capture order ≈ corpus order with interleaved MTP draft rows).
- Per window: `bincount` over the 8 slots/token → count vector c ∈ ℕ²⁵⁶; frequency = c / (8·window_tokens).
- **Kendall τ-b** (tie-corrected, O(E²) pairwise, E=256) between count vectors; **Jaccard** between top-108 sets (108 = the K4-tier population per layer at the 3.42 bpw operating point, PLAN §"N = 108").
- Cumulative convergence: counts of the first X% of the stream vs the full corpus, X = 1..100.
- Concentration: Gini (sorted-index formula), Shannon entropy, top-k mass shares.
- **Gate mass is NOT computable**: the dataset stores hidden states + routed ids only, no gate values. PLAN 0a asks for frequency **and gate mass**; this is the frequency half only. (HOBBIT-style LHU / gate-norm criticality needs a collector change — fold into 0b.)

## 3. Results

### (a) Adjacent windows (99 pairs, ~73k tokens each)
- top-108 Jaccard: **mean 0.794**, median 0.800, min 0.701, p5 0.728, max 0.878.
- Kendall τ-b (full 256 ranking): **mean 0.830**, median 0.836, **min 0.723**, p5 0.782, max 0.890.
- Worst pairs at windows 35, 64, 11, 1, 34 (τ 0.72–0.78) — consistent with axis boundaries in the 4-axis corpus.
- τ restricted to the global top-108 experts only: mean **0.685**, min 0.541 — *within* the head the ordering churns even more, because head frequencies are nearly flat (see (d)).

### (b) Window vs global ranking
- top-108 Jaccard: mean 0.849, min 0.771, max 0.912.
- τ-b: **mean 0.880**, min 0.813, max 0.920; only **16% of windows exceed τ = 0.9**.
- Top-108 membership churn: **67 experts are in the top-108 of every window**; **167** appear in at least one window's top-108. A stable core of ~67 plus a contested band of ~100 candidates for ~41 seats.

### (c) Cumulative convergence (first X% vs full corpus)

| X% | tokens | J(top-108) | τ-b |
|---|---|---|---|
| 1 | 72,883 | 0.831 | 0.899 |
| 2 | 145,766 | 0.862 | **0.917** |
| 5 | 364,420 | 0.895 | 0.946 |
| 10 | 728,840 | 0.946 | 0.952 |
| 20 | 1,457,670 | **0.982** | 0.975 |
| 50 | 3,644,160 | 0.982 | 0.984 |
| 75 | 5,466,235 | **1.000** | 0.992 |
| 100 | 7,288,310 | 1.000 | 1.000 |

τ crosses 0.9 at ~150k tokens; J(top-108) reaches 0.98 by 1.5M tokens and is **exact** from 75% onward. First half vs second half of the corpus: **J(top-108) = 0.982, τ = 0.970** — macro-scale allocation is highly stable within this corpus.

### (d) Concentration / skew
- **Gini = 0.308** (per-window mean 0.314), entropy **7.757 / 8.000 bits** — the router is well load-balanced.
- max/min frequency = 0.01987 / 0.00015 (ratio 136; one near-dead expert, but the bulk is near-uniform).
- Mass coverage: top-8 = 10.6%, top-32 = 27.3%, top-64 = 44.2%, **top-108 = 63.2%**, top-128 = 70.6%.
- The rank-108 cut sits on a **plateau**: frequency falls only from 4.26‰ (rank 90) to 3.60‰ (rank 128); the relative gap between rank 108 and 109 is **0.84%**. Top-108 *set* membership near the cut is therefore noise-dominated by construction — Jaccard penalties in (a)/(b) are largely boundary flicker, not wholesale re-ranking.

## 4. Interpretation vs kill-criterion K3 (τ > 0.9 ⇒ allocation stable, Phases 3–5 die)

1. **What this corpus can say:** within a fixed workload mix, the allocation **converges fast and stops moving** — τ > 0.9 vs the final ranking after ~2% of the stream, half-vs-half τ = 0.97, exact top-108 recovery from 75%. This is the MoE-Infinity "reuse counts even out" pattern and points toward **K3 firing**: a one-time warmup of a few hundred thousand tokens pins the top-108 set for that mix; continuous rebalancing machinery adds little.
2. **What it also says:** at a ~73k-token horizon, τ is 0.83 (min 0.72) — **below** the 0.9 bar. Any online policy with a window that short would chase domain noise; the stats window must aggregate ≥several hundred k tokens (empirically ~150k for τ > 0.9) before acting.
3. **What it cannot say (honest limits):**
   - This is a **4-axis calibration mix replayed in corpus order** (plus live-MTP draft rows), captured on **one day**. Window-to-window deltas measure *domain shift across the mix*, not temporal drift of live traffic. K3's literal test — τ across days/weeks of production traffic — remains unmeasured and needs a live capture.
   - **Layer 78 only**, and layer 78 is the **MTP draft head** (see 0b note in PLAN). Generalization to the other ~77 MoE layers is exactly what Phase 0b must check.
   - **Counts only, no gate mass** — the promotion signal PLAN §4 prefers (gate mass / LHU) is not in this dataset.
4. **Caution flag for K1:** the distribution is near-uniform (Gini 0.31, entropy 97% of max) and a frequency-picked K4 tier of 108 experts covers only **63.2%** of routed mass. Frequency-only specialization has limited headroom on this layer; the case for mixed precision must rest on per-expert *sensitivity* (0c) and gate-mass weighting, not on routing skew.

**Build note (2026-08-10) — 0c answers §4's caution flag.** This report ends
by warning that "the case for mixed precision must rest on per-expert
*sensitivity* (0c) and gate-mass weighting, not on routing skew." The 0c
proxy leg (`../runs/0c-campaign/report.md`) found the opposite emphasis and
a happier result: per-expert Δε is nearly uniform (median CV **0.047**) —
sensitivity alone would not justify per-expert allocation — but
**benefit = Δε · φ is strongly concentrated (median Gini 0.48**, median
top-16-of-256 share **0.318**), and the concentration is *driven by routing
skew*. So the per-expert premise holds, and it holds because of routing
mass, measured on a **non-MTP** layer set (10 MoE layers of the Fruit proxy)
rather than layer 78. The K2 abort criterion does not fire.

**Recommendation:** treat 0a as *supporting* the expected outcome — "allocation converges; startup specialization (Phase 1) captures the value" — but do not close K3 on this evidence alone. Add gate-value capture + a multi-day live-traffic slice to the 0b collector run; re-run this exact script on that trace (it is workload-agnostic).

## 5. Reproduce

```bash
python3 research/fungible-quant/poc/poc_0a_stability.py   # caches ids to ~/.cache/fq_phase0a (58 MB), ~2 min analysis
```

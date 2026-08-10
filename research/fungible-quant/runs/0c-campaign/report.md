# Phase 0c — per-expert sensitivity campaign — PROXY LEG COMPLETE — 2026-08-10

Fruit proxy (256 experts, 10 MoE layers — same expert cardinality as
GLM-5.2), executed end-to-end on this box. Full-model leg launched
separately (layer-streaming BF16 capture, `capture-stream-report.md`).

## Pipeline (all committed, all resumable)

1. **Capture**: transformers-based prefill capture (`capture_hf.py`) —
   1,050,468 tokens × 10 layers, 20.11 GiB, 4-axis owner corpus, sealed
   fingerprint `c338b547…`; routing verified **0/10.5 M mismatches** vs
   model topk.
2. **Encodes**: the sha-pinned production encoder (`encode_tr3_v31.py`
   `e9a85a47…`) driven at K∈{2,3,4,5} from the SAME capture
   (hessian-identical siblings), 40 layer-encodes, per-expert error stats
   in every done-JSON. ~0.5 s/expert on SM120.
3. **Artifacts**: 4 assembled checkpoints (`fruit-k{2,3,4,5}`) + the 4-K
   segment family (`fruit-segments/`, per-K indexes + attestations,
   `encode-of` lineage w/ capture fingerprint).

## Results (fq_eps, `eps-analysis.json`)

| K | mean ε (rel RT MSE) | step improvement |
|---|---|---|
| 2 | 0.09027 | — |
| 3 | 0.02310 | 3.91× |
| 4 | 0.00602 | 3.84× |
| 5 | 0.00159 | 3.79× |

**A clean geometric ladder: ~3.8× error reduction per +1 bit.** The
multi-K bet quantified: K2 base costs ~3.9× vs K3 (progressive fast-load
tier), K5 buys ~3.8× over K4 for hot experts.

- **K2-abort (04-milestones): does NOT fire.** Δε/expert is uniform-ish
  (median CV 0.047) but benefit = Δε×φ is strongly concentrated —
  **median Gini 0.48**, driven by routing skew (top-16 experts of 256
  carry an outsized benefit share per layer). Per-expert allocation is
  the right granularity; the win comes from routing mass.
- **Global solve vs uniform N_L**: +1.3–2.8% at layer level on the proxy
  (10 architecturally-similar layers → modest; the full model's 75
  heterogeneous layers is where layer-level allocation should matter).
  Solve output is genuinely non-uniform: n_k4_per_layer 42…125 at the
  0.42 budget.
- **First solve-derived policy minted**: `policy-fruit-mixed-042.json`
  (fq-policy/2, manifest-bound, capture-fingerprint provenance) — feeding
  the mixed-boot gate now.

## Knob feed-ins (01 §6)

- ε source for the policy engine: encoder-emitted per-expert rel-RT-MSE
  (validated ladder) — measure_model-style dKL deltas unnecessary for v1.
- N_L: solve-derived per-layer counts (proxy evidence); full-model solve
  after the streaming campaign.
- K ladder economics for M6/multi-K: each ±1 bit ≈ ×/÷3.8 error at
  ±33% bytes (K3→K4).

## Residual (full-model leg)

Layer-streaming BF16 capture running (layers 3–40 first pass); 4-K
full-model encodes + solve follow per MULTI-K-PLAN (opportunistic,
hottest-first, rolling publish).

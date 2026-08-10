# 0f(ii) — quantize_exl3 K3 vs K4 encode benchmark — 2026-08-10

Rig: one real GLM-5.2 expert (layer 30, expert 137) from `zai-org/GLM-5.2`
@ `b4734de4`, BF16 → f32, transposed to exllamav3's (in, out) convention.
Synthetic Hessian (X~N(0,1), 4096 samples, count>0 → real LDL + trellis
path), `mcg: True` (the codebook mixed execution is pinned to). GPU 4
(RTX PRO 6000 Blackwell, extracted r33 env). Median of 3, warmup discarded.
Timing representative; quality not measured here (0f(ii) is documentation
only per 01 §6). `results.json` alongside.

| Tensor (in→out) | K3 cold | K3 warm-H | K4 cold | K4 warm-H |
|---|---|---|---|---|
| gate_proj 6144→2048 | 0.92 s | 0.85 s | 0.87 s | 0.79 s |
| up_proj 6144→2048 | 0.87 s | 0.79 s | ~0.86 s | ~0.78 s |
| down_proj 2048→6144 | ~0.76 s | ~0.75 s | 0.75 s | 0.75 s |
| **whole expert** | **2.55 s** | 2.39 s | **2.48 s** | 2.32 s |

(cold = includes finalize_capture_H LDL; warm-H = reused finalized H, the
realistic per-tensor marginal when gate/up share one H.)

## Findings

1. **~2.5 s per expert** — 3× faster than 07-lazy-encode's 7.5 s planning
   number. Lazy encode is more viable than designed for.
2. **K3 ≈ K4 cost** (within 3%): Viterbi/tile work dominates and is
   K-independent at these bitrates. `VLLM_FQ_ENCODE_BUDGET_PCT` sizing needs
   no per-K distinction.
3. At the default **5% encode budget: ~71 experts/hour** on one GPU —
   a freshly promoted working set of ~100 experts fully K4-encodes in
   ~1.4 h of background time.
4. Full offline K4 overlay (all 19,200 routed experts, 75 layers): **~13
   GPU-h** on one card (~3.3 h on the idle quad) — the `artifact` mode
   fallback is cheap if ever wanted.
5. Sanity: proxy error K4 2–4× lower than K3 on the same tensor
   (gate: 0.00183 vs 0.00724) — the expected quality gradient, from real
   weights.

## Knob feed-in (01 §6)

`VLLM_FQ_ENCODE_BUDGET_PCT=5` default stands; encode-queue latency
(seconds-scale per promotion) confirmed acceptable vs promotion cadence
(minutes-scale intervals).

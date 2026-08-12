# PoC v37: Complete hybrid Pareto — K2+K5trsc is new best at 7 bpw

## Updated Best Tiers (10 experts, all combinations tested)

| bpw | Best tier | MSE | vs prev best | Improvement |
|-----|-----------|-----|--------------|-------------|
| 2 | K2 | 1.061e-01 | — | — |
| 3 | K3 | 2.718e-02 | — | — |
| 4 | K4 | 7.286e-03 | K2+K2trsc: 7.305e-03 | K4 wins by 0.3% |
| 5 | K2+K3trsc | 1.892e-03 | — | 32% better than K4+1LM |
| 6 | K2+K4trsc | 5.276e-04 | — | 47% better than K4+2LM |
| **7** | **K2+K5trsc** | **1.726e-04** | K3+K4trsc: 2.139e-04 | **19% better!** |
| 8 | K2+6LM | 8.767e-05 | — | 8% better than K4+4LM |
| 9 | K3+6LM | 2.629e-05 | — | — |
| 10 | K4+6LM | 9.613e-06 | — | — |

## Key findings

1. **K2+K5trsc is the new best at 7 bpw** — K2 base + 5-bit rescaled trellis on
   residual gives 1.726e-04, which is 19% better than K3+K4trsc (2.139e-04)
   and 44% better than K4+3LM (3.085e-04).

2. **K2+K2trsc matches K4 at 4 bpw** — K2+K2trsc (7.305e-03) ≈ K4 (7.286e-03).
   This means the K2 base + rescaled K2 trellis residual is a viable fungible
   alternative to K4 trellis at the same bitrate.

3. **K2 base dominates 5-7 bpw** — K2's larger residual (σ≈0.29) gives the
   rescaled trellis more signal, making K2+KNtrsc the best for N=3,4,5.

4. **LM wins at 8+ bpw** — Lloyd-Max's adaptive c128 clusters (2048 effective
   levels) beat the fixed trellis codebook at higher bitrates.

## Pattern: Base tier vs residual bitrate tradeoff

| bpw | Best base | Residual K | Why |
|-----|-----------|------------|-----|
| 2-4 | Trellis only | — | No residual needed |
| 5-7 | K2 (largest σ) | K3-K5 rescaled trellis | Large residual → trellis codebook works well |
| 8-10 | K2-K4 | 6-bit LM | LM's adaptive clusters handle high bitrate better |

The optimal strategy is: use the base tier with the LARGEST residual that
still fits the bitrate budget, then quantize the residual with rescaled
trellis (for 5-7 bpw) or LM (for 8-10 bpw).

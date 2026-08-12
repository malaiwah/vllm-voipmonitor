# PoC v40: Three-stage rescaled trellis — 2.3× better than LM at 8 bpw!

## BREAKTHROUGH: Multi-stage rescaled trellis with optimal allocation

### Best at each bpw (5 experts)

| bpw | Best tier | MSE | vs single-stage | vs LM |
|-----|-----------|-----|-----------------|-------|
| 5 | K2+K3trsc | 1.893e-03 | — | 32% better than K4+1LM |
| 6 | K2+K1trsc+K3trsc | 5.146e-04 | 3% better than K2+K4trsc | 48% better than K4+2LM |
| 7 | K2+K1trsc+K4trsc | 1.418e-04 | 18% better than K2+K5trsc | 54% better than K4+3LM |
| **8** | **K2+K1+K2+K3trsc** | **3.890e-05** | **2.1× better than K2+K6trsc** | **2.3× better than K2+6LM!** |

### Optimal allocation pattern

The optimal stage allocation follows a clear pattern:
- **6 bpw** (4 residual bits): K1+K3 (small+large)
- **7 bpw** (5 residual bits): K1+K4 (small+large)
- **8 bpw** (6 residual bits): K1+K2+K3 (small+medium+large, 3 stages)

**Rule: start small, end large.** Each stage captures a progressively smaller
residual. The first stage (K1) removes the bulk, leaving a more Gaussian
residual for subsequent stages. The last stage (K3 or K4) provides the
fine refinement.

### Why three-stage beats two-stage at 8 bpw

| Method | Stages | MSE |
|--------|--------|-----|
| K2+K6trsc (single) | 1 | 8.129e-05 |
| K2+K1trsc+K5trsc (2-stage) | 2 | 4.505e-05 |
| K2+K1+K2+K3trsc (3-stage) | 3 | **3.890e-05** |

Each additional stage:
1. Reduces the residual magnitude for the next stage
2. Makes the residual more Gaussian (central limit effect)
3. Better matches the trellis codebook after rescaling

This is multi-stage TCQ (successive refinement of trellis coded quantization),
which is theoretically optimal for Gaussian sources (Jafarkhani 1999).

### Updated best method (v40)

| bpw | Best tier | MSE |
|-----|-----------|-----|
| 2 | K2 | 1.061e-01 |
| 3 | K3 | 2.718e-02 |
| 4 | K4 | 7.286e-03 |
| 5 | K2+K3trsc | 1.893e-03 |
| 6 | K2+K1trsc+K3trsc | 5.146e-04 |
| 7 | K2+K1trsc+K4trsc | 1.418e-04 |
| 8 | K2+K1+K2+K3trsc | 3.890e-05 |
| 9 | K3+6LM | 2.629e-05 |
| 10 | K4+6LM | 9.613e-06 |

Note: At 9-10 bpw, LM still wins (more residual bits → adaptive clustering
advantage). But at 8 bpw, 3-stage rescaled trellis is 2.3× better than LM!

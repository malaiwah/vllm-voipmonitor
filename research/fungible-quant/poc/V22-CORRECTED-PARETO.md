# PoC v21-v22b: Extended Pareto + Corrected bpw + Large Lloyd-Max Codebooks

## v21: 5-tier K3/K4/K5/K6/K7 extended to 7.0 bpw

Extended the Pareto frontier with K7 = K6 + 1-bit scalar.
Result: smooth 3.0-7.0 bpw, but plateau beyond 7.0 (1-bit scalar too weak).

**IMPORTANT**: The bpw labels in v21 (and all prior v15-v20 Pareto frontiers)
were WRONG. Each tier upgrade was treated as +1 bpw, but:

| Tier | Composition | Actual bpw | v21 labeled |
|------|------------|------------|-------------|
| K3   | trellis    | 3          | 3 ✓        |
| K4   | trellis    | 4          | 4 ✓        |
| K5   | K4+2LM     | 6          | 5 ✗        |
| K6   | K5+1sc     | 7          | 6 ✗        |
| K7   | K6+1sc     | 8          | 7 ✗        |

The 2-bit Lloyd-Max adds 2 bits/weight, not 1. The old "5.0 bpw" was actually
6.0 bpw, "6.0" was 7.0, etc.

## v22: Large Lloyd-Max codebooks vs stacking

Tested single large codebook on trellis residual vs stacking smaller codebooks:

| Method              | Actual bpw | MSE       | vs stacking |
|---------------------|------------|-----------|-------------|
| K4+2LM (stack K5)   | 6          | 1.070e-03 | baseline    |
| K3+4LM              | 7          | 3.627e-04 | 1.44× better than K4+2LM+1sc |
| K4+4LM              | 8          | 1.360e-04 | 2.65× better than K4+2LM+2sc |
| K4+6LM              | 10         | 1.759e-05 | 2.25× better than K5+4LM    |
| K4+8LM              | 12         | 9.271e-06 | 1.72× better than K5+4LM+2LM|

**Finding**: A single N-bit Lloyd-Max codebook on the trellis residual is
1.4-2.7× better than stacking independent smaller codebooks at the same
total bitrate. The larger codebook directly represents the residual
distribution without compounding quantization errors.

## v22b: Corrected Pareto with all tiers

Tested all tier combinations with correct bpw labels:

| Tier     | bpw | MSE       | Notes                         |
|----------|-----|-----------|-------------------------------|
| K3       | 3   | 2.718e-02 | Only option                   |
| K4       | 4   | 7.290e-03 | Beats K3+1LM by 26%           |
| K3+1LM   | 4   | 9.964e-03 | = K3+1sc (1-bit LM = scalar)  |
| K4+1LM   | 5   | 2.813e-03 | Beats K3+2LM by 16%           |
| K3+2LM   | 5   | 3.338e-03 |                               |
| K4+2LM   | 6   | 1.070e-03 | ≈ K3+3LM (tied)               |
| K3+3LM   | 6   | 1.079e-03 |                               |
| K3+4LM   | 7   | 3.627e-04 | Beats K4+3LM by 21% (crossover!) |
| K4+3LM   | 7   | 4.588e-04 |                               |
| K4+2LM+1sc | 7 | 5.207e-04 | Old K6 approach               |
| K4+4LM   | 8   | 1.360e-04 |                               |

### Crossover at 6-7 bpw

Below 6 bpw: K4+NLM > K3+(N+1)LM (trellis base is better)
Above 6 bpw: K3+(N+1)LM > K4+NLM (trellis overhead becomes liability)

This suggests the EXL3 trellis has an encoding overhead that becomes
significant relative to the residual at higher bitrates.

### Corrected Pareto frontier (actual bpw)

| bpw | MSE       | Improvement |
|-----|-----------|-------------|
| 3.0 | 2.718e-02 | —           |
| 3.5 | 1.667e-02 | 39%         |
| 4.0 | 7.290e-03 | 56%         |
| 4.5 | 5.117e-03 | 30%         |
| 5.0 | 3.334e-03 | 35%         |
| 5.5 | 2.088e-03 | 37%         |
| 6.0 | 1.064e-03 | 49%         |
| 6.5 | 6.853e-04 | 36%         |
| 7.0 | 4.737e-04 | 31%         |
| 7.5 | 3.488e-04 | 26%         |
| 8.0 | 2.698e-04 | 23%         |

### Comparison with old (mislabeled) Pareto

At actual 5.0 bpw: new 3.334e-03 vs old 3.883e-03 → 14% better
(K4+1LM tier fills the 4-6 gap that was previously only covered by mixing)

At actual 7.0 bpw: new 4.737e-04 vs old 5.161e-04 → 8% better
(K3+4LM beats K4+2LM+1sc)

## Updated best method

**6-tier tile-level mixed precision** with correct bpw:
- K3 (3 bpw), K4 (4 bpw), K4+1LM (5 bpw), K4+2LM (6 bpw), K3+4LM (7 bpw), K4+4LM (8 bpw)
- DP-optimal tile assignment, continuously variable 3.0-8.0 bpw
- 2-bit tier bitmap per tile (0.0078 bpw overhead, shared across all targets)
- Calibration-free (variance proxy works, v12)
- Runtime: trellis dequant + codebook lookup for LM residual

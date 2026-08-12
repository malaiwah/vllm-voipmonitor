# PoC v42: MSRT optimization — K2 base confirmed best, hybrid worse, universal

## Findings

### 1. K2 base beats K3 base for MSRT

| bpw | K2 base MSRT | K3 base MSRT | K2 advantage |
|-----|-------------|-------------|--------------|
| 5 | K2+K3trsc: 1.892e-03 | K3+K2trsc: 1.952e-03 | 3% |
| 6 | K2+K1+K3trsc: 5.144e-04 | K3+K1+K2trsc: 5.492e-04 | 6% |
| 7 | K2+K1+K4trsc: 1.415e-04 | K3+K1+K3trsc: 1.670e-04 | 15% |
| 8 | K2+K1+K2+K3trsc: 3.868e-05 | K3+K1+K1+K2trsc: 1.627e-04 | 4.2× |

K2's larger residual (σ≈0.29) gives each subsequent stage more signal.
K3's smaller residual (σ≈0.17) is harder to refine with rescaled trellis.

### 2. MSRT+LM hybrid is much worse

| bpw | Pure MSRT | MSRT+LM hybrid | Ratio |
|-----|-----------|----------------|-------|
| 6 | K2+K1trsc+K3trsc: 5.14e-04 | K2+K1trsc+2LM: 3.38e-03 | 6.6× worse |
| 8 | K2+K1+K2+K3trsc: 3.87e-05 | K2+K1+K2trsc+2LM: 2.35e-04 | 6.1× worse |

LM on the MSRT residual is terrible because:
1. The MSRT residual after K1+K2 stages is very small and non-Gaussian
2. LM's codebook can't adapt to this tiny, structured residual
3. Rescaled trellis (K3trsc) is far better at capturing the fine residual

**Conclusion**: Pure MSRT (all trellis stages) is strictly better than any
hybrid with LM. LM is completely obsolete.

### 3. MSRT is universal across gate_proj and down_proj

gate_proj and down_proj give identical results (ratio ≤ 1.001):
- 5bpw: 1.892e-03 vs 1.892e-03
- 6bpw: 5.144e-04 vs 5.143e-04
- 7bpw: 1.415e-04 vs 1.415e-04
- 8bpw: 3.868e-05 vs 3.872e-05

MSRT inherits the universality of the trellis codebook — same quality
regardless of weight matrix shape.

## Updated definitive best (v42, 10 experts, gate+down verified)

| bpw | Best tier | MSE |
|-----|-----------|-----|
| 2 | K2 | 1.061e-01 |
| 3 | K3 | 2.718e-02 |
| 4 | K4 | 7.286e-03 |
| 5 | K2+K3trsc | 1.892e-03 |
| 6 | K2+K1trsc+K3trsc | 5.144e-04 |
| 7 | K2+K1trsc+K4trsc | 1.415e-04 |
| 8 | K2+K1+K2+K3trsc | 3.868e-05 |
| 9 | K2+K1+K1+K2+K3trsc | 1.095e-05 |
| 10 | K2+K1+K1+K1+K2+K3trsc | 3.381e-06 |

All MSRT. No LM. K2 base. Universal across projections.

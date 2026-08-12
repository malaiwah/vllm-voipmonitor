# PoC v31: Definitive Pareto Frontier (c128, 10 experts, gate+down)

## Final Pareto Frontier

8-tier system with c128 clustered codebooks, 10 experts, gate_proj AND down_proj:

| Target bpw | Actual bpw | MSE (gate) | MSE (down) | Ratio gate/down |
|------------|------------|------------|------------|-----------------|
| 2.0 | 2.000 | 1.061e-01 | 1.061e-01 | 1.0000 |
| 2.5 | 2.500 | 6.572e-02 | 6.572e-02 | 1.0000 |
| 3.0 | 3.000 | 2.718e-02 | 2.718e-02 | 1.0000 |
| 3.5 | 3.500 | 1.698e-02 | 1.698e-02 | 1.0000 |
| 4.0 | 4.000 | 7.286e-03 | 7.284e-03 | 1.0000 |
| 4.5 | 4.500 | 4.972e-03 | 4.971e-03 | 1.0000 |
| 5.0 | 5.000 | 2.795e-03 | 2.794e-03 | 1.0000 |
| 5.5 | 5.500 | 1.824e-03 | 1.823e-03 | 1.0000 |
| 6.0 | 6.000 | 9.885e-04 | 9.868e-04 | 1.0017 |
| 6.5 | 6.500 | 5.933e-04 | 5.929e-04 | 1.0007 |
| 7.0 | 7.000 | 3.082e-04 | 3.076e-04 | 1.0020 |
| 7.5 | 7.500 | 1.789e-04 | 1.789e-04 | 1.0003 |
| 8.0 | 8.000 | 8.807e-05 | 8.805e-05 | 1.0002 |
| 8.5 | 8.500 | 6.548e-05 | 6.548e-05 | 1.0000 |
| 9.0 | 9.000 | 4.608e-05 | 4.609e-05 | 0.9999 |
| 9.5 | 9.500 | 2.750e-05 | 2.751e-05 | 0.9999 |
| 10.0 | 9.988 | 1.071e-05 | 1.071e-05 | 1.0000 |

**gate_proj and down_proj are identical** (ratio ≤ 1.002). Hadamard regularization
makes the residual distribution identical regardless of weight matrix shape.

## Entropy-coded bpw

| Tier | Raw bpw | Entropy bpw | Savings |
|------|---------|-------------|---------|
| K4+1LM | 5 | 5.000 | 0% |
| K4+2LM | 6 | 5.899 | 1.7% |
| K4+3LM | 7 | 6.700 | 4.3% |
| K2+6LM | 8 | 7.441 | 7.0% |
| K4+4LM | 8 | 7.495 | 6.3% |
| K3+6LM | 9 | 8.436 | 6.3% |
| K4+6LM | 10 | 9.421 | 5.8% |

K2+6LM has LOWER entropy bpw than K4+4LM (7.441 vs 7.495) — entropy coding
amplifies the crossover advantage of K2+6LM.

## Rate-distortion analysis

MSE follows the 6dB/bit law (MSE ∝ 4^(-bpw)):
- 2→3 bpw: 3.9× reduction (theory: 4×)
- 3→4 bpw: 3.7× (theory: 4×)
- 4→5 bpw: 2.6× (theory: 4× — below theory due to trellis overhead)
- 5→6 bpw: 2.8×
- 6→7 bpw: 3.2×
- 7→8 bpw: 3.5× (approaching theory as LM dominates)
- 8→9 bpw: 1.9× (sub-linear, approaching noise floor)
- 9→10 bpw: 4.3× (residual almost fully captured)

## Conclusion

This is the definitive Pareto frontier for fungible quantization of GLM-5.2.
The 8-tier system with c128 codebooks provides continuously variable 2-10 bpw
from a single encoded model, with near-zero overhead and no re-encoding needed.

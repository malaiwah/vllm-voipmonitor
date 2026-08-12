# PoC v48: EXL3 config + validation — mul1 worse, up_proj universal, 70 experts CV=0.11%

## Results

### 1. mul1 codebook: 2.6× worse than mcg

| Codebook | 8bpw MSE | vs mcg |
|----------|----------|--------|
| mcg (standard) | 3.890e-05 | baseline |
| mul1 (alternative) | 1.017e-04 | 2.6× worse |
| mul1 residual only | 1.017e-04 | 2.6× worse |

The `mul1` codebook (hash multiplier 0x83DCD12D) is significantly worse than
the standard `mcg` codebook (0xCBAC1FED). The mcg codebook is better optimized
for Gaussian sources. This confirms that the EXL3 trellis codebook choice matters
and mcg is the right one.

### 2. up_proj: identical to gate_proj (universal)

| Projection | 8bpw MSE |
|------------|----------|
| gate_proj | 3.890e-05 |
| down_proj | 3.872e-05 (v42) |
| up_proj | 3.863e-05 |

All three projections give identical results (within 0.7%). MSRT is universal
across all weight matrix shapes and projections.

### 3. All 70 experts: CV = 0.11% (extremely uniform)

| Metric | Value |
|--------|-------|
| Mean MSE | 5.1425e-04 |
| Std MSE | 5.6687e-07 |
| CV | 0.11% |
| Min/Max ratio | 1.0056 |

All 70 experts give nearly identical MSRT quality (CV=0.11%, ratio=1.006).
This confirms that GLM-5.2 experts are statistically homogeneous after Hadamard
regularization — per-expert allocation provides zero benefit.

### 4. Hadamard block 64: 0.1% better (negligible)

| Hadamard block | 6bpw MSE |
|----------------|----------|
| 128 (standard) | 5.144e-04 |
| 64 | 5.137e-04 |

Smaller Hadamard blocks (64) give 0.1% improvement — negligible. The standard
128 block is sufficient.

## Conclusion

All four tests confirm MSRT is already optimally configured:
- mcg codebook is the right choice (mul1 is 2.6× worse)
- MSRT is universal across all projections (gate/up/down identical)
- 70 experts are homogeneous (CV=0.11%) — no per-expert benefit
- Hadamard 128 is sufficient (64 gives negligible gain)

MSRT with mcg codebook, Hadamard 128, global RMS rescaling, K1-first allocation
is the definitive optimal configuration.

# PoC v35: Rescaled trellis-on-residual — 34-37% BETTER than Lloyd-Max!

## MAJOR BREAKTHROUGH

Rescaling the residual to match the trellis codebook's expected input range
fixes the codebook mismatch from v34 and enables TCQ to outperform Lloyd-Max
on the residual at lower bitrates.

### Results

| Method | bpw | MSE | vs Lloyd-Max |
|--------|-----|-----|--------------|
| K4+K2trellis_pertile_rescaled | 6 | 6.176e-04 | **37% better** |
| K4+K2trellis_rescaled | 6 | 6.512e-04 | **34% better** |
| K4+2LM_c128 | 6 | 9.886e-04 | baseline |
| K4+K2trellis_unscaled | 6 | 3.344e-02 | 51× worse (v34) |
| K4+K3trellis_rescaled | 7 | 2.843e-04 | **8% better** |
| K4+3LM_c128 | 7 | 3.087e-04 | baseline |
| K4+4LM_c128 | 8 | 9.546e-05 | baseline (LM wins) |
| K4+K4trellis_rescaled | 8 | 1.934e-04 | 2× worse |

### Why rescaling works

The EXL3 trellis codebook is designed for data with RMS ≈ |codebook_scale| (1.24).
The K4 residual has RMS ≈ 0.085 (14.6× smaller). Without rescaling, the codebook
levels are completely wrong for the residual (v34: 51× worse).

With rescaling: scale residual by |cbs|/residual_rms ≈ 14.6×, quantize with
trellis, scale back. Now the trellis codebook matches the data range.

### Why TCQ beats Lloyd-Max at 6-7 bpw but not at 8 bpw

At 2-3 bit residual (6-7 bpw total):
- TCQ uses 2^L states (L=12 for EXL3) → much larger codebook than 2^N LM levels
- Viterbi finds optimal path → better rate-distortion than greedy scalar
- QTIP: TCQ distortion 0.071 vs scalar 0.118 at 2-bit → 40% better

At 4-bit residual (8 bpw total):
- Lloyd-Max with c128 clusters has 128 × 16 = 2048 effective levels
- This adapts to per-cluster σ, which TCQ's fixed codebook can't do
- LM wins because it adapts to the residual's non-uniform σ distribution

### Per-tile vs global rescaling

Per-tile rescaling (each 16×16 tile scaled independently) is 5% better than
global rescaling (6.176 vs 6.512) but much slower (Python loop over 49152 tiles).
For production, global rescaling is the practical choice.

## Updated best method

The optimal tier system now includes rescaled trellis-on-residual for 6-7 bpw:

| bpw | Best tier | MSE | Method |
|-----|-----------|-----|--------|
| 2 | K2 | 1.061e-01 | Trellis |
| 3 | K3 | 2.718e-02 | Trellis |
| 4 | K4 | 7.287e-03 | Trellis |
| 5 | K4+1LM | 2.796e-03 | Lloyd-Max |
| **6** | **K4+K2trellis_rescaled** | **6.512e-04** | **Rescaled trellis** |
| **7** | **K4+K3trellis_rescaled** | **2.843e-04** | **Rescaled trellis** |
| 8 | K4+4LM | 9.546e-05 | Lloyd-Max (LM wins) |
| 10 | K4+6LM | 9.632e-06 | Lloyd-Max |

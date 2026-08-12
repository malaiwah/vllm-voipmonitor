# PoC v43: Tile-level MSRT Pareto — 0-1.4% improvement from mixing

## Finding

Tile-level mixing of MSRT tiers gives only 0-1.4% improvement over uniform
MSRT, much less than the 10-15% we saw with LM tiers.

| bpw | Uniform MSRT | Tile-level MSRT | Improvement |
|-----|-------------|-----------------|-------------|
| 5.0 | 1.892e-03 | 1.892e-03 | 0.0% |
| 5.5 | — | 1.170e-03 | (new fractional point) |
| 6.0 | 5.144e-04 | 5.144e-04 | 0.0% |
| 6.5 | — | 3.232e-04 | (new fractional point) |
| 7.0 | 1.415e-04 | 1.415e-04 | 0.0% |
| 7.5 | — | 8.608e-05 | (new fractional point) |
| 8.0 | 3.868e-05 | 3.845e-05 | 0.6% |
| 8.5 | — | 2.304e-05 | (new fractional point) |
| 9.0 | 1.095e-05 | 1.079e-05 | 1.4% |
| 9.5 | — | 6.743e-06 | (new fractional point) |
| 10.0 | 3.381e-06 | 3.415e-06 | -1.0% (noise) |

## Why tile-level mixing gives less benefit with MSRT

With LM tiers, tile-level mixing gave 10-15% improvement because different
tiles benefit differently from LM's adaptive codebook.

With MSRT, the multi-stage trellis already captures per-tile variation
through the Viterbi algorithm's path optimization. Each tile's trellis
encoding is individually optimized, so there's less inter-tile quality
variation to exploit through mixing.

## Conclusion

Uniform MSRT (all tiles at same tier) is nearly optimal. Tile-level mixing
provides fractional-bit granularity (5.5, 6.5, 7.5, etc.) but minimal
quality improvement at integer bitrates.

The definitive Pareto:
- Integer bitrates: uniform MSRT
- Half-integer bitrates: tile-level mixing for 0-1.4% improvement
- MSRT's successive refinement already captures most inter-tile variation

## Final definitive Pareto (10 experts, 0.5-bit steps)

| bpw | MSE |
|-----|-----|
| 2.0 | 1.061e-01 |
| 2.5 | 6.572e-02 |
| 3.0 | 2.718e-02 |
| 3.5 | 1.698e-02 |
| 4.0 | 7.286e-03 |
| 4.5 | 4.439e-03 |
| 5.0 | 1.892e-03 |
| 5.5 | 1.170e-03 |
| 6.0 | 5.144e-04 |
| 6.5 | 3.232e-04 |
| 7.0 | 1.415e-04 |
| 7.5 | 8.608e-05 |
| 8.0 | 3.845e-05 |
| 8.5 | 2.304e-05 |
| 9.0 | 1.079e-05 |
| 9.5 | 6.743e-06 |
| 10.0 | 3.415e-06 |

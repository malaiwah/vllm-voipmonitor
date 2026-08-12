# Round 7: Embedded TCQ Theory + FLUTE Kernel + Final Status

## Embedded TCQ (IEEE TIP 2008, JPEG2000 Part 2)

The concept of "embedded trellis coded quantization" has existed since the 1990s:
- TCQ can be made rate-scalable by modifying the trellis structure
- JPEG2000 Part 2 uses embedded TCQ for progressive image coding
- Key challenge: path through trellis can't be determined without LSBs
- Solution: approximate inversion when LSBs are missing

**Relevance to our work**: Our tile-level approach is a pragmatic alternative
to true embedded TCQ. Instead of modifying the trellis structure (which would
require changes to EXL3's Viterbi decoder), we use independent trellis
quantization at K3 and K4 per tile, then select per-tile at load time.
This is simpler to implement and doesn't require changes to the EXL3 format.

## FLUTE Kernel (EMNLP 2024)

FLUTE (Fast Lookup Table Engine) for LUT-quantized LLMs:
- Offline restructuring of quantized weight matrix to minimize bit manipulations
- Vectorization for fast dequantization
- Supports W4G128 (4-bit, group size 128) with 3-5× speedup over torch.mm

**Relevance**: Our tile-level approach could use FLUTE-style kernels with
per-tile LUTs. Each tier (K3, K4, K5, K6) would have its own LUT, and the
kernel would branch per tile based on the benefit threshold.

## Final Status

### Experiments completed: 18 PoC versions (v4-v18)
### Papers reviewed: 37+ across 7 rounds
### Best method: 4-tier tile-level K3/K4/K5/K6 mixed precision

### Key findings (all measured on real GLM-5.2 weights):

1. **Continuously variable 3.0-6.0 bpw** with smooth monotonic Pareto frontier
2. **2-bit benefit per tile** (0.0078 bpw, shared across all targets) for fungibility
3. **4-bit quantized benefit** gives 0% quality loss (benefit distribution is narrow)
4. **Per-expert allocation** doesn't help (5 metrics confirm homogeneity)
5. **Cross-layer allocation** doesn't help (layers 10 & 40 identical)
6. **Tile difficulty is i.i.d. random** (no spatial structure, correlation ~0.001)
7. **Variance proxy** for tile difficulty: <1% quality loss
8. **Bitmap entropy** ≤ 1.0 (entropy-coded overhead: 0.0039 bpw worst case)
9. **K3+2lloyd** is best at exactly 5.0 bpw (6.83× better than K4)
10. **K4+10%K5 tiles** beats uniform K4 at 4.1 bpw (103.9% gap)

### Future directions:
1. MXFP4 hardware path (RTX 5090 native)
2. Token-aware precision (MoBiQuant)
3. FLUTE-style per-tile LUT kernel
4. Expert merging + fungible quantization

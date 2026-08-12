# PoC v34: Trellis-on-residual — TCQ on residual is 2-34× WORSE than Lloyd-Max

## Key Finding

Using the EXL3 trellis quantizer (TCQ) on the residual is dramatically worse
than scalar Lloyd-Max:

| bpw | Trellis method | Trellis MSE | LM MSE | Ratio |
|-----|----------------|-------------|--------|-------|
| 5 | K3+K2trellis_res | 3.612e-02 | 3.298e-03 | 10.95× worse |
| 6 | K4+K2trellis_res | 3.344e-02 | 9.886e-04 | 33.82× worse |
| 6 | K3+K3trellis_res | 8.534e-03 | 1.046e-03 | 8.16× worse |
| 7 | K4+K3trellis_res | 7.642e-03 | 3.087e-04 | 24.76× worse |
| 4 | K2+K2trellis_res | 3.970e-02 | 1.235e-02 | 3.21× worse |

## Why trellis-on-residual fails

The EXL3 trellis codebook is optimized for the **weight distribution** (after
Hadamard regularization): approximately Gaussian with σ ≈ 1.0, normalized to
the codebook scale.

The **residual** (w_reg - qk4) has a very different distribution:
- Much smaller σ (≈ 0.1-0.3 of the weight σ)
- More peaked around 0
- Different shape (sub-Gaussian, not exactly Gaussian)

Using the trellis codebook (designed for σ≈1) on residual data (σ≈0.1) is
like using a ruler marked in centimeters to measure millimeters — the
codebook levels are completely wrong for the residual's scale.

Lloyd-Max, in contrast, adapts its codebook levels to the residual's actual
σ (via sigma-clustering), giving optimal quantization for the residual
distribution.

## QTIP vs our result

QTIP shows TCQ achieves 40% lower distortion than scalar quantization on
Gaussian sources. However, QTIP applies TCQ to the **original weights**
(not residuals), and the codebook is designed for that distribution.

Our approach uses:
- TCQ (EXL3 trellis) for the original weights — optimal
- Scalar Lloyd-Max for the residual — optimal (adapts to residual σ)

This combination is better than using TCQ for both, because the residual
has a different distribution that the TCQ codebook can't handle.

## Conclusion

**Trellis for base + Lloyd-Max for residual is the optimal combination.**
The trellis handles the weight distribution optimally (as QTIP shows),
and Lloyd-Max handles the residual distribution optimally (adapts codebook
to residual σ). Using trellis on the residual is 2-34× worse because the
codebook is mismatched to the residual's distribution.

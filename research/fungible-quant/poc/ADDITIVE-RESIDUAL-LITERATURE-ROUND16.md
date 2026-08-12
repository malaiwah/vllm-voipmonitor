# Round 16: RateQuant, HyperQuant, optimal rate allocation for MSRT

## RateQuant (arXiv:2605.06675, 2026)

- Optimal mixed-precision KV cache quantization via rate-distortion theory
- Uses reverse waterfilling algorithm (Cover & Thomas) for Gaussian rate-distortion
- Optimally distributes bit budget across parallel channels
- For Gaussian sources: D_i = σ_i² · 2^(-2R_i), minimize sum D_i subject to sum R_i = R

**Relevance to MSRT**: Our MSRT stages are sequential, not parallel. But the
reverse waterfilling principle suggests that optimal rate allocation across
stages should equalize the distortion contribution of each stage. Our v40
finding (start small K1, end large K3) is consistent: the first stage sees
the largest residual (largest σ), so it gets the smallest K (K1) — this is
the reverse waterfilling solution!

## HyperQuant (arXiv:2606.23406, 2026)

- Rate-distortion-optimal quantization pipeline
- Hadamard + optimal packing + entropy Rice-coding
- Combines four known ideas into a unified pipeline
- For LLM weights at 4 bps on H100

**Relevance**: HyperQuant's R-D optimality comes from Hadamard + entropy coding.
Our MSRT achieves better R-D by using multi-stage trellis instead of single-stage
scalar quantization. The Hadamard is common to both approaches.

## Optimal rate allocation for multi-stage quantization

For a Gaussian source with σ² variance, the rate-distortion function is:
D(R) = σ² · 2^(-2R)

For multi-stage (progressive) quantization with N stages, each with rate R_i:
D_total = σ² · Π_i (1 - 2^(-2R_i))

To minimize D_total subject to Σ R_i = R:
By Lagrangian optimization, the optimal allocation equalizes the marginal
distortion reduction per bit across all stages:
∂D/∂R_i = constant for all i

For Gaussian sources, this gives equal rates: R_i = R/N for all i.
However, our experiments show K1+K3 (unequal) beats K2+K2 (equal) at 6bpw,
suggesting the Gaussian assumption is imperfect after trellis quantization
(the residual after K1 is not exactly Gaussian).

## Key Insight from Round 16

The reverse waterfilling theorem explains why K2 base + progressive K1 stages
is optimal: K2 has the largest residual σ, so it gets the base rate (2 bits).
Each subsequent K1 stage refines the residual, and the remaining σ decreases,
so K1 (1 bit) is sufficient for each refinement step. The final K2+K3 stages
provide the fine-grained refinement.

## Total: 75+ papers reviewed across 16 rounds

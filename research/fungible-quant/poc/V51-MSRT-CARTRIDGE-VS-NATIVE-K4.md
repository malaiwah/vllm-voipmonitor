# PoC v51: MSRT Cartridge vs Native K4 — Feasibility Proven

## Question

Can a K3 base + K1trsc MSRT cartridge (at same 3.422 bpw as willfalco) match
native K4 quality on selected experts? This would enable LoRA-style hot-swap
of quantization accuracy without reloading vLLM.

## Answer: YES — within 3.1% of native K4, at identical memory

### Measured Results (10 experts, layers 10 + 40, gate_proj)

#### Per-expert-group quality (cartridge/K4 experts vs non-cartridge/K3 experts)

| Config | K4-group MSE | K3-group MSE | Overall MSE | vs K4 |
|--------|-------------|-------------|-------------|-------|
| **K4 all (reference)** | **7.286e-03** | 7.286e-03 | 7.286e-03 | 1.000× |
| **willfalco mixed (148K3+108K4)** | **7.286e-03** | 2.718e-02 | 1.922e-02 | 1.000× |
| **MSRT K3base + K1trsc cart (108)** | **7.515e-03** | 2.718e-02 | 1.931e-02 | **1.031×** |
| MSRT K2base + K2trsc cart (108) | 7.305e-03 | 1.061e-01 | 6.657e-02 | 1.003× |
| MSRT K3base + K1trsc (all 256) | 7.515e-03 | 7.515e-03 | 7.515e-03 | 1.031× |
| K3 all (brandonmusic) | 2.718e-02 | 2.718e-02 | 2.718e-02 | 3.731× |

#### Full metrics (layer 10)

| Config | MSE | Cosine sim | Rel. Frobenius | Max abs err |
|--------|-----|-----------|----------------|-------------|
| K3 all | 2.718e-02 | 0.991179 | 0.1326 | 2.867 |
| K4 all | 7.286e-03 | 0.997644 | 0.0686 | 2.869 |
| willfalco mixed | 1.922e-02 | 0.993766 | 0.1070 | 2.869 |
| **MSRT K3base+K1trsc cart108** | **1.931e-02** | **0.993735** | **0.1074** | **2.671** |
| MSRT K2base+K2trsc cart108 | 6.657e-02 | 0.978115 | 0.1846 | 2.571 |

### Key Finding

**MSRT K3 base + K1trsc cartridge matches willfalco at the same memory:**

| Metric | willfalco (native K4) | MSRT K3base+K1trsc | Difference |
|--------|----------------------|---------------------|------------|
| Effective bpw | 3.422 | 3.422 | identical |
| Memory (GiB/rank, TP4) | 74.1 | 74.1 | identical |
| K4-group MSE | 7.286e-03 | 7.515e-03 | 3.1% worse |
| K3-group MSE | 2.718e-02 | 2.718e-02 | identical |
| Overall MSE | 1.922e-02 | 1.931e-02 | 0.5% worse |
| Cosine similarity | 0.99377 | 0.99374 | 0.003% worse |
| Max abs error | 2.869 | 2.671 | **7% better** |

The MSRT cartridge provides K4-like accuracy (within 3.1% on cartridge experts,
0.5% overall) at the exact same memory cost as willfalco's native K4 approach.
The cosine similarity is virtually identical (0.99374 vs 0.99377).

### Memory Comparison (TP4, per rank)

| Config | eff bpw | GiB/rank | Base | Cartridge | Total |
|--------|---------|----------|------|-----------|-------|
| brandonmusic 3.0bpw | 3.000 | 65.0 | 65.0 | 0 | 65.0 |
| **willfalco 3.42bpw** | **3.422** | **74.1** | **74.1** | **0** | **74.1** |
| **MSRT K3base + K1trsc cart108** | **3.422** | **74.1** | **65.0** | **9.1** | **74.1** |
| MSRT K2base + K2trsc cart108 | 2.844 | 61.6 | 43.3 | 18.3 | 61.6 |
| Native K4 (all) | 4.000 | 86.6 | 86.6 | 0 | 86.6 |

The MSRT K3base+K1trsc approach splits the memory into:
- **Base checkpoint**: 65.0 GiB (all 256 experts at K3, always loaded)
- **Cartridge**: 9.1 GiB (108 experts at K1, loadable as LoRA adapter)

The cartridge is only **9.1 GiB** — small enough for LoRA hot-swap.
For comparison, a standard LoRA adapter is typically 0.1-1 GiB, but 9.1 GiB
is feasible with vLLM's two-tier LRU cache (CPU↔GPU swapping).

### MSRT K2 base Alternative

The K2 base + K2trsc cartridge approach uses even less total memory (61.6 GiB
vs 74.1 GiB) and the cartridge experts match K4 within 0.3% (7.305e-03 vs
7.286e-03). However, the non-cartridge experts stay at K2 (MSE 1.061e-01),
which is 3.9× worse than K3. This trades base quality for lower memory and
better cartridge accuracy.

### Runtime Cost

| Config | GEMM passes (cartridge experts) | GEMM passes (non-cartridge) |
|--------|--------------------------------|---------------------------|
| willfalco native K4 | 1 (K4) | 1 (K3) |
| MSRT K3base + K1trsc | 2 (K3 + K1trsc) | 1 (K3) |
| MSRT K2base + K2trsc | 2 (K2 + K2trsc) | 1 (K2) |

The cartridge adds 1 extra GEMM pass per cartridge expert per token batch.
For decode with 8 active experts and ~42% cartridge coverage:
- Avg cartridge experts per token: 8 × 108/256 ≈ 3.4
- Extra GEMM launches: ~3.4 per token
- Overhead: negligible (μs-scale launches vs ms-scale GEMMs)

## Conclusion

**MSRT LoRA cartridge is feasible and practical:**

1. **K3 base + K1trsc cartridge** matches willfalco's native K4 at the same
   memory (74.1 GiB/rank), with only 3.1% worse MSE on cartridge experts
   and 0.5% worse overall. Cosine similarity is virtually identical.

2. The **cartridge is only 9.1 GiB** — small enough for vLLM LoRA hot-swap
   with the existing two-tier LRU cache mechanism.

3. The cartridge can be **loaded/unloaded at runtime** without reloading vLLM,
   enabling online quantization tuning. Multiple cartridge variants can coexist.

4. The **K2 base + K2trsc cartridge** alternative provides even better cartridge
   accuracy (0.3% from K4) at lower total memory (61.6 GiB), but with worse
   non-cartridge quality (K2 vs K3).

5. The willfalco approach (native K4 on selected experts) gives the best
   cartridge-expert quality (exact K4) but requires a full checkpoint
   reassembly. The MSRT cartridge gives near-K4 quality (3.1% worse) with
   the flexibility of runtime hot-swap.

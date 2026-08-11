# GSM8K on the assembled 3.0bpw GLM-5.2 — quality baseline

The one thing every other number tonight does NOT measure. Convergence scores
say the loop picks the same experts a human picked; they say nothing about
whether the model is any good. This is the baseline a re-tiered run must be
compared against.

Serve: `/home/mbelleau/glm52-k3-assembled` — assembled by `fq_assemble` from
Progressive Tensors segments, TP4, `exl3`, `B12X_MLA_SPARSE`, `fp8_ds_mla` KV,
FQ loop running in dryrun with gate mass recorded.

| metric | value |
|---|---|
| task | `gsm8k_cot_zeroshot` (lm-eval v3) |
| items | **250-item subsample, seed 1234** (not the full 1319) |
| **flexible-extract exact_match** | **0.892 ± 0.0197** |
| strict-match exact_match | 0.116 ± 0.0203 |
| concurrency | 16 |

## Read the flexible number, not the strict one

`strict-match` requires the answer in a rigid `#### N` form. GLM-5.2 is a
reasoning model and emits chain-of-thought, so it almost never satisfies that
format — 0.116 measures format compliance, not arithmetic. `flexible-extract`
pulls the final number out of the reasoning and is the meaningful score here.
Both are reported because quoting only the good one would be cherry-picking.

## What it establishes

**89.2%** on GSM8K from a checkpoint that was reassembled out of published
per-expert segments rather than downloaded as a monolithic quant. Combined
with the M0 boot gate (219 tok/s at cc8, 0 failures), the artifact claim now
covers correctness as well as bootability: the segments reconstruct a model
that is not merely alive but competent.

## Honest limits

- 250-item subsample, so ±2% stderr; a 1-2 point difference against a future
  re-tiered run would be inside the noise. Any promotion claim needs either
  the full 1319 or a paired comparison on the same subsample.
- No re-tiered arm yet. This is the K3 floor; the interesting number is
  whether promoting experts moves it, and that needs K4 coverage on the
  served layers (campaign is at 16/75 and climbing).

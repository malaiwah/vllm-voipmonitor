# fq-attestation/2 — metadata design (2026-08-10, with Michel)

Additions over v1, grouped by consumer. v1 lines remain valid; v2 adopts
the DSSE/in-toto envelope (payloadType + Statement, subject = fragment
digest, predicateType URI) for ecosystem tooling compatibility.

## Reproducers (encode-of)
- determinism scope: encoder sha, exllamav3 version, torch+CUDA build,
  GPU arch, capture-methodology version (stack-scoped reproducibility is
  MEASURED fact: cross-stack activation drift, CUDA pow ulp)
- full quant_args: K, seed_base, sigma_reg, codebook+mcg multiplier,
  out_scales mode, slice-seed formula version
- capture lineage: capture fingerprint, corpus sha, plan seed, tokens

## Policy engines (ride-along signal)
- per-expert rel_rt_mse at this K; routed_count (phi) from the capture
  -> fetched fragments carry their own allocation signal (cold-start
  policy before local stats warm)

## Loaders
- tensor geometry (shapes/dtype), layout tag, TP/tile granularity,
  base-model config sha, num_experts

## Trust graph
- predicates: repack-of | encode-of | derived-from | equivalence-of |
  assembly-of (NEW: recipe + segment shas -> output shard shas, makes
  assembled checkpoints reproducible artifacts)
- parent/related attestation refs; supersedes/alternatives
- signer role: builder | rebuilder (countersign)
- license propagation (source model + quant licenses)

Rollout: emit v2 from the K3/K4 window pass (#24) + priming (#22);
fq_repack/fq_assemble grow --attest-v2; v1 verification kept forever.

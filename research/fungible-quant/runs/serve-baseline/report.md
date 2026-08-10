# Serving baseline — full GLM-5.2 K3 on the extracted r33 stack — 2026-08-10

First boot of the target model on this box, **without any container runtime**
(JarvisAI managed container, namespaces disabled): extracted r33 rootfs +
`gg-env/gg-run.sh` + staged script chain (`/home/mbelleau/gg-extra/bin`,
patches: absolute-path chain rewrites, vllm CLI shim, LD_PRELOAD
neutralized for host binaries, /cache → /home/mbelleau/cache/glm52).

## Configuration

- Model: `brandonmusic/GLM-5.2-EXL3-TR3-3.0bpw` @ `9297b9f1` (local snapshot)
- Launch: the image's own `serve-gilded-gnosis.sh` contract, profile
  `glm52-exl3`: TP4/DCP4, `B12X_MLA_SPARSE` attention, `moe-backend b12x`,
  `quantization exl3`, `kv nvfp4_ds_mla`, cudagraphs FULL_AND_PIECEWISE
  sizes 1-32, MTP=0, `MAX_MODEL_LEN=131072`, seqs 8, batched 3072, GMU 0.95
- GPUs 0-3 (serving quad), 81.8→92.7 GiB used per GPU after graphs
- DCP transport: `unsafe:topology-unavailable` fallback (PCIe host, no P2P
  path exposed — matches the vast-notes expectation for this shape)
- Boot timeline: launch 15:16Z → weights loaded ~15:33 → warmup + FULL
  graph capture → **Application startup complete 15:41Z** (~25 min cold,
  first-ever JIT/autotune on this box; caches persisted under
  /home/mbelleau/cache)

## Probes

- `/health` OK; `/v1/models` = `glm52-k3`
- Chat completion (reasoning path): 400 tok @ **37.7 tok/s** single request
- Raw completion sanity: coherent text, 48 tok @ 35.4 tok/s (short-run,
  includes prefill overhead in wall time)

Reference: v20 doc cites 44.66-48.48 tok/s matched TP4/DCP4 MTP0 decode on
the all-NODE dev host; this PCIe-fallback shape is expected to trail. MTP=3
and tuned profiles are later baselines, not this gate.

## What this proves

1. The full serving stack (GG vLLM r33 + b12x + EXL3 SM120 + custom NCCL +
   DCP fallback) runs end-to-end in the extracted env — M1 integration, M2
   dryrun, M3 reload, and T7 all have their substrate.
2. Combined with the assembly byte-identity sweep (m0-assemble/): segments
   → `fq_assemble` → checkpoint produces byte-identical shards, and those
   exact bytes boot and serve. The M0 gate's "bootable" requirement is
   satisfied by identity once the full sha sweep completes green.

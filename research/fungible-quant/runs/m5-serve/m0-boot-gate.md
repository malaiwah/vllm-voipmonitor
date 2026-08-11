# M0 boot gate — assembled GLM-5.2 3.0bpw serves, 2026-08-11

**The artifact claim, proven on the real model rather than the proxy: a
checkpoint assembled by our own tool out of Progressive Tensors segments
boots under GG vLLM and generates coherent text at production throughput.**

## What was served

| | |
|---|---|
| checkpoint | `/home/mbelleau/glm52-k3-assembled` — built by `fq_assemble` from K3 segments + z.ai non-expert tensors |
| shards / tensors | 81 safetensors, 935,105 tensors in the index, **0 missing** |
| runtime | GG vLLM r33 rootfs, TP4 on GPUs 0–3, `exl3` quant, `B12X_MLA_SPARSE`, `moe-backend b12x` |
| KV dtype | `fp8_ds_mla`, `use_index_cache=true` |
| mode | `VLLM_FQ_ENABLE=0` — clean A/B baseline, no fungible-quant loop |

Concurrent load on the box: the K2 encode campaign was running on GPUs 4–7
throughout. That is deliberate — it is the coexistence case, and it is why the
serve uses private JIT caches (see below).

## Boot

```
Model loading took 76.14 GiB memory and 400.935483 seconds
Available KV cache memory: 6.54 GiB
GPU KV cache size: 130,048 tokens
Maximum concurrency for 32,768 tokens per request: 3.97x
Application startup complete
```

76.14 GiB/rank against a 95.6 GiB card at `gpu-memory-utilization 0.92` — 80%
for weights, leaving 6.54 GiB of KV. The pre-flight estimate from raw file
sizes was 73.7 GiB/rank; the extra ~2.4 GiB is runtime overhead. Tight but
correct, and it bounds how many experts can be promoted later: **promotion
must come out of a fixed budget, not out of headroom that does not exist.**

### Cold-JIT cost, measured

Total wall time to ready was ~13 min, of which **~9 min was CUDA JIT
compilation before a single weight was read** (four workers pinned at 99.8%
CPU, ~1 GiB of compiled kernels accumulated). Cause: `CUDA_MODULE_LOADING=EAGER`
against an empty private cache. EAGER is an M2 mitigation for a cuBLAS
status-14 flake that appears when the encoder campaign hammers the other four
GPUs; the private cache exists because a *shared* JIT cache killed every M2
boot with an illegal memory access, including the FQ-disabled arm.

The cache persists at `/home/mbelleau/cache/jit-m5/`, so subsequent boots skip
this. Do not delete it between runs.

## Generation probe

Prompt: *"In one sentence, what is mixture-of-experts routing?"*, `max_tokens=64`,
`temperature=0`. The model emitted a structured reasoning preamble and was
truncated by the token cap before its final sentence — expected for a
reasoning model at 64 tokens, and sufficient to prove coherent decoding:

```
1.  **Analyze the Request:**
    *   Topic: Mixture-of-experts (MoE) routing.
    *   Constraint: Exactly one sentence.
```

Single-stream: 24 prompt + 64 completion tokens in 1.83 s = **34.9 tok/s**.

## Baseline throughput

`swap_evidence.py both`, 120 s, concurrency 8, `max_tokens=128`, math family:

| metric | value |
|---|---|
| requests | **208 issued, 208 succeeded, 0 failed** |
| completion tokens | 26,624 |
| aggregate throughput | **219.2 tok/s** |
| median scraped decode rate | 225.6 tok/s |
| range | 108.0 – 232.0 tok/s |

The 108 tok/s minimum is the first scrape interval (ramp-up), not a stall.

## Gate verdict

**PASS.** Assembled-from-segments boots, is healthy, generates coherent text,
and sustains 219 tok/s at cc8 with a zero failure rate, while an encode
campaign runs on the other half of the box.

## What this does NOT yet show

Stated so the claim is not read wider than the evidence:

- No expert re-tiering happened here (`VLLM_FQ_ENABLE=0` by design).
- No quality eval yet — GSM8K/GPQA are the next step, and the orchestrator's
  eval step was silently broken until fixed (it looked for `run-gpqa.sh`
  while the harness ships `eval_gpqa.sh`, so it logged "skipping eval" and
  exited 0).
- Bit-exactness of the assembled tensors against the source quant is the
  assembly report's job, not this one.

# First complete progressive load of GLM-5.2 from segments — 2026-08-11

**A 355B-class MoE was loaded into a live TP4 engine directly from per-expert
Progressive Tensors segments, with no assembled mixed checkpoint on disk, at
exactly the tier composition its policy asked for.**

Scoped precisely: this documents the **load**. Serving, throughput and eval
are separate claims and are not made here.

## The run

| | |
|---|---|
| attempt | 11 (`serve-attempt11-localfirst.log`) |
| policy | `policy-demo1-seeded.json` — mixed K3/K4, 75 layers |
| runtime | GG vLLM r33 rootfs, TP4 on GPUs 0–3, `exl3`, `B12X_MLA_SPARSE`, `moe-backend b12x` |
| sources | local segment dirs first, then `malaiwah/GLM-5.2-EXL3-FQ-segments` |
| concurrent load | K4 encode campaign on GPUs 6–7 throughout |

```
Model loading took 81.86 GiB memory and 3700.810454 seconds
```

## What the loader did

| metric | value |
|---|---:|
| distinct layers loaded | **76** (75 routed + MTP) |
| experts **degraded** below policy | **0** |
| source **rejections** | **0** |
| local-first hits (network skipped) | **160** |
| cross-rank shares (duplicate downloads avoided) | **219** |
| bytes delivered | **194.5 GiB** |
| sustained transfer | 54 MiB/s average, 94–105 MiB/s while fetching |

**Zero degraded experts is the headline.** Every one of GLM-5.2's 19,200
routed experts was installed at the tier the policy specified. The K ladder
never had to substitute, which is what the preceding fixes were for — an
earlier attempt degraded 190 experts to K2 because of an undeclared stats
counter that fired on the prefetch SUCCESS path.

## Tier composition, as installed

```
layer  3: tiers=((3, 206), (4, 50))  bits_digest=d704612a2fdb
layer 58: tiers=((3, 148), (4, 108)) bits_digest=38fb6aebeb45
```

Digest occurrence counts are all multiples of four (160, 112, 4 …) — every
layer's digest appears once per TP rank, so **all four ranks independently
resolved an identical mixed-K composition**. That is the property that makes
a segment-assembled model reproducible rather than merely functional.

## The two optimisations, measured

**Local-first resolution.** `prefetch_layer` previously consulted only the
remote mirrors, so layers already on disk were re-downloaded. Local K4 covers
layers 19–74 here, and the fix fired exactly there:

```
L19: local /home/mbelleau/glm52-segments/layer-019.k4.safetensors (no fetch)
```

160 hits — 160 object fetches that did not happen.

**Cross-rank sharing.** Every TP rank runs its own weight iterator over the
same policy, so all four want the same objects simultaneously. 219 shares
means 219 duplicate downloads avoided; without it this load would have moved
roughly four times the bytes.

## Cost model, stated honestly

194.5 GiB moved for a model whose K3 base tier was **not** held locally. The
split that produced that number:

| tier | local | consequence |
|---|---|---|
| K2 | 75/75 | never fetched |
| K4 | 56/75 | 56 layers skipped the network |
| K5 | 24/75 | never fetched |
| **K3** | **0** | **all 75 fetched — the policy's base tier** |

So 61.7 minutes is the **cold-K3** figure, not the floor. A cache holding the
base tier pays only for the upgrades. The per-layer spread on the same boot
makes the point: 3 s when both objects were present, ~9 min for a fresh 5 GB
tier — same code.

## Outcome: the load succeeded, the ENVELOPE did not

The engine got past weight load and then refused to start:

```
Available KV cache memory: -3.1 GiB
```

Not a loader fault — an arithmetic one, and the most useful result of the run.
At `gpu-memory-utilization 0.92` the effective budget is ~87.5 GiB (0.915
after CUDA-graph profiling). Weights took 81.86, activations and graphs ~8.7,
so KV came out negative. Reaching 6 GiB of KV would need a 96.6 GiB budget on
a 95.6 GiB card: **impossible at this weight size.**

The seeded policy is 26.3% K4, mean **3.263 bpw**. Against the proven flat-K3
point (3.000 bpw, 76.14 GiB, 6.54 GiB KV) that is +5.72 GiB of weights —
**more than the entire KV budget it had to come out of.**

| point | mean bpw | weights | KV |
|---|---:|---:|---:|
| flat K3 (M0 gate) | 3.000 | 76.14 GiB | +6.54 GiB |
| seeded mixed (attempt 11) | 3.263 | 81.86 GiB | **−3.1 GiB** |
| **fitted (attempt 12)** | **3.137** | ~79.1 GiB | ~2.5 GiB |

Calibration falls straight out of those two measured points: **~21.7 GiB of
weights per 1.0 bpw of mean expert bitrate**, on this model at TP4.

**Context length and expert bitrate are the same budget.** M0 measured 6.54
GiB of KV as 130,048 tokens, so ~19.9k tokens per GiB. Dropping
`max-model-len` 32768 → 8192 lets 2.5 GiB of KV serve ~50k tokens — about
**6.1x** concurrency, better than the 32k baseline's 3.97x — and hands the
freed 2.5 GiB back to weights, which buys 13.7% of experts at K4 instead of
2.2%.

That is the fixed-envelope discipline doing its job: the policy is not a wish
list, it is a budget, and this box priced it.

## What this does NOT show

- **Serving.** At the time of writing the engine is past weight load and in
  `torch.compile` + CUDA graph capture. No token has been generated from this
  checkpoint yet, and no throughput or quality number is claimed.
- **Convergence.** Zero deficits were recorded because nothing degraded, so
  the repay path was not exercised. The device-side tier install remains
  unimplemented (`fq_converge_layers` reports `installed: False`).
- **Time-to-ready.** 3700 s is weight loading only; JIT and graph capture are
  additional and were not isolated here.

# 08 — Streaming realignment audit (D3′ → D3″)

Governing principle (the actual concept of the idea): **a streaming pipeline
of late-binding decisions, cached and reused, adapting slowly to live load.**
Anything eager/offline/batch in the plan must justify itself against this or
be re-bound late. Full audit below.

## The unified cache hierarchy (replaces "artifacts" as a concept)

```
T0  VRAM tier slabs          occupancy = map contents (late-bound per interval)
T1  NVMe per-expert K-cache  entries keyed (model-rev, layer, expert, proj, K,
                             hessian-provenance); produced by T3-encode or T2-fetch
T2  HF remote cache          published/subscribed encodes — a K3 checkpoint IS a
                             warmed cache someone uploaded (community practice today)
T3  Sources                  BF16 weights (local NVMe ~1.4 TB cold, or HF range-read
                             ~75 MB/expert) + Hessians (streaming live, per 07)
```
Everything content-addressed in one keyspace; the policy JSON binds names to
tiers at decision time, never at build time.

## Audit table

| Plan element | Was | Disposition |
|---|---|---|
| K4 encode | campaign (D3) → lazy (D3′) | **lazy** — settled in 07 |
| K3 base | shipped artifact (~260 GB) | **Reframed: warmed T2 cache seed.** Deterministic encode ⇒ a published checkpoint is just someone's cache. No campaign; first deployment can crystallize it, or fetch the existing brandonmusic-lineage seed (P1). |
| ε curves | offline measure campaign | **Late-bound: encode reports its own reconstruction error** — ε_e(K) fills in as T1 fills. Cold-start from calibration-free proxies (AlphaQ weight spectra, router norms) computed at boot. Measure campaign demoted to optional validation (0c/0d). **Last rental dissolved.** |
| Hessians | campaign by-product, persisted | **Streaming** (07): live accumulation for candidates, host-resident; stored campaign statistic optional blend input, not dependency |
| Policy prior | "generic offline allocation" | Computed at boot from proxy-ε + uniform budget; refined by live loop. No offline step. |
| K menu (K2/K5/K6) | campaign-scoped (P3) | Config: menu entry + T1 namespace. K5/K6 usable now (intrinsics); K2 stays kernel work (M6). |
| Probe reference | fixed held-out set | **Stays eager deliberately** — comparability needs a fixed baseline; refresh slowly (new reference every N days, both scored across the seam). |
| Slab capacity / signatures / compiled launches | boot-time | **Stays eager deliberately** — geometry must exist before CUDA-graph capture (D1/L2 safety boundary). Capacity menu is the late-binding concession (05). |
| Quiesce/commit protocol | per-interval | Unchanged — the transaction layer late binding rides on. |

## When are BF16 full weights on disk needed?

**Only lazily, per-expert, at encode time.** The encoder reads one expert's
three tensors (~75.5 MB BF16) + its Hessian. Never at boot; never loaded
whole; not needed at all once T1/T2 cover the Ks the workload uses — except
for opportunistic re-encode under a refreshed Hessian blend. Placement
options: local NVMe (~1.4 TB cold storage, recommended), or pure-remote via
safetensors range reads (valid, slow fallback). BF16-on-disk is a T3 cache
tier, not a prerequisite.

## Cold-boot ladder (honest)

1. **T1 warm** (any prior run on this box): boot from local cache. Minutes.
2. **T1 cold, T2 reachable**: fetch the K3 seed — a normal quant download,
   exactly today's community flow. ~48 min at 0.9 Gbit/s.
3. **T1+T2 cold, BF16 local**: crystallize K3 locally (~41 GPU-h across 4
   GPUs ≈ 10 h background). Correct but a bad first-boot story — document,
   don't optimize.
   **Build note (2026-08-10):** measured, it is **~13 GPU-h ≈ 3.3 h on a
   quad** (2.5 s/expert × 19,456) — `../runs/encode-bench/report.md`.
4. **Fully streaming day-one** (boot in fast online NVFP4/MXFP8, serve
   immediately, trellis crystallizes expert-by-expert in background):
   requires mixed-format tiers in the kernel → **M6 extension**, noted not
   promised.

## Milestone deltas

- **M0 → "cache infrastructure"**: keyspace/schema, T1 layout, T2
  fetch backend, T3 range-read, seed import from the existing K3 checkpoint.
  No encode campaign anywhere. (P1 becomes "does the seed's provenance
  satisfy the manifest"; P2 gone; P3 = menu config.)
- **M2 policy** gains proxy-ε bootstrap + ε-refinement-at-encode.
- **M6 additions**: publish-back to T2 (share your converged cache entries —
  the community exchange story), fast-format cold boot (ladder rung 4).

## What this buys, stated once

**Build note (2026-08-10) — this is now literal, not aspirational.**
`--load-format progressive` boots a mixed-K EXL3 model **directly from
segments + a policy document**: no `fq_assemble` run, no assembled shards,
zero per-policy disk. Two different policies booted back-to-back from the
same segment store; the 042-policy boot is **token-identical** to the
assembled checkpoint's greedy output; streaming assembly costs **+1.8 s** in
the weights phase and decode throughput is at parity
(`../runs/loader-v2/report.md`). Two confirmations for the audit table
above: **ε really does ride along with the encode** (per-expert rel-RT-MSE
from the encoder's own done-JSONs drove the 0c solve — no `measure_model`
campaign), and **the T2 remote tier is one fetch path** (FragmentResolver
does local dirs → manifest/env source chain via HF ranged reads →
sha-verified content-addressed cache). One correction: the campaign itself
ran as a **streaming ring** (capture window → encode K2..K5 → publish →
delete) under a fixed 3 TB disk, so the "capture is preserved" variant in
`../runs/0c-campaign/MULTI-K-PLAN.md` did not happen.

The system has no build step. A new model needs: BF16 on HF (day zero) +
one K3 seed (crystallized by the first deployment that cares, shared via
T2). Every downstream deployment late-binds everything else — membership,
cardinality-within-capacity, which Ks exist, which experts deserve them,
ε itself — against its own live load, slowly, with caching at every tier.
That is the concept, now applied uniformly.

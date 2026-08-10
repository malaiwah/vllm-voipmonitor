# 07 — Lazy K4 encode-and-cache (revises D3)

Question answered here: why pre-encode all K4 offline instead of online
quantizing at first usage? Answer: no good reason — D3 inherited "everyone
pages pre-built variants" from the literature. Lazy encode is better aligned
with the product and with MoE calibration reality. D3 → **D3′**.

## Is Trellis streamable? (three senses)

| Sense | Answer | Detail |
|---|---|---|
| Incremental over the weight tensor | **No** | Hadamard rotation signs (suh/svh) + global scale are chosen by test-quantizing the whole tensor at target K — whole-tensor prepass required |
| Chunked/parallel encode compute | **Yes** | After the prepass, every 16×16 tile is an independent tail-biting Viterbi codeword; pausable, parallel. Natural unit = one expert (3 tensors, ~37.7M weights, ~7.5 s @ ~5M w/s on one GPU) |
| Streaming **calibration** | **Yes** | The Hessian is a running sum over routed tokens (H += xᵀx) — a streaming sufficient statistic; live accumulation proven by the mtp78-collector |

**Build note (2026-08-10) — the 7.5 s planning number is 3× too slow.**
Measured on one real GLM-5.2 expert (layer 30, expert 137) on SM120:
**2.55 s** whole-expert K3 cold / **2.48 s** K4 cold (2.39 / 2.32 s with a
warm finalized H), K3 ≈ K4 within 3 % (`../runs/encode-bench/report.md`).
Every "7.5 s" and "~41 GPU-h" below should be read as **2.5 s** and
**~13 GPU-h** (one card; ~3.3 h on an idle quad); at the default 5 % encode
budget that is **~71 experts/hour**, so a freshly promoted ~100-expert
working set fully K4-encodes in ~1.4 h of background time. Two caveats the
bench did not cover: K-independence holds for K3/K4 only — on the real model
**K2 costs ~4.8 s/expert** and K5 ~3.4 s — and the routed-expert count is
**19,456** (76 MoE layers × 256, MTP included), not 19,712.

## The structural advantage

**Promotion candidates are hot experts; hot experts have well-fed live
Hessians.** The classic MoE calibration failure (rarely-routed experts
starved of samples — EAQuant's EA-CDB problem) cannot occur in the
promotion direction: an expert is only ever lazily encoded *because* the
live workload routes to it heavily. Lazy K4 encodes are calibrated on
fresher, denser, workload-true data than any offline campaign — for exactly
the tensors they touch. (TASA guard still applies: blend the live Hessian
with the stored general-corpus statistic; reweight, never replace.)

## D3′ (revised decision)

- **K3 base artifact stays mandatory** (~260 GB): boot floor, instant
  demotion path, and the fallback when a K4 encode hasn't landed yet.
- **K4 is a grow-on-demand cache**, not a shipped artifact:
  `VLLM_CACHE_ROOT/fq/k4/{manifest}/{layer}/{expert}/…`, regenerable,
  LRU-prunable, worst-case bounded by the old artifact size (347 GB), in
  practice proportional to experts ever promoted.
- `VLLM_FQ_K4_SOURCE = artifact | lazy | hybrid` — `artifact` keeps the old
  pre-encoded-pair fast path (boot-complete, for operators who want it);
  `hybrid` seeds from a partial artifact and lazily fills the rest.

## Pipeline (additions to 01/02)

1. **Pending state in the policy engine**: a proposed promotion whose K4
   tensor is not cached enters `pending`; the swap list only includes
   promotions whose encodes have landed. Demotions never pend (K3 base
   always available). Pending promotions don't count against swap caps
   until applied.
2. **Candidate Hessian accumulation — host-resident (DRAM), by design.**
   VRAM cannot host it: ~20 candidates × (151 MB H1 + 16.8 MB H2) ≈ 3.4 GB
   would come out of KV capacity. Architecture = the mtp78-collector v3
   pattern verbatim: device ring for candidate-expert activations (tens of
   MB VRAM total, the only GPU footprint) → zero host-blocking forward path
   → background drain over CUDA events → drain thread accumulates
   `H += xᵀx` into host fp32 buffers (batched as GEMM per drained block;
   fp32 with per-block partials or fp64 targets — DRAM is cheap). D2H
   bandwidth ~36 MB/s worst case (candidates receive ~T/32 of tokens each).
   Blend with the stored campaign statistic per TASA.
   - **Encode-time only**: upload the finished H once per encode
     (151 MB H2D ≈ 6 ms vs ~7.5 s Viterbi), encode, free. The Hessian
     visits the GPU; it never lives there.
   - **w2 subtlety**: H2 needs the expert's post-activation intermediate,
     which the fused MoE kernel never materializes. Selected approach:
     **side-stream recompute** from captured x
     (`act(x@w1ᵀ) ⊙ (x@w3ᵀ)`, three small GEMMs per drained block, on the
     encode executor's budget). Kernel modification to emit intermediates
     rejected as invasive. Minor bias from using currently-quantized w1/w3
     instead of BF16 — diluted by the campaign-Hessian blend.
3. **Encode executor**: background low-priority CUDA stream (or optional
   sidecar GPU), rate-limited (`VLLM_FQ_ENCODE_BUDGET_PCT`), one expert at
   a time, ~7.5 s each. Encode once globally (rank 0 or sidecar), write to
   shared cache, all ranks slice at 16-column tile granularity. Encode
   demand decays to zero as the allocation converges — total lifetime work
   ≈ (experts ever promoted) × 7.5 s ≈ tens of GPU-hours spread over days,
   vs ~41 GPU-h rental for all 19,712 experts, most never used.
4. **Cold-start option**: encode with the stored campaign Hessian
   immediately (no live accumulation wait), opportunistically re-encode
   later with the blended statistic. Two-tier quality, zero promotion
   latency beyond the encode itself.

## What changes elsewhere

- **P2 (encode venue/rental) is dissolved** — no campaign needed in `lazy`
  mode. P1 reduces to "does the K3 base's provenance satisfy the manifest";
  P3 becomes "which Ks the lazy encoder may target" (config, not campaign).
- **M0 shrinks** to: K3 base packaging + per-expert index + manifest +
  Hessian statistic export from the measure campaign (the campaign is still
  run once for ε curves — its Hessians are now *kept*, not discarded).
- **M4 grows** by the encode executor + pending-state plumbing (≈ +1 wk).
  Net schedule ≈ unchanged; cash cost → ~0.
- **T4 fidelity test** gains a leg: lazy-encoded tensor (fixed Hessian
  input) must equal the offline encoder's output bit-for-bit — the encoder
  is the same code either way, so this is a determinism check, not a
  quality tolerance.
- **New knobs**: `VLLM_FQ_K4_SOURCE` (default `lazy`),
  `VLLM_FQ_ENCODE_BUDGET_PCT` (default 5), `VLLM_FQ_HESSIAN_BLEND`
  (default 0.5 pending 0d), `VLLM_FQ_K4_CACHE_LIMIT_GB` (default unlimited).

## Honest costs of lazy vs artifact

| | artifact | lazy |
|---|---|---|
| Cash | ~41 GPU-h rental | ~0 |
| Promotion latency | ms (page from NVMe) | seconds–minutes (encode queue) — acceptable: promotions are never latency-critical |
| Serving interference | none | bounded by encode budget %; measure in T7 |
| Calibration | one campaign, uniform provenance | live-blended, per-expert fresher; provenance recorded per cached tensor (hessian blend hash) |
| Disk | 607 GB day one | 260 GB day one, grows with use |
| New-model day-one story | needs a campaign first | **K3 base + empty cache = deployable immediately** — strictly better fit for the product statement |

**Build note (2026-08-10):** the lazy-encode *plumbing* shipped ahead of the
executor. `VLLM_FQ_K_FALLBACK` lets boot substitute an available K for a
missing one (marked, never silent: `Fragment.requested_k`/`.substituted`,
and the serve's tier bitmap is written from the **loaded** Ks, not the
requested ones), and every substitution or hard miss appends to a persisted
`encode-queue.jsonl` drained by a worker CLI — `--drain` is a dry run that
validates BF16 and capture availability, `--execute` shells out to the
sha-pinned encoder (`../runs/loader-v2/trust-and-lazy-encode.md` §3).
Unbuilt so far: the live host-resident Hessian accumulation of §2, the
side-stream w2 recompute, and the encode executor's rate limiter. Note the
driver's unit of work is a **layer**, not an expert.

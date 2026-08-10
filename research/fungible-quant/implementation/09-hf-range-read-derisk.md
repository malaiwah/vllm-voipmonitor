# 09 — HF range-read de-risk (T3 remote source) — MEASURED, PASSED

Live test 2026-08-10 against `zai-org/GLM-5.2` (public, 282 shards), from a
proxied cloud sandbox (bandwidth-constrained — treat absolute MB/s as floor).
Method: `curl -sL -r <range>` against
`huggingface.co/<repo>/resolve/main/<shard>`; safetensors parsed by hand
(8-byte little-endian header length, JSON header, `data_offsets` + 8 + hlen).

## Results

| Check | Result |
|---|---|
| CDN honors Range after redirect | **Yes** — HTTP/2 206 Partial Content |
| Index (`model.safetensors.index.json`) | 5.4 MB, 59,585 tensors; expert→shard lookup trivial |
| Shard header | 8 B + 26.8 KB JSON, 213 tensors/shard; one fetch covers the expert |
| Tensor bytes | Exact: `[2048, 6144]` BF16 = 25,165,824 B per projection — **confirms the plan's I=2048/H=6144 arithmetic from the real checkpoint** |
| Whole expert (3 projections) | 75.5 MB — **byte-adjacent** (`down|gate|up` contiguous per expert) → one coalesced range read per expert |
| Locality | All 3 projections of layer-30/expert-137 in one shard (`model-00080`); shard layout is expert-major |
| Serial throughput | 14 MB/s (single connection, via proxy) → 5.4 s/expert |
| 4-way parallel | 104.9 MB in 2.37 s ≈ **44 MB/s** (3.2× scaling) → ~1.7 s/expert |
| Repeatability | Re-read slice byte-stable (sha match across requests) |

## Implications for the design

1. **T3-remote is real**: lazy per-expert BF16 sourcing works today with zero
   HF-side cooperation — no custom server, no full download, no token for
   public repos. Fetch (~2-5 s) pipelines cleanly ahead of the ~7.5 s encode;
   the encode queue, not the network, is the bottleneck even at proxy-floor
   bandwidth.
2. **Coalesce per expert**: one 75.5 MB range per expert (offsets adjacent),
   not three requests. Cache the 5.4 MB index + per-shard headers (27 KB each)
   at boot — then every expert fetch is exactly one request.
3. **Parallelism knob**: 4 concurrent streams ≈ 3.2× — the fetch backend
   should pool 4-8 connections (`VLLM_FQ_FETCH_STREAMS`, default 4).
4. Same mechanics serve the **T2 remote K-cache** (published encodes are also
   safetensors) — one fetch path implementation covers both tiers.

## Residual risks (small, named)

- Throughput here is proxy-floored; measure on the serving box (expect
  ~0.9 Gbit/s per the appliance repo's transfer logs → ~15 s/expert serial,
  still encode-dominated with 4 streams).
- Private repos need `Authorization: Bearer` on the resolve URL — supported
  by the same curl/httpx path, untested here.
- Xet-backed repos served fine via `resolve/` (this repo is CDN-fronted);
  if HF ever gates resolve for huge repos, `hf_transfer`/`HfFileSystem`
  offer the same ranged access — fallback exists.
- Shard layout (expert-major, 3-adjacent) is this repo's convention, not a
  safetensors guarantee — the fetch backend must trust `data_offsets`, and
  coalesce opportunistically (it does the right thing either way).

Test script: reproduced in-session (curl + 40-line python); worth landing as
`tools/fq_range_probe.py` in M0 so the check runs against any new model repo
as part of onboarding.

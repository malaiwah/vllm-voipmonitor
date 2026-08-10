# 11 — Slice/repack PoC: existing 3.0bpw quant → FQ segments — MEASURED, PASSED

Live PoC 2026-08-10 (`poc/poc_slice.py`, run against real HF repos, no GPU).

## Question answered

**Can the existing flat 3.0bpw quant be sliced into the FQ segment format and
distributed on HF — without downloading BF16 or re-encoding — while still
carrying provenance?** Yes, measured end to end:

| Step | Result |
|---|---|
| `brandonmusic/GLM-5.2-EXL3-TR3-3.0bpw` layout | **Already per-layer sharded** (`model-layer-030.safetensors`, 4.09 GB) — his convention converged on our segment shape |
| Per-expert addressability | 48 tensors per expert (3 proj × 4 ranks × trellis/suh/svh/mcg), **contiguous**: one coalesced 14.3 MB range read = one whole K3 expert (matches 3.375 MiB/rank × 4 + scales) |
| Provenance anchors, zero download | HF API `paths-info` returns the shard's **LFS sha256**; revision API returns the **commit hash** — materials pinned without fetching anything |
| Repack | Pure-python safetensors writer; segment `layer-030.e137.k3.safetensors` (14.3 MB), round-trip **byte-identical** |
| Attestation | `fq-attestation/1`, predicate **`repack-of`**: fragment sha256 + materials {repo, commit, file sha256, per-tensor offsets + digests} (`poc/attestation.sample.json`) |
| Signature | ed25519 sign + verify with stock openssl: **"Signature Verified Successfully"** |

## Provenance answer, precisely

- **Repack needs neither BF16 nor requant.** Trellis bytes are copied
  verbatim; the attestation predicate is `repack-of`, verifiable by anyone
  against brandonmusic's repo at the pinned commit (range-read + hash — the
  PoC does exactly this). The trust chain terminates at the source quant's
  reputation — the same trust the community already extends when loading it
  whole, now made explicit, pinned, and spot-checkable per fragment.
- **`encode-of` (chain to BF16) requires a re-encode** with recorded
  deterministic parameters. Legacy quants likely can't be bit-reproduced
  (their encoder params/Hessians weren't recorded), so they keep `repack-of`
  grandfather status; fresh `fq_encode` output starts the reproducible
  `encode-of` chain. Optional middle rung: name BF16 materials by the
  *original repo's* HF-published file sha256s + tensor offsets — also
  download-free — pinning what the quant claims descent from without proving
  fidelity.

## New facts with design impact

1. The bootstrap seed for GLM-5.2 costs **hours of repacking, zero GPU,
   zero BF16**: stream brandonmusic's shards → emit attested segments.
   M0's seed-import path is this PoC scaled up.
2. Source tensors are **rank-sliced (TP4-baked)**. v1 segments may keep
   rank granularity verbatim (repack is lossless either way); unsharding to
   the topology-neutral layout means concatenating rank slices along the
   16-column tile axis — mechanical, but needs a GPU fidelity check (T4
   harness) before it's default. Both layouts carry identical provenance.
3. `mcg` is a fourth per-tensor component (alongside trellis/suh/svh) —
   segment schema and swap-engine row inventory updated to carry it.
4. Xet-backed repo (`xet: true`) served ranged reads fine — 09's residual
   risk retired.

**Build note (2026-08-10) — scaled up, plus a second layout family.** The
PoC became `tools/fq_repack.py`: 76 layers of `brandonmusic@9297b9f1`
repacked and published, and the all-K3 reassembly is **79/79 shards
sha256-identical** at full-model scale (`../runs/m0-assemble/`). The
48-tensors-per-expert / 14.3 MB figure above is the **`per_expert_v1`**
layout. A second family exists in the wild: **`shared_h_v1`** (e.g.
`willfalco/GLM-5.2-EXL3-TR3-3.42bpw`) keeps only **36 tensors per expert**
plus **12 shared H-side rows per layer** — expert unit 14,168,112 B (K3) /
18,886,704 B (K4), shared profile 147,456 B/layer. The two are convertible
in exactly one direction: replicating the shared rows into each expert
yields a byte-size-exact `per_expert_v1` unit
(14,168,112 + 147,456 = 14,315,568 B, matching item 2 above), predicate
**`derived-from`** (bytes are added, so not `repack-of`), and the expanded
units are mechanically mixable with the brandonmusic base — same
trellis/mcg geometry, same mcg multiplier, same calibration corpus. The
reverse (deduplicating per-expert artifacts into shared-H form) and
re-basing a shared-H trellis onto different H rows are both **impossible
without re-encode**. One-expert numeric decode check still pending.
(`../runs/0c-campaign/quant-342-layout-report.md`.)

# Reconstruction table — community GLM-5.2 quants from Progressive Tensors segments

Date: 2026-08-10. Proves the community-reassembly claim end to end: each
published quant reconstructed from its segments and VALIDATED — byte-identity
where it should hold, bounded numeric similarity where it cannot. Every row
was produced by `tools/fq_verify.py`; machine-readable reports live under
`runs/0c-campaign/verify/`. Public companion:
`progressive-tensors/docs/RECONSTRUCTION.md`.

## Subjects and pins

| Quant | Revision | Layout | Segment family |
|---|---|---|---|
| brandonmusic/GLM-5.2-EXL3-TR3-3.0bpw | `9297b9f1d53af5c67cffa01e30cc071a1ff7144b` | per_expert_v1, flat K3 | `~/fq-segments/GLM-5.2-EXL3-FQ` (hf: `malaiwah/GLM-5.2-EXL3-FQ-segments`), layers 3–78, repack-of |
| willfalco/GLM-5.2-EXL3-TR3-3.36bpw | `8d9aa923a17502675ca23737349b67f2e66bb69d` | per_expert_v1, mixed K3/K4 | `~/fq-primed/segments-336`, K4 only, layers 3–10, repack-of |
| willfalco/GLM-5.2-EXL3-TR3-3.42bpw | `ae68c65947efa90bea37308e15421872f124c46d` | shared_h_v1, mixed K3/K4 | `~/fq-primed/segments-342/shared-h` (repack-of) + `.../expanded` (derived-from), layers 3–10 |
| zai-org/GLM-5.2 | `b4734de4` (local snapshot) | BF16 base | ground truth, similarity only |

Layer windows are honest scope statements: the willfalco families were primed
for layers 3–10, so their byte-identity is proven per fragment inside that
window. Whole-shard identity is applicable — and claimed — only for the fully
repacked brandonmusic family.

## Byte-identity rungs

| # | Reconstruction | Proof granularity | Result | Report |
|---|---|---|---|---|
| 1 | brandonmusic 3.0bpw: **all 76 MoE shards** (layers 3–78) stream-reassembled from K3 segments + source non-expert bytes | whole-shard sha256 vs source `MANIFEST.sha256` | **PASS 76/76** — 278,523,691,008 B of expert bytes from segments, 31,684,790,520 B source pass-through, zero mismatches | `verify/identity-k3-local.{json,md}` |
| 2 | brandonmusic segments vs signed attestations (sampled layers 6, 17, 38) | per-expert span sha256 + ed25519 | **PASS** — 768 expert spans re-hashed, 0 mismatches, signatures verified | same run |
| 3 | willfalco 3.36bpw K4 fragments (layers 3–10) | fragment bytes vs **fresh** ranged re-reads of the pin | **PASS** — 722/722 spans vs attestations (sig verified); 24/24 sampled experts byte-equal to freshly re-fetched source bytes (469 MB re-fetched) | `verify/identity-336-remote.{json,md}` |
| 4 | willfalco 3.42bpw shared-h fragments + per-layer shared profiles (layers 3–10) | fragment bytes vs fresh ranged re-reads; profiles compared in full | **PASS** — 2048/2048 spans vs attestations (all sigs verified); 48/48 sampled experts byte-equal to freshly re-fetched source bytes; 8/8 shared profiles byte-equal (804 MB re-fetched) | `verify/identity-342-remote.{json,md}` |
| 5 | willfalco 3.42bpw **expanded** per-expert view (derived-from) | full re-derivation, every tensor byte-compared, parent sha256 pins re-hashed | **PASS 2048/2048 experts** across 16 segments — 32,819,478,528 B verbatim + 301,989,888 B replicated H-rows, 0 tensor mismatches, all parent pins intact | `verify/identity-342-derived.{json,md}` |

Row 5 is the numerically interesting one: the shared-H → per-expert expansion
is the operation that makes a `shared_h_v1` quant mixable with the
`per_expert_v1` base. It is exact by construction, and now exact by
measurement over the entire primed window, not a sample.

## Numeric rung — two independent producers, same fragments interoperating

24 experts spread over layers 3–10 (seed 42; experts K4 in both willfalco
quants), all three projections, 4 rank slices each, decoded with the
reference exllamav3 path (`LinearEXL3.get_weight_tensor` = `ext.reconstruct`
+ Hadamard-128 + `diag(suh)/diag(svh)`), metrics in fp64. n = 72 per pair
(24 experts × 3 projections).

| Pair | cos mean / min | relF mean / max | max abs | reading |
|---|---|---|---|---|
| 3.42-expanded vs 3.42-shared-h | **1.00000 / 1.00000** (bitwise EQUAL) | 0.0000 / 0.0000 | 0.0000 | exact expansion, proven at the decoded-weight level |
| 3.42-K4 vs BF16 | 0.99684 / 0.99625 | 0.0793 / 0.0866 | 0.0129 | producer A's K4 quality |
| 3.36-K4 vs BF16 | 0.99684 / 0.99626 | 0.0792 / 0.0864 | 0.0142 | producer B's K4 quality — **statistically indistinguishable from A** |
| 3.42-K4 vs 3.36-K4 | 0.99370 / 0.99255 | 0.1120 / 0.1221 | 0.0196 | two independent K4 encodes of the same weights |
| brandonmusic-K3 vs BF16 | 0.98761 / 0.98566 | 0.1569 / 0.1692 | 0.0284 | K3 tier, for scale |
| brandonmusic-K3 vs 3.42-K4 | 0.98452 / 0.98229 | 0.1754 / 0.1881 | 0.0340 | cross-producer, cross-bitrate |
| brandonmusic-K3 vs 3.36-K4 | 0.98466 / 0.98245 | 0.1746 / 0.1873 | 0.0305 | cross-producer, cross-bitrate |

**This is the two-independent-producers evidence** the prior-art review
flagged as missing. Three unrelated producers (brandonmusic, willfalco ×2
distinct encode runs with different layouts and different K partitions)
each emitted fragments for the *same* (layer, expert, projection) of the
*same* base model; all three were decomposed by the same segment tooling and
decoded through the same reference path, and all three land within
0.0079 relative Frobenius error of each other's distance-to-ground-truth:

- Both K4 producers sit at relF ≈ 0.079 from BF16 — a 0.0001 spread between
  independent encodes, i.e. the fragments are *interchangeable in quality*,
  not merely loadable.
- Two independent K4 encodes differ from **each other** (relF 0.112) by more
  than either differs from ground truth (0.079) — exactly the geometry of two
  unbiased quantizers scattering independently around the same target
  (√2 × 0.079 = 0.112, observed 0.112). That is a positive result: the
  encodes are independent, not copies, and they agree on the weight.
- The K3↔K4 ordering is monotone in bitrate everywhere (K3 relF 0.157 vs K4
  0.079, a 1.98× ratio against the campaign's measured ~1.96× per-K error
  ladder), so a mixed recipe drawn from either producer's K4 fragments buys
  the same quality step over the K3 base.

What is **not** claimed: cross-producer fragments are not bit-identical, and
nothing here asserts they should be. The claim is bounded: the same expert
slot, filled from either producer, decodes to weights that are equally close
to the BF16 original.

## Related measurements

- `verify/reflink-xfs-measurement.md` — `fq_assemble --reflink` on this box's
  XFS (Ceph RBD): byte-identity always holds (plain and reflink outputs
  sha256-identical to each other and to the source shards), `copy_file_range`
  used for all 12,288 expert regions/layer with zero fallbacks, run somewhat
  faster (6.3 s → 3.8 s for 3 shards) — but **zero extents shared**
  (`filefrag`: no shared flags; identical block usage). Cause measured:
  0.00 % of expert bytes are 4K-congruent between segment and shard offsets
  (213 distinct residues, none zero). Positive control with aligned offsets
  shares extents immediately, so the limit is layout alignment, not the
  kernel. `df` deltas were unusable (±16 GB swings from concurrent priming);
  extent maps are the authoritative measurement.
- Priming transport: 3.36 = 13.74 GB in 464 range requests (full-download
  counterfactual 36.04 GB, 62 % saved — K4 experts only); 3.42 = 32.82 GB in
  254 requests (counterfactual 36.14 GB, 9 % saved — all experts + profiles;
  most of a MoE shard *is* experts). Selectivity, not compression, is where
  the saving comes from.

## Reproduce

```bash
V=runs/0c-campaign/verify
# rows 1-2 (local snapshot)
python tools/fq_verify.py --identity --segments ~/fq-segments/GLM-5.2-EXL3-FQ \
  --source <brandonmusic-snapshot> --attest 3 --seed 42 \
  --json $V/identity-k3-local.json --md $V/identity-k3-local.md
# rows 3-4 (network: fresh ranged reads of the pinned revisions)
python tools/fq_verify.py --identity --segments ~/fq-primed/segments-336 \
  --sample 3 --seed 42 --json $V/identity-336-remote.json
python tools/fq_verify.py --identity --segments ~/fq-primed/segments-342/shared-h \
  --sample 3 --seed 42 --json $V/identity-342-remote.json
# row 5 (local, full coverage)
python tools/fq_verify.py --identity --segments ~/fq-primed/segments-342/expanded \
  --parent ~/fq-primed/segments-342/shared-h --json $V/identity-342-derived.json
# numeric rung (GPU with the exllamav3 ext; ~1 min, <1 GB VRAM)
CUDA_VISIBLE_DEVICES=<free> runs/gg-env/gg-run.sh python tools/fq_verify.py --similarity \
  --family bm-k3=~/fq-segments/GLM-5.2-EXL3-FQ,k=3 \
  --family 342-k4=~/fq-primed/segments-342/expanded,k=4 \
  --family 342-sh=~/fq-primed/segments-342/shared-h,k=4 \
  --family 336-k4=~/fq-primed/segments-336,k=4 \
  --bf16 <zai-org-GLM-5.2-snapshot> --layers 3-10 --experts 24 --seed 42 \
  --json $V/similarity-3quants.json --md $V/similarity-3quants.md
```

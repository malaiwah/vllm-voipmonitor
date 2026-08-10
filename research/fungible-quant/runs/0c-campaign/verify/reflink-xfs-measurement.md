# Measured: does `fq_assemble --reflink` actually share extents on this XFS?

Date: 2026-08-10. Box: `/home` = XFS on Ceph RBD (`/dev/rbd2`, kernel
6.8.0-71-generic, `cp --reflink=always` works, so the filesystem itself is
reflink-capable). Question under test: the `--reflink` mode's tests assert
byte-identity and that `copy_file_range` was used, but deliberately do not
assert extent sharing. On this filesystem we can measure it for real.

## Procedure

Assembled layers 3–5 of GLM-5.2 twice from the K3 family
(`/home/mbelleau/fq-segments/GLM-5.2-EXL3-FQ`, brandonmusic lineage) with the
local source snapshot: once plain, once `--reflink`, everything on the same
XFS. Then compared sha256, wall time, `df` deltas, `du` (block vs apparent),
and `filefrag -v` physical extent maps of outputs vs segment files.

## Results

| Measure | plain | `--reflink` |
|---|---|---|
| wall time (3 MoE shards, warm-ish cache) | 6.3 s | 3.8 s |
| expert regions copied | 12,288/layer via read+write | 12,288/layer via `copy_file_range` (0 fallbacks) |
| output sha256 (3 shards) | == source MANIFEST | == source MANIFEST, == plain run |
| `du` block usage of output dir | 16,132,894,720 B | 16,132,894,720 B (identical) |
| `filefrag -v` shared-flag extents in outputs | 0 | **0** |

**Verdict: byte-identity holds, `copy_file_range` is used throughout and is
somewhat faster (server-side copy, no user-space bounce) — but on this
filesystem, for this real checkpoint, NO extents end up shared. `--reflink`
saved zero bytes here.**

`df` deltas were unusable as evidence in either direction: adjacent snapshots
swung by ±16 GB from the concurrent priming extraction and Ceph RBD's
asynchronous allocation, so the extent maps above are the authoritative
measurement.

## Why sharing fails: alignment, measured

XFS can only share blocks when source and destination file offsets of the
copied region are congruent modulo the 4096-byte block. For layer 3:

- segment body offset 1,503,224 (≡ 4088 mod 4096); source shard body offset
  1,507,936 (≡ 608 mod 4096);
- across all 12,288 expert tensors, `(segment_offset − shard_offset) mod 4096`
  takes 213 distinct values and is **never 0**: exactly **0.00 % of the
  3.66 GB of expert bytes are 4K-congruent**.

The canonical per-expert reordering of the segment layout plus the differing
safetensors header lengths shift every tensor's offset; with no congruent
regions, the kernel's `remap_file_range` attempt fails internally for every
region and `copy_file_range` silently performs a (server-side) plain copy —
which is precisely the behavior the mode's caveats predicted.

## Positive control: the syscall + filesystem do reflink when aligned

Copying 1 GiB of the layer-3 segment file at offsets (0, 0) — 4K-congruent —
with the same `os.copy_file_range` loop produced a destination whose single
extent carries the `shared` flag at the same physical blocks as the source
(`filefrag -v`: `171704384..171966527 … shared` on both files). So the
machinery works on this XFS; the negative result above is purely the offset
congruence of the real safetensors layouts, not kernel or filesystem refusal.

## Implication

On this box `--reflink` is safe (bytes always identical — re-verified here on
real shards) and mildly faster, but it is not a disk-space optimization for
the GLM-5.2 families as laid out today. If extent sharing is ever wanted for
real, a future `fq-segment/2` could pad each tensor's segment offset so that
`(segment_body + offset) ≡ (source_body + offset) mod 4096` per tensor
(≤ 4 KB padding per tensor, sub-1 % size overhead) — only then can XFS share
the interior blocks.

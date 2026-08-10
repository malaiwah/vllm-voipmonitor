# Priming report — community-quant K3/K4 fragments, window 1 (layers 3-10)

Date: 2026-08-10. First execution of the community-quant priming pass
(#22): per-expert K3/K4 fragments extracted from two published GLM-5.2
mixed quants **via ranged HTTP reads only** — safetensors headers plus
exactly the needed expert byte-ranges; no shard was downloaded in full.
Tool: `tools/fq_prime.py` (new, tested by `tools/test_fq_prime.py`);
extraction procedure and layout facts per
`quant-342-layout-report.md`. Network + CPU only; no GPU was touched
(ring encodes owned 4-7, a serve owned 0-3 throughout).

## Sources (pinned)

| Source | Revision | Layout | Scope taken |
|---|---|---|---|
| `willfalco/GLM-5.2-EXL3-TR3-3.42bpw` | `ae68c659...f124c46d` | `shared_h_v1` | layers 3-10, K3+K4, **both families** |
| `willfalco/GLM-5.2-EXL3-TR3-3.36bpw` | `8d9aa923...e66bb69d` | `per_expert_v1` | layers 3-10, **K4 only** |

K identified from trellis geometry in the fetched headers (last dim =
16*K), never from metadata alone; `tier_bitmap.json` agreed with the
header-derived per-expert K on **all 16 shard headers** (FULL per-expert
agreement). Per-expert tensor-set signatures validated against the
detected rotation layout before any payload fetch.

Header facts confirmed at run time: experts are stored contiguously in
**lexicographic** id order (0, 1, 10, 100, ...) with **alphabetical**
within-expert tensor order; fq_prime re-orders into the fq_repack
canonical order (ascending int id; trellis, suh, svh, mcg per
proj/rank) while writing, via a preallocated pwrite writer.

New fact surfaced by the 3.36 headers (r26 notes only covered layers
4-77): **3.36 layer 3 is 206 K3 + 50 K4 — the same partition as 3.42
layer 3.** Bulk layers 4-10: 160 K3 + 96 K4/layer (96 K4 as the r26
notes claim).

## Emitted families and fragment counts

Output root: `/home/mbelleau/fq-primed/`. Published (see below) to
`malaiwah/GLM-5.2-EXL3-FQ-segments` under `sources/` — deliberately
separate from the brandonmusic-derived K3 base at the repo root.

| Family | Predicate / layout | K3 frags | K4 frags | Unit bytes (K3 / K4) | Payload |
|---|---|---|---|---|---|
| `segments-342/shared-h` | repack-of / `shared_h_v1` | 1,242 | 806 | 14,168,112 / 18,886,704 | 32,819,478,528 B |
| `segments-342/shared-h` profiles | repack-of / `shared_h_v1` | 8 fragments x 12 rows | — | 147,456 per layer | 1,179,648 B |
| `segments-342/expanded` | **derived-from** / `rank_sliced_tp4` | 1,242 | 806 | 14,315,568 / 19,034,160 | 33,121,468,416 B |
| `segments-336` | repack-of / `rank_sliced_tp4` | — | 722 | — / 19,034,160 | 13,742,663,520 B |

- Counts check out exactly: 1,242 = 206 + 7x148; 806 = 50 + 7x108;
  722 = 50 + 7x96. Every unit of a given (family, K) has the identical
  byte size listed — no deviating expert anywhere.
- **Size identities from the layout report reproduced on real data**:
  expanded K3 unit 14,315,568 = 14,168,112 + 147,456 = native
  `per_expert_v1` K3 unit; expanded K4 unit 19,034,160 B is
  **byte-size- and signature-identical to the native 3.36 K4 unit** —
  the two artifacts' K4 experts are mechanically interchangeable at the
  loader level (mixing remains a quality decision).
- The expansion is the exact replication rule (`shared_h_expand_v1`):
  the layer's 12 shared rows are copied into each expert's
  `{gate,up}_proj.rank{R}.suh` / `down_proj.rank{R}.svh` slots;
  trellis/mcg/expert-local bytes stay byte-identical. Bit-exactness is
  asserted in `test_fq_prime.py` (expanded unit == hand-built
  per-expert unit) and the derived-from attestations carry parent refs
  (expert segment sha + profile sha) plus the rule id.
- Every shared-h expert segment pins its layer profile
  (`shared_profile_file` + sha in segment metadata; hard dependency map
  in `fq-manifest.json`) — an expert fragment is decodable only with
  its layer's 12 shared rows.

## Bytes fetched vs full-download counterfactual

| Source | Fetched | Range requests | Full-download counterfactual | Fetched % |
|---|---|---|---|---|
| 3.42bpw (layers 3-10) | 32,820,658,176 B (30.6 GiB) | 254 | 36,136,952,328 B | 90.8 % |
| 3.36bpw (layers 3-10) | 13,742,663,520 B (12.8 GiB) | 464 | 36,044,355,408 B | 38.1 % |
| **Total** | **46,563,321,696 B (43.4 GiB)** | **718** | **72,181,307,736 B (67.2 GiB)** | **64.5 %** |

25.6 GB (35.5 %) never left Hugging Face. The 3.42 ratio is high
because the pass takes all 256 experts of every layer (the skipped
bytes are the ~408 MiB/layer of unquantized attention/norms/gate plus
headers); the 3.36 K4-only pass shows the real selective-fetch shape —
38 % of the shard bytes for 37.5 % of the experts. A future pass that
cherry-picks experts (e.g. top-phi upgrades only) inherits the 3.36
economics. The whole-layer counterfactual is itself conservative: the
naive alternative downloads the full repos (351.6 GB / ~350 GB), not
just layers 3-10.

Transport profile: strictly-adjacent range coalescing capped at 128 MiB
per GET; ~29-31 payload requests/layer on 3.42 (all experts contiguous
-> near-pure sequential chunks), 42-64 on 3.36 (K4 experts scattered
into runs between skipped K3 spans). Paced at >= 1 s between request
starts, single stream per source, two sources in parallel (combined
request start rate ~0.1/s — far under the ~1/s account budget; each
128 MiB chunk takes tens of seconds). Achieved ~6 MB/s single-stream,
~10 MB/s with both streams (per-connection CDN throttling, shared with
the paced uploader running on the same box). Wall time ~100 min for
3.42 (including one preemption + clean per-layer resume) and ~45 min
for 3.36.

## Attestations

Every fragment carries a signed `fq-attestation/1` line (ed25519, the
existing campaign key, pubkey in each `fq-manifest.json`), with the
v2-direction provenance extras as additional payload keys:

- `materials`: repo, pinned revision, shard name, shard sha256 (from
  the source repo's own `MANIFEST.sha256`);
- `expert_sha256` (digest per expert unit) + `expert_k` (per-expert
  bitrate ride-along for cold-start policy) + `source_spans` (absolute
  source byte ranges per expert);
- `source_config` (rotation_layout, producer_version,
  exllamav3_version, mcg_multiplier, expert_bpw_mean) and
  `calibration_corpus_sha256` (`cf247acc...` — same corpus as the
  brandonmusic artifacts, both sources);
- derived-from lines add `parents` (expert-segment + profile refs with
  shas) and `derivation.rule = shared_h_expand_v1`.

## Validation

1. **Unit-size sweep**: every expert range in every index matches the
   expected unit size for its (family, K) exactly (tables above).
2. **tier_bitmap cross-check**: FULL agreement, all 16 layers.
3. **Transport spot-check** (`fq_prime.py spot-check`, seed 20260810):
   3 random experts per source re-fetched **independently** from HF
   (fresh header + payload range), sha256 of the canonical-order
   concatenation compared against the local segment span and the
   signed attestation digest:

   | Source | Samples | Verdict |
   |---|---|---|
   | 3.42 shared-h | L10 k4 e40; L3 k3 e157; L4 k4 e215 | **3/3 OK** (55.4 MB re-fetched) |
   | 3.36 | L5 k4 e82; L10 k4 e225; L6 k4 e164 | **3/3 OK** (61.6 MB re-fetched) |

   All three digests (source, segment, attestation) identical in every
   sample.
4. **Tool tests**: `test_fq_prime.py` — K identification, coalescing,
   both-family emission, expansion bit-exactness, attestation
   verification, resume idempotence (zero HTTP calls on a done tree),
   K-filter fetch-avoidance (no K3 byte is ever requested on a K4-only
   pass), spot-check corruption detection. Green in this tree and in
   `progressive-tensors` (with the whole tools suite).

## Publication

Published to `malaiwah/GLM-5.2-EXL3-FQ-segments` under source-scoped
paths (collision-free with the K3 seed family at the repo root):

```
sources/willfalco-3.42bpw/shared-h/   # verbatim shared_h_v1 family + profiles
sources/willfalco-3.42bpw/expanded/   # derived per-expert view (rank_sliced_tp4)
sources/willfalco-3.36bpw/            # verbatim K4 per_expert_v1 family
sources/fq-manifest.json              # multi-source manifest (repo-scoped paths)
```

Each subtree carries its own `index-k*.json`, `attestations/`, and
`fq-manifest.json` exactly as built locally; the repo card gained a
`sources/` section describing the layout and the two 3.42 families.

## What's next

- Remaining layers 11-78 of both sources follow opportunistically
  (same tool, same state-resume; ~284 GB payload for the full 3.42
  shared-h family per the layout report).
- The one-expert numeric decode equivalence check (shared-h loader vs
  expanded per-expert loader on the same expert) remains the last
  rung before mixing expanded 3.42 experts into brandonmusic-based
  assemblies — transport-level identity is now proven, decode-algebra
  identity is asserted by the layout docs and should be spot-verified
  when a GPU window opens.

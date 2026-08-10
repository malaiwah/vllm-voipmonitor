# Layout report: willfalco/GLM-5.2-EXL3-TR3-3.42bpw (pinned ae68c659)

Date: 2026-08-10. Empirical, byte-level verification of the tensor layout of
`willfalco/GLM-5.2-EXL3-TR3-3.42bpw` at revision
`ae68c65947efa90bea37308e15421872f124c46d`, and a salvage analysis for
Progressive Tensors fragment extraction. All evidence below was obtained
read-only from the HF API and ranged HTTP reads (safetensors headers plus a
few KiB of vector payloads); no shard was downloaded in full.

## Verdict

**Layout: `shared_h_v1` — confirmed from bytes, not just metadata.**

The published repo really has the shared-H layout the docs
(`glm5.2_v20.md` r28 gate, `shared_h_quant.md`) claim. Per MoE layer there are
exactly 12 per-layer shared H-side rows
(`...mlp.experts.shared_h.{gate_proj,up_proj}.rank{0-3}.suh` and
`...shared_h.down_proj.rank{0-3}.svh`, each `F16 [6144]`), and every routed
expert carries only 36 tensors (trellis + mcg + one expert-local
intermediate-side vector per projection per rank) instead of the 48 tensors of
`per_expert_v1`. The per-expert gate/up `suh [6144]` and down `svh [6144]` of
the legacy layout are absent, exactly as `shared_h_v1` specifies.

## Source pins and provenance

| Artifact | Revision | Role |
|---|---|---|
| `willfalco/GLM-5.2-EXL3-TR3-3.42bpw` | `ae68c65947efa90bea37308e15421872f124c46d` | subject (r28 shared-H) |
| `willfalco/GLM-5.2-EXL3-TR3-3.36bpw` | `8d9aa923a17502675ca23737349b67f2e66bb69d` | triangulation (per-expert, mixed-K) |
| `brandonmusic/GLM-5.2-EXL3-TR3-3.0bpw` | `9297b9f1d53af5c67cffa01e30cc071a1ff7144b` (local snapshot) | triangulation (per-expert, K3) |

Note: `main` of the 3.42 repo has since moved to
`a350292cb2038f2c31732569a711a89e5d72fd46`; the pinned revision still resolves.
Repo tree at the pin: 93 entries, 351.6 GB total; 79 layer shards
(`model-layer-000..078.safetensors`) + `model-embed` / `model-head` +
`model.safetensors.index.json` (63,326,305 B) + small metadata files.

Shard SHA-256 (from repo `MANIFEST.sha256`, 92 entries), for the shards whose
headers were inspected:

```
ee857fe9c5727a870eb151e763a01edf11370265f47c98fd658050923c587b2c  model-layer-003.safetensors
ba1b0cd8125ff30bbbfc6c764b3daf8c5e1d07a6c07ca0c49a5d91a9dea4943a  model-layer-030.safetensors
6b6db28986239adc0f0eca5d50912a7597aa3ab1eedfbf1a788afab8e6962d1e  model-layer-078.safetensors
```

Shard sizes at the pin: layer 3 = 4,272,963,040 B; layer 30 = 4,546,650,816 B;
layer 78 (MTP) = 4,206,780,104 B; dense layers 0–2 = 801,799,648 B each (BF16,
unquantized per `quantization_config.ignore`).

## Config metadata (config.json at pin)

`hybrid_tr3_tail` declares (selected keys):

```
rotation_layout        = "shared_h_v1"
producer_version       = "shared-h-v1"
mtp78.rotation_layout  = "shared_h_v1"
expert_bpw_mean        = 3.418854
exllamav3_version      = "0.0.43"
bits                   = "mixed"; bits_per_expert = "tier_bitmap.json:k"
tensor_schema          = "model.layers.{L}.mlp.experts.{E}.{proj}.rank{r}.{trellis|suh|svh|mcg}"
shared_h_tensor_schema = "model.layers.{L}.mlp.experts.shared_h.{proj}.rank{r}.{suh|svh}"
scope.quantized        = "routed MoE expert gate/up/down projections, all 256 experts, layers 3..78 (incl. MTP draft layer)"
```

Model: `GlmMoeDsaForCausalLM`, 78 hidden layers + 1 MTP (layer index 78),
hidden 6144, 256 routed experts, moe_intermediate 2048, TP4 artifact
(`calibration_manifest.json`: `output_tp: 4`, `capture_tp: 8`; rank indices
0–3 in tensor names). Calibration corpus `corpus_sha256 =
cf247acc7c5da9f0600c7d6ab3b7c2fcfc54ec30b794e3b6047559285fa44df4` — the same
`reap_recall_calib.jsonl` as the brandonmusic `calibration_encoder` bundle
(matches the hash table in `shared_h_quant.md`), i.e. same calibration data as
the per-expert artifacts.

## Header-verified tensor patterns (3.42bpw)

Safetensors headers fetched by ranged read (first 8 bytes -> header length,
then the JSON): layer 3 (header 1,136,088 B, 9,242 tensors), layer 30
(1,146,176 B, 9,247), layer 78 (1,107,136 B, 9,251). Collapsing expert index
`{E}` (0–255) and rank `{R}` (0–3), the MoE payload of **every** sampled layer
is exactly:

```
x    4  model.layers.{L}.mlp.experts.shared_h.gate_proj.rank{R}.suh   F16 [6144]
x    4  model.layers.{L}.mlp.experts.shared_h.up_proj.rank{R}.suh     F16 [6144]
x    4  model.layers.{L}.mlp.experts.shared_h.down_proj.rank{R}.svh   F16 [6144]
x 1024  model.layers.{L}.mlp.experts.{E}.gate_proj.rank{R}.trellis    I16 [384, 32, 48|64]
x 1024  model.layers.{L}.mlp.experts.{E}.gate_proj.rank{R}.svh        F16 [512]
x 1024  model.layers.{L}.mlp.experts.{E}.gate_proj.rank{R}.mcg        I32 []
x 1024  model.layers.{L}.mlp.experts.{E}.up_proj.rank{R}.trellis      I16 [384, 32, 48|64]
x 1024  model.layers.{L}.mlp.experts.{E}.up_proj.rank{R}.svh          F16 [512]
x 1024  model.layers.{L}.mlp.experts.{E}.up_proj.rank{R}.mcg          I32 []
x 1024  model.layers.{L}.mlp.experts.{E}.down_proj.rank{R}.trellis    I16 [32, 384, 48|64]
x 1024  model.layers.{L}.mlp.experts.{E}.down_proj.rank{R}.suh        F16 [512]
x 1024  model.layers.{L}.mlp.experts.{E}.down_proj.rank{R}.mcg        I32 []
```

- 256 experts x 36 tensors + 12 shared = **9,228 EXL3 tensors per MoE layer**,
  matching release gate #2 in `shared_h_quant.md` exactly.
- Per expert there is **no** `gate_proj.*.suh`, `up_proj.*.suh`, or
  `down_proj.*.svh` — the H-side rows exist only under `shared_h.`. Pattern
  counts of exactly 1024 (=256x4) per remaining pattern prove no expert
  deviates.
- Trellis tile geometry: gate/up `[in_tiles=384, out_tiles=32, 16*K]`
  (6144 -> 512 per rank), down `[32, 384, 16*K]` (512 -> 6144). Last dim 48 =
  K3, 64 = K4.
- Non-expert tensors per MoE shard: 19 BF16/F32 tensors (MLA attention incl.
  DSA indexer from layer 5 up, `mlp.gate.*`, `mlp.shared_experts.*`,
  layernorms), 427,457,024 B (407.7 MiB) at layer 30. Layer 78 (MTP) adds
  `eh_proj.weight BF16 [6144, 12288]`, `enorm`, `hnorm`,
  `shared_head.norm.weight` (eh_proj deliberately unquantized per config
  ignore list).

## K partition (header-verified vs tier_bitmap.json)

`tier_bitmap.json` (922,664 B) holds, per layer 3..78: `k` (256-entry list of
3/4), `expert_rel_rt_mse`, `keep_nvfp4`, `tail_tr3`. Counting the **actual
trellis last dims in the headers** per expert (all 12 trellis tensors of each
expert agree on K; no intra-expert mixing):

| Layer | K3 (lastdim 48) | K4 (lastdim 64) | tier_bitmap agreement |
|---|---|---|---|
| 3 | 206 | 50 | FULL (per-expert) |
| 30 | 148 | 108 | FULL (per-expert) |
| 78 (MTP) | 256 | 0 | FULL (per-expert) |

tier_bitmap for the unfetched layers: **every layer 4–77 is 148 K3 + 108 K4**.
Global totals: **11,414 K3 + 8,042 K4 = 19,456 experts across 76 MoE layers**.
This confirms the doc claims ("206 K3 + 50 K4 in layer 3, 148 K3 + 108 K4 in
layers 4-77, 256 K3 in layer 78") and the "148/108" partition for the bulk
layers. Mean bpw 3.418854 per config.

## Triangulation (same layer 30, all three artifacts)

| Artifact | Layout | Tensors/expert | Expert tensor set (gate/up ; down) | L30 K split |
|---|---|---|---|---|
| brandonmusic 3.0bpw (local header) | `per_expert_v1` | 48 (12,307 total in shard) | trellis + mcg + `suh F16 [6144]` + `svh F16 [512]` ; trellis + mcg + `suh F16 [512]` + `svh F16 [6144]` | 256 K3 / 0 K4 |
| willfalco 3.36bpw @ 8d9aa923 (ranged header) | `per_expert_v1` | 48 (12,307 total) | same as brandonmusic | 160 K3 / 96 K4 |
| willfalco 3.42bpw @ ae68c659 (ranged header) | `shared_h_v1` | 36 (9,247 total) | trellis + mcg + `svh F16 [512]` (+ per-layer shared `suh [6144]`) ; trellis + mcg + `suh F16 [512]` (+ per-layer shared `svh [6144]`) | 148 K3 / 108 K4 |

Neither per-expert artifact contains any `shared_h.` tensor; the 3.42 artifact
contains no per-expert H-side row. The three artifacts share trellis/mcg
geometry and the mcg multiplier (see below), so the only structural difference
is where the H-side rows live.

## Byte-level probes (small ranged reads, layer 30 of 3.42bpw)

Fetched payload bytes for 9 tensors (~50 KiB total) to validate semantics:

- `shared_h.gate_proj.rank0.suh` `[6144]`: values like `-0.013306, -0.013412,
  +0.013298, ...` — a **sign vector times a smooth ~0.0134 magnitude profile**
  (|v| in [0.01366, 0.01363] band; not ±1). `shared_h.up_proj.rank0.suh` is a
  distinct vector (not equal to gate's), magnitude ~0.0132.
- `shared_h.down_proj.rank0.svh` `[6144]`: signed values in ±[0.996, 1.024] —
  sign vector with near-unit magnitude profile.
- Expert-local `experts.{0,137}.gate_proj.rank0.svh` `[512]`: signed values
  with per-element magnitudes 0.95–1.65, **different between experts** — this
  is where the expert `g_scale` now lives, exactly as documented ("gate/up ...
  move each expert's scalar g_scale from shared SU to expert-local SV").
- Expert-local `experts.{0,137}.down_proj.rank0.suh` `[512]`: sign vector x
  ~0.014 magnitude, different between experts ("down retains expert-local SU").
- `mcg` scalars: `I32 [] = -877912083` (unsigned `3417055213`) for both probed
  experts — the constant MCG multiplier, identical to
  `hybrid_tr3_tail.mcg_multiplier = 3417055213` declared in the 3.36bpw
  config.

Layout-relevant file geometry: each expert's 36 tensors are **contiguous** in
the shard (single coalesced range read per expert: 14,168,112 B for K3,
18,886,704 B for K4, verified sum == span), and the 12 shared_h tensors form
one contiguous 147,456 B block. Same coalesced-extract pattern as
`poc/poc_slice.py` used on brandonmusic.

## Salvage analysis

Exact unit sizes (from layer-30 header offsets; identical algebra for every
layer):

| Unit | Tensors | Bytes |
|---|---|---|
| 3.42 expert, K3 (12 trellis `[.,.,48]` + 12 mcg + 12 local `[512]`) | 36 | 14,168,112 (13.51 MiB) |
| 3.42 expert, K4 | 36 | 18,886,704 (18.01 MiB) |
| Per-layer shared_h profile (12 x `F16 [6144]`) | 12 | 147,456 (144 KiB) |
| Replicated H rows to expand one expert to per_expert_v1 shape | +12 | +147,456 |
| Native brandonmusic K3 expert unit (for comparison) | 48 | 14,315,568 |

Note 14,168,112 + 147,456 = 14,315,568 — an expanded 3.42 K3 expert is
byte-for-byte the same *size and tensor-set signature* as a native
`per_expert_v1` K3 expert.

### Class (b) — shared-H segment family: fully salvageable, pure repack-of

The cleanest extraction. Two fragment kinds, both byte-identical slices of the
pinned shards (predicate `repack-of`, attestable against `MANIFEST.sha256`):

1. **Expert fragments**: 19,456 units of 36 tensors each (contiguous range
   read). Total payload: 11,414 x 14,168,112 + 8,042 x 18,886,704 =
   **313,601,703,936 B = 292.06 GiB**.
2. **Per-layer shared profiles**: 76 units of 12 tensors,
   76 x 147,456 = **11,206,656 B = 10.69 MiB** total.

An expert fragment is *decodable only together with its layer's profile
fragment*, so the family manifest must pin (layer -> profile) as a hard
dependency. Consumable by any `shared_h_v1`-aware loader (Gilded Gnosis r28+;
rejected by legacy loaders per the compatibility contract). This family is
internally complete and self-consistent: same calibration corpus, same encode
pass, per-expert K recorded in `tier_bitmap.json` (ship it, 922,664 B, as the
family's bitrate map).

### Class (a) — per-expert family mixable with the brandonmusic base: salvageable via exact expansion (derived-from, not repack-of)

The decode algebra of both layouts is the same diagonal sandwich
`W_slice = diag(v_out) . dequant(trellis, mcg) . diag(v_in)`; `shared_h_v1`
merely stores one `[6144]` H-side row per (layer, proj, rank) instead of 256,
and re-factors the scalar g_scale onto the expert-local side ("the loader
keeps each shared row physically shaped [1, H]; it does not expand it back to
256 rows" — i.e. broadcast is the defined semantics, and the doc's tests
"verify the algebraic equivalence" of the g_scale move). Therefore:

- **Replicating the layer's shared `suh` into each expert's
  `{gate,up}_proj.rank{R}.suh` slot and shared `svh` into
  `down_proj.rank{R}.svh` produces a 48-tensor unit with the exact
  per_expert_v1 name/shape/dtype signature** (verified size match above), and
  a plain per_expert_v1 loader multiplying the stored vectors decodes
  bit-identical weights to the shared_h loader. No re-encode; the trellis,
  mcg, and expert-local vectors stay byte-identical; only 147,456 B/expert of
  replicated rows are added.
- Cost: 19,456 x 147,456 = 2,868,903,936 B (2.67 GiB) of replication; family
  total **316,470,607,872 B = 294.74 GiB**.
- These expanded units are **mechanically mixable with brandonmusic
  per_expert_v1 experts** (every expert decode is self-contained given its own
  rows; shapes and mcg multiplier match; same calibration corpus
  `cf247acc...`). Mixing is a quality decision, not a compatibility one.
- Caveats: (i) the fragments are `derived-from` (exact-expansion), not
  `repack-of` — attestation should record the replication rule and the source
  shared-row digests; (ii) the g_scale factorization differs from a native
  per-expert encode (gate/up scale rides in `svh [512]`, `suh [6144]` is a
  near-unit-norm sign/profile row) — invisible to any loader that treats
  suh/svh as opaque diagonals, but a loader that *re-derives* or asserts
  anything about the vectors would notice; (iii) before campaign extraction,
  run a one-expert numeric check: decode one (layer, expert, proj, rank) slice
  under both loaders and diff (expected exact to F16/BF16 rounding).
- Recommended granularity: whole expert (36->48 tensors, all 3 projections x
  4 ranks). Per-(proj, rank) sub-fragments are also self-contained but keep
  the whole-expert unit for routing semantics and simpler manifests.

### Also salvageable (either family): unquantized BF16 payload

Byte-identical repack-of fragments, independent of the rotation layout:
attention + indexer + `mlp.gate` + `mlp.shared_experts` + norms per MoE shard
(~407.7 MiB/layer, ~30 GiB across 76 layers), dense layers 0–2
(801,799,648 B each), `model-embed` / `model-head` (~1.9 GB each), MTP extras
(`eh_proj` et al.). These come from the same BF16 base (`zai-org/GLM-5.2`) as
the brandonmusic artifact and are layout-neutral.

### Class (c) — not salvageable

- **Re-basing a shared-H trellis onto brandonmusic's own per-expert
  `suh`/`svh` rows (or any different H rotation) without re-encode.** The
  trellis indices were chosen by calibrated LDLQ *with the shared profile
  forced on the H side* (two-pass recipe); under a different diagonal
  rotation the quantization objective changes and there is no closed-form
  remap of Trellis/MCG codes. Only the replication route in class (a) is
  exact; substituting different H rows requires a full re-encode from BF16.
- **The reverse direction — deduplicating per_expert_v1 artifacts into
  shared_h form — is equally impossible losslessly** ("existing EXL3
  checkpoints cannot be deduplicated losslessly because their expert-local
  H-side vectors are not identical", confirmed by our probes: expert rows
  differ).
- Mixing at *sub-expert* granularity across artifacts with different
  calibrations (e.g. rank0 from 3.42, rank1 from 3.0 for the same expert) is
  mechanically loadable but crosses calibration boundaries inside one logical
  weight; exclude it from the fragment vocabulary.

## Extraction recommendation

Extract **class (b)** as the primary family: 19,456 contiguous expert
fragments (292.06 GiB) + 76 shared-profile fragments (10.69 MiB) +
`tier_bitmap.json`, all `repack-of` the pinned revision with per-shard SHA-256
from `MANIFEST.sha256` as attestation materials. Generate **class (a)** lazily
as a *deterministic view* over class (b) — the expansion is a pure function
(replicate 12 rows), so there is no need to store the 294.74 GiB expanded form;
materialize expanded per_expert_v1 units on demand when mixing into the
brandonmusic base, after the one-expert numeric equivalence check. Do not
attempt trellis re-basing (class c) — it is a re-encode, not an extraction.

## Reproduction notes

- Headers: `curl -r 0-7` for the length prefix, then `curl -r 8-(7+len)` for
  the JSON, against
  `https://huggingface.co/willfalco/GLM-5.2-EXL3-TR3-3.42bpw/resolve/ae68c65947efa90bea37308e15421872f124c46d/model-layer-0{03,30,78}.safetensors`.
- Saved intermediates (scratchpad, session-local):
  `hdr-342-L{003,030,078}.json`, `hdr-336-L030.json`, `hdr-bm30-L030.json`,
  `config.json`, `tier_bitmap.json`, `calibration_manifest.json`,
  `MANIFEST.sha256`, `tree.json`.
- Local comparison shard:
  `/home/mbelleau/.cache/huggingface/hub/models--brandonmusic--GLM-5.2-EXL3-TR3-3.0bpw/snapshots/9297b9f1d53af5c67cffa01e30cc071a1ff7144b/model-layer-030.safetensors`.

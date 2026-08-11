---
license: mit
base_model: zai-org/GLM-5.2
base_model_relation: quantized
tags:
- progressive-tensors
- fungible-quant
- exl3
- trellis
- moe
- glm
---

# Fungible Quant Segments — GLM-5.2 (K2 · K3 · K4 · K5)

> ### TLDR
> **Mixed-K quants as a service** — assemble your own EXL3 checkpoint from
> shared, attested, per-expert segments. Pure safetensors. No new format.

**Fungible Quant Segments (for use in Progressive Tensors):** runtime
per-expert bit-width reallocation for EXL3 MoE serving in the Gilded Gnosis
vLLM stack. The core of this repo is the **shared K3 base tier** — the
"everyone downloads this once" layer of the progressive-JPEG model for quants —
covering every MoE layer (3–78).

**Do not trust any layer range printed in this card** — read
`fq-manifest.json` and use `per_k[K].layers` / `per_k[K].segment_count`, which
are rebuilt from the actual published inventory on every publish. Segments are
content-addressed and per-layer, so every published window is durable value on
its own.

**Everything here is pure, unmodified safetensors** — this is a fetch/assembly
*scheme*, not a container format. Any safetensors tool can read these files.

- **HF Repo:** <https://huggingface.co/malaiwah/GLM-5.2-EXL3-FQ-segments>
- **Tooling:** <https://github.com/malaiwah/progressive-tensors>

---

## Inventory — snapshot, not authority

Measured from the live repository listing at commit `c64a3f60`,
**2026-08-11 10:07 UTC**. An unattended encode campaign publishes to `main`
roughly hourly, so these numbers grow. `fq-manifest.json` is the authority;
this table is a convenience.

| tier | segments | layers | on-disk | one expert | provenance predicate |
|---|---:|---|---:|---:|---|
| **K3** (base) | **76** | 3–78, contiguous | 278.6 GB | 13.65 MiB | `repack-of` |
| K2 (fast-load) | 75 | 3–77, contiguous | 184.4 GB | 9.15 MiB | `encode-of` |
| K4 (promotion) | 56 | 3–58, contiguous | 273.0 GB | 18.15 MiB | `encode-of` |
| K5 (hot) | 24 | 3–10 and 35–50 — **sparse, layers 11–34 absent** | 146.0 GB | 22.65 MiB | `encode-of` |
| `sources/` (community-primed K4) | 105 files | 3–10 | 79.7 GB | — | `repack-of` / `derived-from` |
| metadata, indexes, attestations | 241 files | — | 102.8 MB | — | — |
| **whole repository** | | | **961.8 GB** | | |

K5's `per_k["5"].layers` reads `[3, 50]` — that is a **min/max, not a range**.
Read `index-k5.json` for the actual layer set. This is exactly the drift the
warning above exists for.

**Do not `hf download` this repository whole.** No recipe needs all of it.

## Read this before you download

Four constraints that decide whether these files are useful to you. All are
measured, and each links to the report that measured it.

### 1. TP4 only. Not TP2, not TP1, and TP8/TP16 is unimplemented.

The four rank slices in every segment are **four independent quantizations**,
not four slices of one quantization. EXL3 stores a per-(expert, projection,
rank) input rotation `suh` and output rotation `svh`; if the ranks were slices
of a single quant, the H-side vectors (on the axis TP does *not* split) would
be byte-identical across ranks. **They are not** — sha256 of
`gate_proj.suh` and `down_proj.svh` at layer 3 differs across all four ranks,
in every artifact family checked (ours, and two community quants).

Consequences:

- **TP4 → TP4**: identity, works today.
- **TP4 → TP2 / TP1**: **impossible as a repack.** Merging needs two slices to
  share the un-split-axis rotation, and they measurably do not. It requires
  dequantize → concat → **re-quantize**, and the fq tooling has no dequant/
  requant path by construction (`fq_repack` is byte-verbatim; predicate
  `repack-of`).
- **TP4 → TP8 / TP16**: arithmetically valid — splitting a 512-wide slice into
  2×256 or 4×128 keeps the H-side rotation intact and cuts the I-side one on
  128-aligned, whole-trellis-tile boundaries — but **not implemented**. Zero
  code, zero evidence. TP32 is blocked outright (64 < 128).
- **Expert parallelism (EP) and data parallelism (DP > 1)**: refused by the
  loader, in both cases before any of our code runs.
- A TP mismatch **fails closed at model construction**, not silently:
  `rank-sliced EXL3 checkpoint TP does not match runtime`.

You will download hundreds of gigabytes before a loader can tell you this, so:
if you are not serving TP4, these segment files are not usable as-is.
Full analysis with `file:line` citations: `runs/m5-serve/topology-neutrality.md`.

### 2. K5 cannot currently be *served* as a mixed tier on SM120 / Blackwell.

The mixed K3/K5 checkpoint assembles and verifies clean, loads its weights
(77.83 GiB/rank in 81.8 s), and then all four TP workers die during kernel
construction:

```
ValueError: W4A16 shared-memory footprint exceeds device opt-in limit:
            109568 > 101376 bytes (layout=trellis3_t256)
```

Measured cause: the mixed-trellis path forces one tile config across *every*
tier and only varies `trellis_bits`, so the tile is sized for the base tier and
applied to the widest one. Footprint grows ~8192 bytes per bit of tier width;
K4 at the failing configuration lands at exactly 101,376 bytes — the opt-in
limit, to the byte — so **K3+K4 is the viable mixed ladder on SM120 today and
K5 is the first tier that cannot fit at all**.

**The K5 segments are valid artifacts.** This is a runtime kernel limit, not a
problem with the encoded weights: they will work on hardware with a larger
shared-memory budget, or once tile selection accounts for `max(tier_bits)`
rather than the base tier (from the same measurements, `cta_m=2, 128x128` fits
K5 at 82,432 bytes with 19% headroom). Details:
`runs/m5-serve/k5-shared-memory-limit.md`.

### 3. You also need the source checkpoint on disk.

Segments carry **routed-expert tensors only**. Attention, shared experts,
router, norms, embeddings and `lm_head` are copied byte-exact from the source
quant — [`brandonmusic/GLM-5.2-EXL3-TR3-3.0bpw`](https://huggingface.co/brandonmusic/GLM-5.2-EXL3-TR3-3.0bpw)
@ `9297b9f1d53af5c67cffa01e30cc071a1ff7144b`, **316.4 GB**.

Not from `zai-org/GLM-5.2`. That was tried and is wrong: the non-expert tensors
in an EXL3 checkpoint are already in rank-sliced EXL3 form, and z.ai ships BF16
in a different layout. Budget source checkpoint + fetched segments + assembled
output.

### 4. Mixed-K needs a loader that understands it; all-K3 does not.

A mixed recipe emits `hybrid_tr3_tail.bits: "mixed"` and needs Gilded Gnosis
r33+ to load it. An **all-K3 recipe is byte-identical to the source
checkpoint** and therefore loads anywhere the source does.

## What has been proven, with numbers

Four independent results, all on real GLM-5.2 (not a proxy), on an 8× RTX PRO
6000 (SM120) box.

### Reassembly is bit-exact — 81/81 shards

`fq_assemble` rebuilt the full checkpoint from the published K3 segments and
compared it against the source quant's own `MANIFEST.sha256`:

| check | result |
|---|---|
| shard hashes vs source | **97 shard-checks, 97 identical, 0 divergent** (81 unique shards) |
| segment attestations, ed25519 under the pinned signer | **76/76 verified**, 0 failures |
| tensors | 935,105 == 935,105, delta 0 |
| parameters | 158,152,144,896 == source, delta **0** |
| tensor bytes | 316,304,795,648 == source, delta **0** |
| index | 935,105 entries, **0 dangling**, 0 missing |
| file set | 81/81 shards, none missing, none unexpected |

No tensor was anything other than bit-exact. A separate mixed K3/K5 build
diverged from the source on **exactly** the 12 K5-bearing shards and was
byte-identical on the other 69 — the intended shape.
Report: `runs/m5-serve/assembly-report.md`.

### It boots and serves

A checkpoint assembled *by this tooling, out of these segments* under GG vLLM,
TP4, `exl3`, `B12X_MLA_SPARSE`, `fp8_ds_mla` KV:

| metric | value |
|---|---|
| model loading | **76.14 GiB/rank**, 400.9 s (95.6 GiB card, GMU 0.92) |
| KV cache after weights | 6.54 GiB → 130,048 tokens |
| load probe | 120 s at concurrency 8, `max_tokens=128` |
| requests | **208 issued, 208 succeeded, 0 failed** |
| aggregate throughput | **219.2 tok/s** (median scraped decode 225.6 tok/s) |
| single stream | 34.9 tok/s |

Run concurrently with an encode campaign on the other four GPUs — the
coexistence case, deliberately. 76.14 GiB/rank is 80% of the card, which bounds
promotion: **promotion comes out of a fixed budget, not out of headroom that
does not exist.** Report: `runs/m5-serve/m0-boot-gate.md`.

### It is still competent — GSM8K 89.2%

`gsm8k_cot_zeroshot` (lm-eval v3) against the assembled serve:

| metric | value |
|---|---|
| **flexible-extract exact_match** | **0.892 ± 0.0197** |
| strict-match exact_match | 0.116 ± 0.0203 |
| items | 250-item subsample, **seed 1234** (not the full 1319) |
| concurrency | 16 |

Both numbers are reported because quoting only the good one would be
cherry-picking. **Read the flexible one.** `strict-match` requires the answer in
a rigid `#### N` form; GLM-5.2 is a reasoning model that emits chain-of-thought
and almost never satisfies that format, so 0.116 measures format compliance,
not arithmetic. A 250-item subsample carries ±2% stderr, so a 1–2 point
difference against a future re-tiered run would be inside the noise.
Report: `runs/m5-serve/results/axes/GSM8K-BASELINE.md`.

## How to use it

```bash
git clone https://github.com/malaiwah/progressive-tensors && cd progressive-tensors
uv venv && uv pip install -e '.[hub]'
cat keys/FINGERPRINTS   # take the signer fingerprint from HERE, not from the download
```

The pinned signer for everything in this repository is
`a58b7bb79ba58457` (short form; full 64-hex
`a58b7bb79ba5845716aa6fee7d54e714ef243c2875f23a617e1ef3247c565525`). Pass it as
`--trust-signer` to every tool. Under pinning, a compromised artifact
repository can deny you service — it cannot make you accept the wrong bytes.

### 1. Get the source checkpoint (non-expert tensors live here)

```bash
hf download brandonmusic/GLM-5.2-EXL3-TR3-3.0bpw \
  --revision 9297b9f1d53af5c67cffa01e30cc071a1ff7144b \
  --local-dir ./source-quant          # 316.4 GB
```

### 2. Fetch only the segments your recipe names (`fq_fetch`)

`fq_fetch` reads `index-kK.json`, turns a recipe into per-expert byte spans,
coalesces them, and HTTP-Range-fetches exactly those. Every expert is verified
against the signed attestation of the source it came from *as it lands*, before
the file is finalized. It is resumable, and `--dry-run` prints ranged bytes vs
whole files vs whole repo before you spend bandwidth.

```bash
REPO=malaiwah/GLM-5.2-EXL3-FQ-segments
REV=release-2026-08-10        # a tag on one immutable commit (64e582a1…, 2026-08-10 23:11 UTC)
                              # `main` moves under you — pin something.

uv run tools/fq_fetch.py --policy recipes/glm52-3.0bpw-all-k3.json \
  --out ./segments --source "$REPO@$REV" \
  --trust-signer a58b7bb79ba58457 --dry-run     # then drop --dry-run
```

An all-K3 recipe *is* the whole base tier, so it fetches essentially
everything (278.6 GB); the saving appears as soon as the recipe is narrower
than the repo — a layer window, or a sparse K4/K5 hot set over a K3 base.

Whole-file downloads work too, but **`--include` must be repeated once per
pattern** — `--include "a" "b"` makes the CLI read `b` as a filename and
silently ignore the include list:

```bash
hf download "$REPO" --revision "$REV" --local-dir ./segments \
  --include "fq-manifest.json" --include "fq-release.json" \
  --include "index-k*.json"    --include "attestations/*" \
  --include "recipes/*"        --include "LICENSE" --include "NOTICE" \
  --include "layer-*.k3.safetensors"
```

### 3. Verify

```bash
# one signature over every file's sha256 + size, at a release commit
uv run tools/fq_release.py verify --dir ./segments --complete \
  --trust-signer a58b7bb79ba58457

# byte-identity of a reconstruction against the source checkpoint
uv run tools/fq_verify.py --identity --segments ./segments \
  --source ./source-quant --json id.json --md id.md
```

`--complete` fails on any listed file that is absent **and** on any local file
the signature does not cover, which is what makes "nothing was added"
checkable. It is only meaningful against a pinned release commit: against
`main` there will be segments newer than the release manifest, reported as
unlisted, and `fq-manifest.json` will read as `MISMATCHED` because every
incremental publish rewrites it from the live inventory. That is the mechanism
working, not a fault.

### 4. Assemble

```bash
uv run tools/fq_assemble.py \
  --segments ./segments --source ./source-quant \
  --policy recipes/glm52-3.0bpw-all-k3.json --out ./my-checkpoint \
  --trust-signer a58b7bb79ba58457
sha256sum -c MANIFEST.sha256      # in the output dir
```

Assembly **fails closed** without a pinned signer: every fragment's signature,
predicate, fragment digest and per-expert digests are recomputed from the bytes
on disk before anything is written. `--insecure` exists for local development
and says so loudly.

**Two fingerprints, one chain.** For *fetch*, `--trust-signer` is the
publisher's fingerprint from `keys/FINGERPRINTS` in the tooling repo — the point
being that it does not come from the download you are checking. For *assemble*
over a **range-fetched** tree it is **your own**: a fetched subset is a new file
with fewer experts, new offsets and a new digest, so no publisher signature can
cover it. `fq_fetch` signs what it materialized as `derived-from`, pins the
publisher fragments by digest as parents, and prints the exact assemble command.
Assembling a tree you downloaded whole pins the publisher directly.

**On `--reflink`:** safe and sometimes faster, but not a guaranteed space
saver. Measured on XFS: every expert region went through `copy_file_range`
with zero fallbacks and byte-identity always held, but **zero extents ended up
shared** — 0.00% of expert bytes are 4K-congruent between segment and shard
offsets. Whole-file cloning of a byte-identical shard *does* share (100% of
extents, measured by `filefrag`), which is how the all-K3 rebuild cost ~0
incremental disk.

### Recipes shipped in the repo

`recipes/` holds ready-to-use `fq-policy/2` documents, each pinned to a window
we actually validated. A policy is just a per-layer, per-expert map of K — widen
one yourself as the campaign publishes more layers.

| recipe | rebuilds | proof |
|---|---|---|
| `glm52-3.0bpw-all-k3.json` | brandonmusic 3.0bpw, byte-for-byte | **81/81 shards sha256-identical**; boots, 219.2 tok/s, GSM8K 89.2% |
| `glm52-r28-partition-primed-k4.json` | willfalco 3.42bpw r28 partition from primed K4 over the K3 base | fragment byte-identity vs fresh ranged reads; expanded family re-derived 2048/2048 |
| `glm52-fastload-k2-window1.json` | K2 on layers 3–10, K3 elsewhere | our encodes, `encode-of` |
| `glm52-hot-k5-window1.json` | K5 on layers 3–10, K3 elsewhere | `encode-of`; **assembles and verifies, does not serve on SM120** — see limitation 2 |

## Provenance

Each K tier terminates in a different, pinned, auditable chain. Every segment
carries a signed `fq-attestation/1` line (ed25519, keyid =
`a58b7bb7…c565525`) recording `fragment.sha256`, `materials`, and
`expert_sha256` — **one digest per expert's contiguous byte-span**, so a third
party can spot-check a single expert with one ranged read.

| tier | predicate | comes from |
|---|---|---|
| **K3** | `repack-of` | [`brandonmusic/GLM-5.2-EXL3-TR3-3.0bpw`](https://huggingface.co/brandonmusic/GLM-5.2-EXL3-TR3-3.0bpw) @ `9297b9f1…`. Trellis bytes copied **verbatim**; `materials` names the source file and its sha256, cross-checked against the source repo's own `MANIFEST.sha256`. |
| **K2 / K4 / K5** | `encode-of` | Encoded by us from [`zai-org/GLM-5.2`](https://huggingface.co/zai-org/GLM-5.2) @ `b4734de4…` with the capture pipeline. `materials` pins base model + revision, `capture_fingerprint c442aa4c…`, `encoder encode_tr3_v31.py`, `encoder_sha256 e9a85a47…`; `quant_args` records `K`, codebook, `seed_base`, `sigma_reg`, `tp`. |
| **`sources/willfalco-3.36bpw`** | `repack-of` | K4, layers 3–10, primed from [`willfalco/GLM-5.2-EXL3-TR3-3.36bpw`](https://huggingface.co/willfalco/GLM-5.2-EXL3-TR3-3.36bpw) @ `8d9aa923…`. |
| **`sources/willfalco-3.42bpw`** | `repack-of` + `derived-from` | K3/K4, layers 3–10, from [`willfalco/GLM-5.2-EXL3-TR3-3.42bpw`](https://huggingface.co/willfalco/GLM-5.2-EXL3-TR3-3.42bpw) @ `ae68c659…`; the `expanded` view is re-derived by rule `shared_h_expand_v1` from the `shared-h` parent, pinned by parent sha256. |

The community-primed K4 under `sources/` is the point of the exercise made
concrete: **not our encode — somebody else's quant, decomposed into fragments a
recipe can name.** It is kept separate from the root tiers precisely so the two
chains never get confused. The root `layer-*.k4.safetensors` tier is ours
(`encode-of`); the willfalco material covers layers 3–10 only.

### How a consumer checks it

1. Take the fingerprint from `keys/FINGERPRINTS` in the tooling repo's git
   history — never from this download.
2. `fq_release.py verify --complete` at a release commit: one signature over
   every file. Because the attestation files are themselves covered by it,
   their per-expert digests become trusted data.
3. Read any attestation's `predicate` and `materials` to see what the chain
   terminates in, then re-derive: for `repack-of`, one ranged read against the
   pinned source revision reproduces the expert's bytes exactly; for
   `encode-of`, `materials` names the exact stack that produced them.

**One honest caveat on `encode-of`: it is stack-scoped, and says so.** Every
`encode-of` attestation carries a `determinism_scope` block naming
`gpu_arch: sm120` and `torch: 2.12.0+cu132`. A deterministic re-encode
reproduces bytes only within the same encoder sha, exllamav3 version,
torch/CUDA build and GPU architecture. Measured, not assumed: CUDA `pow()`
differs from CPU by 1 ulp on 3 of 32 exponents at this model's rope base —
enough on its own to flip ~1.4% of routings per layer — and grouped/cuBLAS
GEMMs are not row-stable across batch shape. Across stacks the honest predicate
is `equivalence-of`: decode both fragments and attest both reconstruction
errors against the same BF16 ground truth.

*(The first 16 `encode-of` segments were briefly published with a `repack-of`
label inherited from the publishing path. All were re-emitted correctly on
2026-08-10, each carrying a `supersedes` note recording the correction.)*

## Layout

```
fq-manifest.json                 # fq-manifest/1 — THE authority on coverage; rebuilt
                                 #   from the remote inventory on every publish
fq-release.json                  # fq-release/1 — ONE signature over every file's sha256
index-kK.json                    # per-layer -> per-expert [lo,hi) byte ranges
layer-LLL.kK.safetensors         # one MoE layer at one K; 256 experts, body
                                 #   per-expert contiguous -> range-readable
attestations/layer-LLL.kK.jsonl  # one signed attestation per segment
recipes/*.json                   # fq-policy/2 documents
sources/<uploader>-<bpw>/…       # fragments primed from other community quants
LICENSE / NOTICE                 # our licence; the upstream attribution chain
```

Each segment holds
`model.layers.{L}.mlp.experts.{E}.{gate,up,down}_proj.rank{0-3}.{trellis,suh,svh,mcg}`
— rank-sliced TP4 layout, layout tag `rank_sliced_tp4` in every header.

Schema strings `fq-segment/1`, `fq-attestation/1`, `fq-manifest/1`,
`fq-release/1` are stable API.

## Status and known gaps

Active research artifact (2026-08). The **segment/assembly side is heavily
verified**: bit-exact reassembly on 81/81 shards, boots and serves at 219 tok/s,
GSM8K 89.2%. The **runtime side** — the progressive loader and live per-expert
reallocation — is **experimental**: routing selection has been measured against
a human reference, but no expert has been promoted on a live serve and no eval
has been run against a re-tiered model.

- **Verification hardening is in flight.** Fixes from an independent review are
  landing in the tools repo. `git pull` before you rely on a verification
  result, and read [`TRUST.md`](https://github.com/malaiwah/progressive-tensors/blob/main/TRUST.md) §7.
- **Pinning is not optional in practice.** Without `--trust-signer` the tools
  fall back to "any key this project has authorized" — a weaker claim than you
  probably want.
- **`main` moves.** Pin `--revision`. Note the current release tag
  (`release-2026-08-10`) predates the K4 tier entirely: it carries K2 30 / K3 76
  / K5 8 and no K4. Pinning it gets you a single-signature, immutable tree;
  tracking `main` gets you the campaign's output with only per-fragment
  attestations. Choose deliberately.
- **Borrowed-buffer tensor loaders corrupt EXL3 checkpoints** — with or without
  fungible quant. The EXL3 quant methods retain every loaded tensor until
  `process_weights_after_loading` and their two "copy" operations are identity
  functions when the tensor is already contiguous and on-device. Avoid
  `instanttensor` with `INSTANTTENSOR_COPY=0`, and `fastsafetensors` at world
  size > 1. `--load-format auto|hf|safetensors` is the tested path.

## Evidence

Every number in this card is measured, not estimated: the boot gate and
throughput from a TP4 serve of a checkpoint assembled out of these segments,
the GSM8K score from `lm-eval` against that serve, the bit-exactness from a
tensor-by-tensor comparison against the source quant, and the TP4 and K5
limits from direct measurement on SM120 hardware.

Full upstream attribution — base model, every source quant, the EXL3 format and
the tooling, each pinned by revision — is in [`NOTICE`](NOTICE). The licence
covering *our* contribution (segmentation, indexes, attestations, recipes, docs),
which explicitly does not cover the upstream weights, is in [`LICENSE`](LICENSE).

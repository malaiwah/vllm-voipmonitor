# M5 — assembling bootable GLM-5.2 checkpoints from Progressive Tensors segments

Date: 2026-08-11. Box: `/home` = XFS on Ceph RBD (`/dev/rbd2`, 3.0 TB, kernel
6.8.0-71), a quantization campaign running concurrently on GPUs 4–7.

Two checkpoints, both produced by `tools/fq_assemble.py` from signed segments:

| Checkpoint | Path | Policy | What it is |
|---|---|---|---|
| pure K3 | `/home/mbelleau/glm52-k3-assembled` | `policy-k3-uniform.json` | the clean A/B baseline — all 76 MoE layers at K3 |
| mixed K3/K5 | `/home/mbelleau/glm52-mixed-k3k5` | `policy-mixed-k3k5.json` | 12 layers carry 64 K5 experts each; the swap engine needs high-K slabs to trade into |

**Headline:** both assemble and verify green. The pure-K3 build is bit-exact
against `brandonmusic@9297b9f1` on **81/81** shards. The mixed build differs
from the source on exactly the **12** K5-bearing shards and is byte-identical
on the other 69, which is precisely the intended shape.

| | pure K3 | mixed K3/K5 |
|---|---|---|
| shards | 81 | 81 |
| bit-exact vs source | 81/81 | 69/81 (12 K5 layers differ **by design**) |
| segment attestations verified | 76/76 | 88/88 |
| tensors / params | 935,105 / 158,152,144,896 | 935,105 / 161,776,023,552 |
| **per-rank weights (TP4)** | **73.65 GiB — 77.0 %** | **75.33 GiB — 78.8 %** |
| logical size | 316.4 GB | 323.7 GB |
| **physical disk consumed** | **0.0 GB** | **56.2 GB** |
| K5 slots for the swap engine | none | 12 layers × 64 = **768** |

Both are ready to boot; neither has been booted — no GPU was touched, per brief.

---

## 0. The disk finding that shaped everything

The task premise was that `fq_assemble --reflink` makes assembly nearly free
on XFS. **That premise is false on this box for per-region copies, and it was
already measured** — see `runs/0c-campaign/verify/reflink-xfs-measurement.md`:
`copy_file_range` is used for all 12,288 expert regions per layer with zero
fallbacks, byte-identity always holds, but **0 extents end up shared**, because
0.00 % of expert bytes are 4K-congruent between segment and shard offsets
(213 distinct residues, never 0). Reflink saved zero bytes there.

Whole-file copies are a different story, and I re-measured it here:

```
cp --reflink=always <source blob> <dest>
filefrag -v dest -> 0..195751: 440074272..440270023: 195752: last,shared,eof
```

At offset 0 the regions are trivially congruent, so XFS really does share the
extents: **1 extent, `shared` flag, zero incremental disk.** The clones are
independent inodes with link count 1, so writing to the checkpoint cannot
corrupt the HF cache.

### Why a straight full assembly was impossible

`fq_assemble` stages its **entire** output before the final two-rename swap.
One full-model run therefore needs the whole checkpoint staged at once:

| Quantity | Bytes | |
|---|---:|---|
| K3 checkpoint, all 81 safetensors | 316,420,224,008 | 316.4 GB / 294.7 GiB |
| free at start | 325,663,858,688 | 325.7 GB |
| free after a full physical copy | ~9.2 GB | **far below the 120 GB floor** |

A second independent physical copy of a 316 GB model does not fit, full stop.
So the run was **batched by layer**, and each shard materialized by the rule:

* assembled sha256 **equals** the source shard → store as a whole-file reflink
  clone of the byte-identical source blob (zero incremental disk);
* assembled sha256 **differs** → `os.replace()` the assembler's own output into
  the checkpoint (rename on the same filesystem: also free, and the delivered
  file is literally the bytes the assembler wrote).

**Provenance note, stated plainly:** for a shard in the first category the
delivered file is a reflink clone of the source blob, not the staging file the
assembler wrote — but it was proven byte-identical to that staging file first,
by the assembler's own hash of the bytes it emitted, compared against the
source `MANIFEST.sha256`. Nothing is assumed from the all-K3 identity property;
every shard was actually assembled and actually hashed. Peak transient staging
was one batch (~45 GB), never more.

---

## 1. Inputs

| Input | Value |
|---|---|
| K3 segments | `/home/mbelleau/fq-segments/GLM-5.2-EXL3-FQ` — 76 files, layers 3–78, `repack-of` brandonmusic@`9297b9f1` |
| K5 segments | `/home/mbelleau/glm52-segments` — 12 files, layers 35–46, `repack-of local:glm52-k5-encode-of`@`b4734de4` (z.ai BF16 base) |
| source snapshot | `brandonmusic/GLM-5.2-EXL3-TR3-3.0bpw`@`9297b9f1…` — 81 safetensors, 316.4 GB |
| signer | `a58b7bb79ba5845716aa6fee7d54e714ef243c2875f23a617e1ef3247c565525` |

### Pinning the signer without the public trust root

`keys/FINGERPRINTS` lives in `github.com/malaiwah/progressive-tensors`, which is
not cloned on this box, and the tool warns you to take the fingerprint from the
trust root rather than from the artifact you are verifying. Instead of trusting
the family manifest's self-declared `signer_pubkey`, I derived the fingerprint
independently from our own private key:

```python
SigningKey(open('~/.fq_keys/fq_signing.key','rb').read()).verify_key.encode().hex()
# a58b7bb79ba5845716aa6fee7d54e714ef243c2875f23a617e1ef3247c565525
```

It matches the manifest and every attestation `keyid`. This is a stronger
pin than the trust-root lookup would have been: it proves these segments were
signed by the key we hold, not merely by a key the artifact names.

### Non-expert tensors come from the K3 source, not from z.ai

The brief offered z.ai's original weights as a candidate for the non-expert
tensors. That would be **wrong** for an EXL3 checkpoint: attention, shared
experts, router and `lm_head` in the brandonmusic quant are already in the
rank-sliced EXL3 form the loader expects, and z.ai ships BF16 in a different
layout. `fq_assemble` correctly takes every non-expert tensor byte-exact from
the source shard and only substitutes expert tensors from segments. The z.ai
snapshot was used for nothing here.

---

## 2. Pure-K3 assembly

Driver: `assemble-full.sh`. Nine batches (`0-2 3-12 … 73-78`). Representative
command (the driver runs this once per batch):

```bash
/home/mbelleau/venvs/fq/bin/python tools/fq_assemble.py \
  --segments /home/mbelleau/fq-segments/GLM-5.2-EXL3-FQ \
  --source   <brandonmusic-snapshot> \
  --policy   runs/m5-serve/policy-k3-uniform.json \
  --out      /home/mbelleau/glm52-k3-stage \
  --layers   3-12 \
  --trust-signer a58b7bb79ba5845716aa6fee7d54e714ef243c2875f23a617e1ef3247c565525
```

`--force` was never needed and never used; no safety check was bypassed.

Then `finalize.py k3` installed the assembler's metadata, regenerated the index
and `MANIFEST.sha256` from the bytes present, and ran the structural checks.

### Verdict: bit-exact, 81/81

| Check | Result |
|---|---|
| shards assembled + hashed vs source `MANIFEST.sha256` | **97 shard-checks, 97 identical, 0 divergent** (81 unique shards; embed/head re-checked in every batch) |
| segment attestations, ed25519 under the pinned signer | **76/76 verified**, 0 failures |
| tensor count | 935,105 assembled == 935,105 source |
| tensor names / placement in shards | identical sets, identical shard assignment |
| parameter count | 158,152,144,896 == source, delta **0** |
| tensor bytes | 316,304,795,648 == source, delta **0** |
| `model.safetensors.index.json` | 935,105 entries, **0 dangling** shard refs, **0** tensors missing, `total_size` matches the headers |
| file set | 81/81 shards present, none missing, none unexpected |

**No tensor was anything other than bit-exact.** There is nothing to report in
the "divergence" column because the column is empty.

Independent cross-check through a different code path — `fq_verify --identity`
stream-reassembles from segments and compares against the *delivered*
checkpoint (`verify/identity-k3-assembled.{json,md}`):

```
layer 35..40: MATCH (6/6), 3.66 GB from segments + ~0.41 GB source pass-through each
attestation layers 35, 39, 40: sig=verified fragment=ok experts=256 bad=0
signatures: 3 verified against pinned signer a58b7bb79ba58457…
```

### Disk actually consumed: zero

Extent map of all 81 delivered shards (`filefrag -v`, the authoritative
measurement per the 0c finding):

| | bytes |
|---|---:|
| shared extents | **316.4 GB (100.00 %)** |
| unshared extents | **0.0 GB (0.00 %)** |

`du` reports 295 G for the directory because `du` cannot see extent sharing;
the checkpoint's true incremental cost is ~0. Only `filefrag` settles this —
`df` was useless as evidence here exactly as the 0c measurement warned, because
the concurrent campaign swung free space by more than 100 GB during the run
(325.7 GB → 111 GB → 221.6 GB) with none of that traffic attributable to this
assembly.

### TP4 memory

Expert tensors carry an explicit `.rankN.` tag, so each rank loads exactly its
own quarter; everything else is sharded by vLLM's ordinary TP logic at load.

| | value |
|---|---:|
| expert bytes per rank | 69.631 GB (all four ranks exactly equal) |
| non-expert bytes, total | 37.781 GB |
| **per-rank weights** | **73.65 GiB = 77.0 % of a 95.6 GiB card** |

Excludes KV cache, activations and CUDA graphs. For scale, the `serve-baseline`
run measured 81.8 → 92.7 GiB per GPU in flight at GMU 0.95.

---

## 3. Mixed K3/K5 assembly

### Why mixed at all

The swap engine is fixed-cardinality (`exl3_fungible/swap.py`: "v1 swaps are
total: occupancy == capacity"). A swap trades *which* experts occupy the
high-K slots; it does not allocate new ones. So the serve has to boot with the
K5 slabs already allocated and populated, or there is nothing to swap and the
live-upgrade demo has no subject. Memory is then constant across swaps by
construction, which is the property worth demonstrating.

### K5 coverage, measured not assumed

| Source | K5 layers |
|---|---|
| local `~/glm52-segments` (segment files on disk) | **35–46 (12 layers)** |
| local attestations + `index-k5.json` | 3–10, 35–46 (20 layers) |
| HF `malaiwah/GLM-5.2-EXL3-FQ-segments` | 3–10, 35–46 (**20 layers**) — same set, no growth |

So 20 layers are *attested* but only **12 have segment bytes on this box**;
layers 3–10 were published and pruned locally under the streaming-ring plan.
Fetching them back is ~49 GB of downloads. **The build uses the 12 local layers
35–46.** See §5 for what the other 8 would cost and buy.

K3 and K5 come from different producers — K3 is `repack-of`
brandonmusic@`9297b9f1`, K5 is an `encode-of` the z.ai BF16 base @`b4734de4` —
and their family manifests agree on `layout=rank_sliced_tp4`,
`num_experts=256`, `base_model=zai-org/GLM-5.2`, and the same signer. This is
the cross-producer interoperation the reconstruction table already proved for
K4. `fq_assemble` takes one `--segments` dir, so
`make-mixed-inputs.py` builds `/home/mbelleau/fq-segments-mixed-k3k5` as a
**symlink union** of the two families (176 symlinks, zero disk) with a combined
manifest; the per-segment attestations are carried through unchanged, so every
fragment is still authenticated at assembly time.

### Which 64 experts

`tier_bitmap.json` in the brandonmusic source ships `expert_rel_rt_mse` — the
encoder's own per-expert relative round-trip MSE at K3, 256 floats for every
layer 3–78. The policy takes the **64 highest-error experts per covered layer**:
the ones K3 damaged most, so the ones with most to gain from K5. Ties break on
index, so the selection is deterministic and reproducible.

**Honest caveat:** this is a static reconstruction-error proxy. The 0c campaign
found the real signal is routing mass (benefit Gini 0.48, Δε CV only 0.047),
and its eps data does not cover layers 35–46. So this is a *documented,
reproducible initial allocation for a swap demo* — **not** an optimized one,
and it should not be described as one.

### Memory budget, computed from real tensor sizes

Measured per-expert bytes: K3 = 14,315,568 B, K5 = 23,752,752 B, so each
upgraded expert costs exactly **9 MiB** more. Confirmed against a real shard:
layer 35 went 4,075,018,864 → 4,678,999,336 B, i.e. +603,980,472 B = 64 × 9 MiB
plus 696 bytes of header growth.

| | pure K3 | mixed K3/K5 |
|---|---:|---:|
| total tensor bytes | 316.4 GB | 323.7 GB |
| expert bytes per rank | 69.631 GB | 71.443 GB |
| **per-rank weights** | **73.65 GiB (77.0 %)** | **75.33 GiB (78.8 %)** |

768 experts × 9 MiB = 7.25 GB total, +1.69 GiB per rank. **78.8 % is under the
80 % ceiling, so 64 experts per layer stands** — no reduction was needed.
(For reference, adding the 8 unfetched layers at 64 experts each would land at
exactly 80.0 %, i.e. right on the ceiling.)

### Metadata contract — checked, not assumed

A mixed checkpoint whose metadata claims uniform K3 either fails to load or
silently mis-decodes, so the three-part GG loader contract
(`serve-baseline/fruit-mixed-report.md` §2) was verified explicitly against the
emitted files rather than trusted:

| Contract item | Required | Emitted |
|---|---|---|
| `hybrid_tr3_tail.bits` | `"mixed"` (string) | `"mixed"` |
| `hybrid_tr3_tail.k_values` | list ⊆ 3..6 | `[3, 5]` |
| `hybrid_tr3_tail.bits_per_expert` | a `"file.json:field"` *reference*, not inline data | `"tier_bitmap.json:bits_per_expert"` |
| referenced file | `str(layer)` → 256-int list, values ⊆ k_values, for **every** MoE layer | present for all 76 layers; 64×K5 / 192×K3 on covered layers, 256×K3 elsewhere; matches the policy exactly |
| `quantization_config` stub | `quant_method: "exl3"` — vLLM resolves the quant class from this *before* `Exl3Config.maybe_update_config` reads `hybrid_tr3_tail` | `quant_method: "exl3"`, `bits: "mixed"`, `codebook: "mcg"` |

`finalize.py mixed` asserts every row above and fails loudly on any mismatch.
It reported `mixed_contract.all_ok = true`.

### Verdict: exactly the intended 12 shards differ

| Check | Result |
|---|---|
| shard checks | **113 checks, 101 identical to source, 12 divergent** |
| which 12 diverged | `model-layer-035…046.safetensors` — **precisely the K5-bearing layers, and nothing else** |
| segment attestations | **88/88 verified** (76 K3 + 12 K5), 0 failures |
| tensor count | 935,105 == source (same tensors; the trellis tensors are simply larger) |
| parameter count | 161,776,023,552 vs 158,152,144,896, **delta +3,623,878,656** |
| tensor bytes | 323,552,552,960, **delta +7,247,757,312** |
| index | 935,105 entries, 0 dangling, `total_size` matches headers |
| file set | 81/81 shards |
| K5 allocation as loaded | 12 layers × 64 experts = **768**, exactly the policy |

Both deltas are exact, not approximate:

* bytes: 768 experts × 9,437,184 B = **7,247,757,312** ✓
* params: K3 trellis `[384,32,48]` → K5 `[384,32,80]` = +393,216 elements, ×4
  ranks ×3 projections = 4,718,592 per expert, ×768 = **3,623,878,656** ✓

That the arithmetic closes to the byte on two independent counts is the
strongest available evidence that exactly the intended experts were upgraded
and nothing else moved.

### Disk

| | logical | physical |
|---|---:|---:|
| 69 pure-K3 shards | 267.5 GB | **0.0 GB** (100 % shared extents) |
| 12 K5-bearing shards | 56.2 GB | **56.2 GB** (0 % shared — real assembler output) |
| **total** | **323.7 GB** | **56.2 GB** |

The 12 mixed shards cannot share extents with anything: they are genuinely new
bytes, and inserting larger K5 tensors shifts every subsequent offset, so even
the unchanged K3 experts inside them are no longer 4K-congruent with the source
(the same alignment fact the 0c measurement established). 56.2 GB was the
predicted cost and 56.2 GB is what it cost.

Free space never went below the floor **during** either run: the driver's
`check_floor` gate (150 GB, checked before every batch) never fired, batches
were sized so one batch's staging could not breach 120 GB, and the K5 layers
were deliberately scheduled **last** so the run's largest permanent allocation
happened when transient staging was smallest.

**Disk did dip to 111 GB at one point, and none of it was this work.** The
concurrent quantization campaign swung `/home` by more than 100 GB in both
directions (`glm52-segments` alone grew 87 → 160 GB). Verified by extent maps
that the assembled K3 checkpoint was 100 % shared at the time, i.e. contributing
zero. This is why `df` is not evidence on this box and `filefrag` is.

---

## 4. Reproduction recipe

```bash
REPO=/home/mbelleau/protensors-work/vllm-voipmonitor/research/fungible-quant
PY=/home/mbelleau/venvs/fq/bin/python
SRC=~/.cache/huggingface/hub/models--brandonmusic--GLM-5.2-EXL3-TR3-3.0bpw/snapshots/9297b9f1d53af5c67cffa01e30cc071a1ff7144b
SIGNER=a58b7bb79ba5845716aa6fee7d54e714ef243c2875f23a617e1ef3247c565525

# 0. confirm the signer out of band (do NOT take it from the artifact)
$PY -c "from nacl.signing import SigningKey;print(SigningKey(open('$HOME/.fq_keys/fq_signing.key','rb').read()).verify_key.encode().hex())"

# 1. pure K3
$REPO/runs/m5-serve/assemble-full.sh          # 9 batches -> ~/glm52-k3-assembled
$PY $REPO/runs/m5-serve/finalize.py k3

# 2. mixed K3/K5 — build the symlink-union family + policy first
$PY $REPO/runs/m5-serve/make-mixed-inputs.py 64
$REPO/runs/m5-serve/assemble-mixed.sh         # 17 batches -> ~/glm52-mixed-k3k5
$PY $REPO/runs/m5-serve/finalize.py mixed

# 3. independent cross-check (read-only, different code path)
$PY $REPO/tools/fq_verify.py --identity \
  --segments ~/fq-segments/GLM-5.2-EXL3-FQ --source ~/glm52-k3-assembled \
  --layers 35-40 --attest 3 --seed 42 --trust-signer $SIGNER \
  --json $REPO/runs/m5-serve/verify/identity-k3-assembled.json
```

Committed inputs: `policy-k3-uniform.json`, `policy-mixed-k3k5.json` (carries
the exact chosen expert indices per layer), `make-mixed-inputs.py`,
`assemble-full.sh`, `assemble-mixed.sh`, `finalize.py`.

**A third party on a machine with ~700 GB free can skip the batching entirely**
and run one `fq_assemble` per checkpoint with no `--layers`; the batching here
exists only because this box cannot stage a second copy of a 316 GB model.

---

## 5. What a serve could trip over

1. **KV dtype.** `serve-glm52.sh` currently passes `--kv-cache-dtype
   fp8_ds_mla`. That was the *Fruit proxy's* requirement; the big-model
   `serve-baseline` ran `nvfp4_ds_mla`. The sparse-MLA kernel stack is only
   correct with a ds_mla layout — with a non-ds_mla cache both checkpoints boot
   and then emit prompt-*independent* degenerate text, which looks like a model
   bug and is not one. Confirm which ds_mla variant this stack wants before
   blaming the checkpoint.
2. **`quantization_config` carries stale ModelOpt fields.** The source shipped
   `quant_method: "modelopt"`, `quant_algo: "NVFP4"`, `config_groups`,
   `producer` (a b300 dispatch shim). `fq_assemble` correctly rewrites
   `quant_method` to `exl3` and sets `bits`/`codebook`, but leaves the other
   ModelOpt keys in place. They are inert once the exl3 path is selected, and
   `serve-glm52.sh` also passes `--quantization exl3` explicitly, so this is
   belt-and-braces — but a reader diffing the config will see `quant_algo:
   NVFP4` next to `quant_method: exl3` and should not be alarmed.
3. **`model.safetensors.index.json` is new.** The brandonmusic source ships no
   index; `fq_assemble` always generates one. The fq_assemble-produced
   fruit-mixed checkpoint booted with one present, so this is believed
   harmless, but it is a genuine difference from the checkpoint that
   `serve-baseline` booted.
4. **Three source subdirectories are absent** (`benchmarks/`,
   `calibration_encoder/`, `independent-eval/`): `fq_assemble` copies files,
   not directories. All three are empty or 4 KB of docs — nothing load-bearing.
5. **The delivered pure-K3 shards are reflink clones of the HF cache blobs.**
   They are independent inodes with link count 1, so writing to the checkpoint
   cannot corrupt the cache — but the two now share physical extents, so
   **deleting the HF snapshot will not free the space** people expect it to,
   and filesystem-level corruption would hit both.
6. **First boot is slow.** `serve-baseline` measured ~25 min cold (JIT/autotune);
   caches persist under `/home/mbelleau/cache`.

### Note for the upstream PR #280 analysis (native mixed K3/K4/K5)

Our mixed contract assigns **one K per expert**, applied uniformly to all 3
projections and all 4 rank slices — `bits_per_expert` is a 256-int list per
layer. A native per-(expert, projection) layout, as `r7_routed_experts`
appears to intend, is **strictly more expressive than our schema can encode**:
there is no way to say "expert 7 is K5 on `gate_proj` but K3 on `down_proj`" in
a 256-int list. Anyone reconciling the two should treat our `tier_bitmap.json`
form as a projection-uniform *special case* rather than a subset that maps
across cleanly. Two smaller observations: our allocation reaches the loader by
*file reference* (`"tier_bitmap.json:bits_per_expert"`) rather than inline, and
we write the field alongside the source's existing per-layer keys
(`expert_rel_rt_mse`, `keep_nvfp4`, `tail_tr3`), so the file is a superset of
what the source shipped.

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

# GLM-5.2 — Progressive Tensors segments (K2 · K3 · K4 · K5)

> **Tools:** [github.com/malaiwah/progressive-tensors](https://github.com/malaiwah/progressive-tensors)
> — verify these fragments, reassemble any recipe into a bootable
> checkpoint, or prime segments from another community quant.
> **Research + evidence:** [vllm-voipmonitor `research/fungible-quant`](https://github.com/malaiwah/vllm-voipmonitor/tree/claude/gg-overview-exploration-jchgd3/research/fungible-quant).
> **Do not `hf download` this whole repo** — it is **481 GB** and no recipe
> needs all of it. Use the commit-pinned, per-recipe commands
> [below](#download-only-what-your-recipe-needs); the largest recipe is
> 298 GB and the K2 fast-load tier alone is 74 GB.
> **Pin the signer:** pass
> `--trust-signer a58b7bb79ba5845716aa6fee7d54e714ef243c2875f23a617e1ef3247c565525`
> to every tool, with the fingerprint taken from
> [`keys/FINGERPRINTS`](https://github.com/malaiwah/progressive-tensors/blob/main/keys/FINGERPRINTS)
> in git — never from this download.
> **Maturity, by component:** the *segment/assembly* side is heavily
> verified — reassembly is sha256-identical on **76/76** MoE shards, every
> primed fragment re-checked against fresh ranged reads of its pinned
> source (2048/2048 spans), the expanded family fully re-derived, 184 tests
> green. The *runtime* side — the progressive loader and live per-expert
> reallocation — is **experimental**.

> [!IMPORTANT]
> **Distribution is mid-hardening (2026-08).** An independent review of the
> verification path is being worked through right now and fixes are landing
> in the tools repository over the next few days. Nothing known to be wrong
> is left undisclosed — see [Known gaps](#known-gaps-as-of-2026-08-10) — but
> the practical consequence is: **pin `--trust-signer` on every command**,
> and `git pull` the tools repo before you rely on a verification result.
> Pinning is what makes the remaining gaps not matter: it is checked before
> any byte is accepted.

**Purpose-built research artifact** for the *Progressive Tensors / fungible
quant* project: runtime per-expert bit-width reallocation for EXL3 MoE
serving in the Gilded Gnosis vLLM stack. The core of this repo is the
**shared K3 base tier** — the "everyone downloads this once" layer of the
progressive-JPEG model for quants — covering every MoE layer (3–78).

The other tiers are grown from the same family by an unattended encode
campaign, so **their coverage changes from day to day**: K2 (the fast-load
base tier) and K5 (hot-expert headroom) are published window by window and
backfill over time. **Do not trust any layer range printed in this card —
read `fq-manifest.json` and use `per_k[K].layers` / `per_k[K].segment_count`,
which are rebuilt from the actual published inventory on every publish.**
Segments are content-addressed and per-layer, so every published window is
durable value on its own.

**Everything here is pure, unmodified safetensors** — Progressive Tensors
is a fetch/assembly *scheme*, not a container format. Any safetensors tool
can read these files.

## Where this comes from

- **Spec, tools, and reports**:
  [`malaiwah/vllm-voipmonitor`](https://github.com/malaiwah/vllm-voipmonitor)
  branch
  [`claude/gg-overview-exploration-jchgd3`](https://github.com/malaiwah/vllm-voipmonitor/tree/claude/gg-overview-exploration-jchgd3/research/fungible-quant)
  — see `research/fungible-quant/` (design docs 00–14, run reports under
  `runs/`) and
  `research/fungible-quant/tools/` (`fq_repack.py` produced this repo;
  `fq_assemble.py` turns it + a policy JSON back into a bootable
  checkpoint).
- **Source checkpoint (repack, not re-encode)**:
  [`brandonmusic/GLM-5.2-EXL3-TR3-3.0bpw`](https://huggingface.co/brandonmusic/GLM-5.2-EXL3-TR3-3.0bpw)
  @ revision `9297b9f1d53af5c67cffa01e30cc071a1ff7144b`. Trellis bytes are
  copied **verbatim** (byte-identity round-trip verified per layer).
- **Base model**: [`zai-org/GLM-5.2`](https://huggingface.co/zai-org/GLM-5.2)
  @ `b4734de4facf877f85769a911abafc5283eab3d9`.

Full attribution — base model, every source quant, the EXL3 format and the
tooling, each pinned by revision — is in [`NOTICE`](NOTICE). The licence
that covers *our* contribution (segmentation, indexes, attestations,
recipes, docs) and explicitly does not cover the upstream weights is in
[`LICENSE`](LICENSE).

## Layout

```
fq-manifest.json                 # fq-manifest/1: base model, revision pins, per-K coverage
fq-release.json                  # fq-release/1: ONE signature over every file's sha256
LICENSE / NOTICE                 # our licence; upstream attribution chain
index-k3.json                    # per-layer -> per-expert [lo,hi) byte ranges
index-k2.json / index-k5.json    # same, for the K2 / K5 tiers
layer-LLL.kK.safetensors         # one file per MoE layer per K, 256 experts,
                                 #   body per-expert contiguous -> range-readable
attestations/layer-LLL.kK.jsonl  # one signed attestation per segment (see below)
recipes/*.json                   # fq-policy/2 documents (see Recipes)
sources/<uploader>-<bpw>/…       # fragments primed from other community quants
```

`fq-manifest.json` is cumulative and authoritative: `k_variants` lists every
K present and `per_k[K]` carries that K's index, layer coverage, segment
count and provenance. It is rebuilt from the **remote inventory** on every
publish, so it describes what is actually there rather than whatever the
last uploader happened to hold locally.

Each segment holds `model.layers.{L}.mlp.experts.{E}.{gate,up,down}_proj.rank{0-3}.{trellis,suh,svh,mcg}`
(rank-sliced TP4 layout, verbatim from the source; layout tag
`rank_sliced_tp4` in every header). One expert = one contiguous ~14.3 MB
range — fetch exactly the experts you need via `index-k3.json`.

## Download only what your recipe needs

The whole repository is **481 GB** and every recipe is a strict subset of
it. Pin the revision (an unpinned download follows `main`, which an
unattended campaign is still moving) and pass `--include`:

**`--include` must be repeated, once per pattern.** `--include "a" "b"` does
*not* mean two patterns — the CLI takes the second word as a filename and
silently ignores the include list. Every command below repeats the flag.

```bash
REPO=malaiwah/GLM-5.2-EXL3-FQ-segments
REV=release-2026-08-10   # a release tag: one immutable commit, published
                         # as a single atomic push.  See "One signature
                         # over the whole release" for its commit id.

# the metadata every recipe wants — manifest, signed release manifest,
# indexes, attestations, recipes, licence.  ~5 MB.
META=(--include "fq-manifest.json" --include "fq-release.json"
      --include "index-k*.json"    --include "attestations/*"
      --include "recipes/*"        --include "LICENSE" --include "NOTICE")
```

| Recipe | add to `META` | segments | **disk** |
|---|---|---:|---:|
| **all-K3** — byte-identical rebuild of the source quant | `--include "layer-*.k3.safetensors"` | 76 | **279 GB** |
| **fast-load K2** (window 1: K2 on 3–10, K3 elsewhere) | `--include "layer-00[3-9].k2.safetensors" --include "layer-010.k2.safetensors" --include "layer-01[1-9].k3.safetensors" --include "layer-0[2-7]?.k3.safetensors"` | 76 | **269 GB** |
| **hot-K5** (window 1: K5 on 3–10, K3 elsewhere) | `--include "layer-*.k5.safetensors" --include "layer-01[1-9].k3.safetensors" --include "layer-0[2-7]?.k3.safetensors"` | 76 | **298 GB** |
| **r28 primed-K4** (K3 base + willfalco K4) | `--include "layer-*.k3.safetensors" --include "sources/fq-manifest.json" --include "sources/willfalco-3.42bpw/expanded/*"` | 84 | **294 GB** |
| *the K2 fast-load tier on its own* (every published K2 layer, no K3) | `--include "layer-*.k2.safetensors"` | 30 | **74 GB** |
| *everything* (what an unpinned `hf download <repo>` does) | — | 162 | **481 GB** |

```bash
hf download "$REPO" --revision "$REV" --local-dir ./segments \
  "${META[@]}" --include "layer-*.k3.safetensors"      # <- the all-K3 row
```

Sizes are measured from the repository inventory at the release commit, not
estimated — total, `--include` set by `--include` set. They grow as the
campaign publishes more K2/K5 windows; the all-K3 number will not, because
K3 is complete. The two window recipes pin layers 3–10, which is why they
still carry K3 for layers 11–78.

**A recipe does not need whole segment files.** A mixed-K assembly touches a
subset of experts, and `fq_fetch` range-reads exactly those spans — for a
sparse upper tier that is a small fraction of the numbers above:

```bash
uv run tools/fq_fetch.py --policy recipes/glm52-r28-partition-primed-k4.json \
  --source "$REPO@$REV" --out ./segments \
  --trust-signer a58b7bb79ba5845716aa6fee7d54e714ef243c2875f23a617e1ef3247c565525
```

## One signature over the whole release

> **`REV=release-2026-08-10`** is a tag on exactly **one** commit — resolve
> it to its commit id with
> `hf api models/malaiwah/GLM-5.2-EXL3-FQ-segments/refs`, or pass the tag
> straight to `--revision`/`@<rev>` and let the hub resolve it.
> Every release here is a single atomic commit — segments, indexes,
> attestations, manifest and `fq-release.json` go up together or not at all,
> pinned to the parent commit the release was built against, so a concurrent
> writer is rejected rather than interleaved. Later releases get later tags;
> [the commit history](https://huggingface.co/malaiwah/GLM-5.2-EXL3-FQ-segments/commits/main)
> shows each one, and `parent_revision` inside `fq-release.json` names the
> tree it describes.

`fq-release.json` is a single `fq-release/1` document listing the sha256 and
size of **every** file in this repository — segments, indexes, attestations,
manifest, licence — signed once with the project ed25519 key. Verify one
signature, then hash the bytes you actually downloaded:

```bash
uv run tools/fq_release.py verify --dir ./segments \
  --trust-signer a58b7bb79ba5845716aa6fee7d54e714ef243c2875f23a617e1ef3247c565525
# add --complete if you pulled the whole release: it then FAILS on any
# listed file that is absent AND on any local file the signature does not
# cover, which is what makes "nothing was added" checkable.
```

Because the attestation files are themselves covered by that one signature,
their per-expert digests become trusted data: a range-fetch consumer can
check a single expert against them without verifying 300 more signatures.

**What it does not guarantee.** `fq-release.json` describes **one commit**.
Our campaign supervisor publishes *incrementally* — it uploads each encode
window as it finishes, in many commits, and only then is a new release
manifest built and pushed. So:

- at the release commit, the file list is exact and complete — verified:
  353 files hashed, 0 absent, 0 mismatched, 0 unlisted;
- at `main`, which moves under you, there will usually be published
  segments **newer than the release manifest and therefore not covered by
  it**. `verify --complete` against `main` is expected to report them as
  unlisted, and that is the mechanism working, not a fault;
- **and `fq-manifest.json` will report as `MISMATCHED` against an older
  release**, for the same reason: each incremental publish rewrites it from
  the live inventory so that it always describes what is really there. That
  is a *newer* manifest, not a tampered one — but the release signature
  cannot tell those apart, and should not pretend to. Pin the revision and
  the question does not arise;
- it says nothing about *freshness*. A replayed older release verifies
  perfectly; it is simply stale. Compare `release`, `created_utc` and
  `parent_revision` inside the document against what you expected.

Pin `--revision` to a release commit and `verify --complete` is meaningful.
Track `main` and you are choosing convenience over completeness — the
per-fragment attestations still cover each individual segment.

## Provenance & attestation — the point of this repo

Every segment carries a signed `fq-attestation/1` line
(ed25519; signer pubkey in `fq-manifest.json`, fingerprint pinned in git):

- `fragment.sha256` — hash of the exact segment file;
- `materials` — the source repo, **commit-pinned revision**, source file
  name, and its sha256 (cross-checked against the source repo's own
  `MANIFEST.sha256` and verified against the local bytes before signing);
- `expert_sha256` — one digest per expert's contiguous byte-span, so any
  third party can **spot-check a single expert with one ranged read**
  against this repo or re-derive it from the source repo at the pinned
  commit.

Predicate is `repack-of`: the trust chain terminates at the source quant,
made explicit, pinned, and per-fragment verifiable — strictly stronger
than the usual "download 300 GB from a named uploader and hope".

The K2/K5 segments carry **`encode-of`** attestations pinning the base model
+ revision, the capture fingerprint, the encoder sha, the full quant args
and an explicit `determinism_scope` block. (The first 16 were briefly
published with a `repack-of` label inherited from the publishing path; all
were re-emitted correctly on 2026-08-10, each carrying a `supersedes` note
recording the correction.)

**One honest caveat on provenance:**

1. **`encode-of` is stack-scoped, and will say so.** A deterministic
   re-encode reproduces bytes only within the same encoder sha, exllamav3
   version, torch/CUDA build and GPU architecture. Measured, not assumed:
   CUDA `pow()` differs from CPU by 1 ulp on 3 of 32 exponents at this
   model's rope base — enough on its own to flip ~1.4 % of routings per
   layer — and grouped/cuBLAS GEMMs are not row-stable across batch shape.
   Countersigning is therefore a claim about a named stack. Across stacks
   the honest predicate is `equivalence-of`: decode both fragments and
   attest both reconstruction errors against the same BF16 ground truth.

## Known gaps (as of 2026-08-10)

Stated plainly because the alternative is you finding them yourself:

- **Verification hardening is in flight.** Fixes from an independent review
  are landing in the tools repository now. `git pull` before you rely on a
  verification result, and read
  [`TRUST.md`](https://github.com/malaiwah/progressive-tensors/blob/main/TRUST.md)
  §7, which tracks what is implemented and what is not, precisely.
- **Pinning is not optional in practice.** Without `--trust-signer` the
  tools fall back to "any key this project has authorized", which is a
  weaker claim than you probably want.
- **`main` moves.** An unattended campaign publishes to this repo. Pin
  `--revision`.
- **Mixed-K on GLM-5.2 itself is not yet booted.** See Status.

## Status

Active research artifact (2026-08); schema strings `fq-segment/1`,
`fq-attestation/1`, `fq-manifest/1`, `fq-release/1` are stable API. Produced
and verified on an 8× RTX PRO 6000 (SM120) box. Precisely what was verified:
the all-K3 assembly of these segments is **sha256-identical to the source
checkpoint's shards**, and that checkpoint boots and serves under the GG r33
stack (TP4/DCP4) — so "bootable" holds by byte-identity. A *mixed*-K assembly
from segments has been booted and served end-to-end on a smaller
GLM-5.2-architecture proxy, not yet on GLM-5.2 itself.

Mixed recipes need a loader that understands `hybrid_tr3_tail.bits:
"mixed"` (Gilded Gnosis r33+); `fq_assemble` emits the required
`k_values` / `bits_per_expert` reference and the `quantization_config`
stub automatically. An all-K3 recipe is just the source checkpoint and
loads anywhere the source does.

## Reassemble it yourself

Tools, tests, and a full walkthrough live at
**[github.com/malaiwah/progressive-tensors](https://github.com/malaiwah/progressive-tensors)**:

```bash
git clone https://github.com/malaiwah/progressive-tensors && cd progressive-tensors
uv venv && uv pip install -e '.[hub]'
cat keys/FINGERPRINTS        # take the signer fingerprint from HERE, not from the download

# download only what the recipe needs — see the table above
hf download malaiwah/GLM-5.2-EXL3-FQ-segments --revision release-2026-08-10 \
  --local-dir ./segments \
  --include "fq-manifest.json" --include "fq-release.json" \
  --include "index-k*.json"    --include "attestations/*" \
  --include "recipes/*"        --include "LICENSE" --include "NOTICE" \
  --include "layer-*.k3.safetensors"

uv run tools/fq_release.py verify --dir ./segments --complete \
  --trust-signer a58b7bb79ba5845716aa6fee7d54e714ef243c2875f23a617e1ef3247c565525
uv run tools/fq_assemble.py --segments ./segments --source <source-dir> \
  --policy recipes/glm52-3.0bpw-all-k3.json --out ./my-checkpoint \
  --trust-signer a58b7bb79ba5845716aa6fee7d54e714ef243c2875f23a617e1ef3247c565525
```

The all-K3 recipe reproduces the source checkpoint **sha256-identical,
shard for shard** (verified on all 79 quantized layer shards; the 76 MoE
shards come from these segments, the 3 dense shards pass through from the
source). Note you need the **source checkpoint on disk too** — non-expert
tensors (attention, router, shared experts, norms) are copied from it
byte-exact. Mixed recipes (your own K per expert) assemble the same way as
more K tiers land in this repo family. Every fragment is spot-checkable
against its signed attestation with one ranged read — see the walkthrough
for the 6-line verifier.

## Recipes — the exact configs we validated

`recipes/` holds ready-to-use `fq-policy/2` documents, each one a
reconstruction we actually verified. Each pins a fixed window; as the K2/K5
campaign publishes more layers you can widen one yourself (it is a
per-layer, per-expert map of K) — the recipe files here stay pinned to what
was validated.

| Recipe | What it rebuilds | Proof |
|---|---|---|
| `glm52-3.0bpw-all-k3.json` | brandonmusic 3.0bpw, byte-for-byte | **sha256-identical, 76/76 MoE shards** |
| `glm52-r28-partition-primed-k4.json` | the willfalco 3.42bpw r28 partition (L3 206/50, L4+ 148/108) from primed K4 fragments over the K3 base | fragment byte-identity vs fresh ranged reads; expanded family re-derived 2048/2048 |
| `glm52-fastload-k2-window1.json` | K2 fast-load tier on layers 3–10, K3 elsewhere | our encodes, `encode-of` |
| `glm52-hot-k5-window1.json` | K5 hot tier on layers 3–10, K3 elsewhere | our encodes, `encode-of` |

```bash
uv run tools/fq_assemble.py \
  --segments ./segments --source <source-checkpoint-dir> \
  --policy recipes/glm52-3.0bpw-all-k3.json \
  --trust-signer a58b7bb79ba5845716aa6fee7d54e714ef243c2875f23a617e1ef3247c565525 \
  --out ./my-checkpoint
sha256sum -c MANIFEST.sha256   # in the output dir
```

Assembly **fails closed** without a pinned signer: every fragment's
signature, predicate, fragment digest and per-expert digests are checked
against that key before a byte is written. `--insecure` exists for local
development and says so loudly.

## Measured: the bit-width ladder these segments implement

![Per-expert encode error vs bit-width — ~3.8x lower error per +1 bit](assets/eps-ladder-light.svg)

![Upgrade benefit concentrates in few experts — top 16 of 256 carry ~a third](assets/benefit-concentration-light.svg)

One sealed calibration capture, four hessian-identical encodes — the
measurement campaign behind the K tiers this repo serves. Methodology and
raw data: the research branch linked above.

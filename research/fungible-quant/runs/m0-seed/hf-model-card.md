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
> **Maturity, by component:** the *segment/assembly* side is heavily
> verified — reassembly is sha256-identical on **76/76** MoE shards, every
> primed fragment re-checked against fresh ranged reads of its pinned
> source (2048/2048 spans), the expanded family fully re-derived, 121 tests
> green. Mandatory signer-pinned verification inside assembly is landing
> now (until it does, pass `--trust-signer` yourself). The *runtime*
> side — the progressive loader and live per-expert reallocation — is
> **experimental**.

**Purpose-built research artifact** for the *Progressive Tensors / fungible
quant* project: runtime per-expert bit-width reallocation for EXL3 MoE
serving in the Gilded Gnosis vLLM stack. The core of this repo is the
**shared K3 base tier** — the "everyone downloads this once" layer of the
progressive-JPEG model for quants — covering every MoE layer (3–78).

It has since started growing the other tiers from the same family: **K2 and
K5 segments for layers 3–10** (the first window of the multi-K encode
campaign) are also here. K2 is the fast-load base tier; K5 is hot-expert
headroom. Coverage is deliberately partial and backfills over time —
segments are content-addressed and per-layer, so every published window is
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

## Layout

```
fq-manifest.json                 # fq-manifest/1: base model, revision pins, layout
index-k3.json                    # per-layer -> per-expert [lo,hi) byte ranges  (layers 3..78)
index-k2.json / index-k5.json    # same, for the K2 / K5 window               (layers 3..10)
layer-LLL.k3.safetensors         # one file per MoE layer (3..78), 256 experts,
                                 #   body per-expert contiguous -> range-readable
layer-LLL.k2.safetensors         # K2 / K5 window, layers 3..10
layer-LLL.k5.safetensors
attestations/layer-LLL.kK.jsonl  # one signed attestation per segment (see below)
```

`fq-manifest.json` is cumulative: `k_variants` lists every K present and
`per_k[K]` carries that K's index, layer coverage, segment count and
provenance. (An earlier last-writer-wins bug that described only the most
recent K was fixed 2026-08-10 — tool and published manifest both corrected.)

Each segment holds `model.layers.{L}.mlp.experts.{E}.{gate,up,down}_proj.rank{0-3}.{trellis,suh,svh,mcg}`
(rank-sliced TP4 layout, verbatim from the source; layout tag
`rank_sliced_tp4` in every header). One expert = one contiguous ~14.3 MB
range — fetch exactly the experts you need via `index-k3.json`.

## Provenance & attestation — the point of this repo

Every segment carries a signed `fq-attestation/1` line
(ed25519; signer pubkey in `fq-manifest.json`):

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

The K2/K5 window segments carry **`encode-of`** attestations pinning the
base model + revision, the capture fingerprint, the encoder sha, the full
quant args and an explicit `determinism_scope` block. (They were briefly
published with a `repack-of` label inherited from the publishing path; all
16 were re-emitted correctly on 2026-08-10, each carrying a `supersedes`
note recording the correction.)

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

## Status

Active research artifact (2026-08); schema strings `fq-segment/1`,
`fq-attestation/1`, `fq-manifest/1` are stable API. Produced and verified
on an 8× RTX PRO 6000 (SM120) box. Precisely what was verified: the all-K3
assembly of these segments is **sha256-identical to the source checkpoint's
shards**, and that checkpoint boots and serves under the GG r33 stack
(TP4/DCP4) — so "bootable" holds by byte-identity. A *mixed*-K assembly
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
uv venv && uv pip install pynacl numpy huggingface_hub
hf download malaiwah/GLM-5.2-EXL3-FQ-segments --local-dir ./segments
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
reconstruction we actually verified:

| Recipe | What it rebuilds | Proof |
|---|---|---|
| `glm52-3.0bpw-all-k3.json` | brandonmusic 3.0bpw, byte-for-byte | **sha256-identical, 76/76 MoE shards** |
| `glm52-r28-partition-primed-k4.json` | the willfalco 3.42bpw r28 partition (L3 206/50, L4+ 148/108) from primed K4 fragments over the K3 base | fragment byte-identity vs fresh ranged reads; expanded family re-derived 2048/2048 |
| `glm52-fastload-k2-window1.json` | K2 fast-load tier where published (layers 3-10), K3 elsewhere | our encodes, `encode-of` |
| `glm52-hot-k5-window1.json` | K5 hot tier where published (layers 3-10), K3 elsewhere | our encodes, `encode-of` |

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

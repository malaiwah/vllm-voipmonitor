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

# GLM-5.2 — Progressive Tensors segments (K3 base)

**Purpose-built research artifact** for the *Progressive Tensors / fungible
quant* project: runtime per-expert bit-width reallocation for EXL3 MoE
serving in the Gilded Gnosis vLLM stack. This repo is the **shared K3 base
tier** — the "everyone downloads this once" layer of the progressive-JPEG
model for quants. Per-expert K4+ enhancement fragments are encoded lazily
by deployments and published separately.

**Everything here is pure, unmodified safetensors** — Progressive Tensors
is a fetch/assembly *scheme*, not a container format. Any safetensors tool
can read these files.

## Where this comes from

- **Spec, tools, and reports**:
  [`malaiwah/vllm-voipmonitor`](https://github.com/malaiwah/vllm-voipmonitor)
  branch
  [`claude/gg-overview-exploration-jchgd3`](https://github.com/malaiwah/vllm-voipmonitor/tree/claude/gg-overview-exploration-jchgd3/research/fungible-quant)
  — see `research/fungible-quant/` (design docs 00–13) and
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
index-k3.json                    # per-layer -> per-expert [lo,hi) byte ranges
layer-LLL.k3.safetensors         # one file per MoE layer (3..78), 256 experts,
                                 #   body per-expert contiguous -> range-readable
attestations/layer-LLL.k3.jsonl  # one signed attestation per segment (see below)
```

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
than the usual "download 300 GB from a named uploader and hope". Freshly
encoded fragments (the K4 overlay path) will carry the reproducible
`encode-of` predicate: deterministic re-encode from pinned BF16 + Hessian
statistics + encoder version, enabling independent countersigning.

## Status

Active research artifact (2026-08); schema strings `fq-segment/1`,
`fq-attestation/1`, `fq-manifest/1` are stable API. Produced and verified
on an 8× RTX PRO 6000 (SM120) box; the all-K3 assembly of these segments
was verified byte-identical to the source checkpoint and served under the
GG r33 stack (TP4/DCP4).

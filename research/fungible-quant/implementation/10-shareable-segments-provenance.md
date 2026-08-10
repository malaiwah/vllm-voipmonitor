# 10 — Shareable encode segments on HF + provenance

## 1. Storage format (T2 remote K-cache)

**Per-layer, per-K safetensors segment files** — not per-expert files
(19,712 tiny files strains repo ops), not monolithic checkpoints:

```
<model>-EXL3-FQ-segments/
  fq-manifest.json                      # pins base repo + revision (commit hash)
  layer-030.k3.safetensors              # 256 experts × {gate,up,down} + suh/svh/
  layer-030.k4.safetensors              #   rotations + per-tensor error stats
  attestations/layer-030.k3.jsonl       # one DSSE line per fragment (see §3)
```

- ~3.6 GB per K3 layer file: chunky enough for Xet, small enough to fetch
  whole; **range-readable per expert** via `data_offsets` exactly as proven
  in 09 (unsharded; slice per rank at 16-column tile granularity).
- `__metadata__` header carries: manifest hash, hessian_id, encoder version
  + config, per-tensor mse/KLD (**this is the late-bound ε riding along**).
- Upload granularity = one layer file whenever N new fragments accumulate;
  Xet dedupes re-uploads of unchanged regions.

## 2. The progressive-JPEG community model

- **Base layer = flat K3 everywhere** (today's intrinsics floor; K2 stays a
  kernel project — when it lands, it becomes the faster base; the RRQ
  residual-plane roadmap item makes this literally progressive: base +
  enhancement planes, downgrade = drop a plane).
- **Enhancement = per-expert K4+ fragments**, encoded lazily by whoever's
  workload promotes them, published back.
- **Fragment source chains**: `fq-manifest.json` lists `sources:
  [repo, ...]` (default: the repo the base came from). Resolution order:
  local NVMe → sources in order → local encode from BF16 (T3). Like apt
  mirrors; the flag from the user's original proposal, now first-class.
- First user of a new model: stream/encode flat K3 (~41 GPU-h, or spread
  lazily), publish. Everyone after boots primed and only ever fetches or
  encodes the fragments their own promotions need.

## 3. Provenance: attestation + reproducibility, not trust-me

Threat model: fragments are pure data (safetensors, no code execution) —
the risk is **behavioral poisoning** (a tampered expert implanting a
backdoor), not RCE. Probes don't catch designed-to-pass backdoors, so trust
must come from provenance, and it can:

1. **Content addressing**: every fragment named by sha256 of its bytes.
2. **Signed attestation per fragment** (DSSE/in-toto envelope, one JSONL
   line): `{fragment_sha, materials: {bf16_tensor_sha, base_repo,
   revision(commit), tensor_name}, hessian_id, encoder_version, config,
   error_stats, signer}` — sign with minisign or Sigstore. Prior art to
   reuse rather than invent: **OpenSSF `model-signing`** (Sigstore for
   models) and in-toto predicates.
3. **The killer property — deterministic re-encode**: given (BF16 tensor,
   hessian_id's statistic, encoder version, config), the trellis encode is
   bit-reproducible. So any third party holding the BF16 can re-encode any
   fragment and countersign — the **reproducible-builds model for quants**.
   Trust accumulates via independent rebuilders instead of resting on one
   uploader. (A cryptographic *proof* without re-execution would need
   zkML-grade verifiable computation over Viterbi — orders of magnitude
   impractical today; not proposed.)
4. Note the baseline being improved: today the community loads entire
   multi-hundred-GB quants from named uploaders with **zero** verification.
   Signed, content-addressed, independently re-encodable fragments are
   strictly stronger at every step.

## 4. Load-time gates (belt and braces, regardless of signature)

- Manifest revision pin: fragments only accepted for the exact base-repo
  commit the manifest names.
- Structural: shape/dtype/nbytes vs manifest; suh/svh sanity; NaN/Inf scan
  on a dequantized sample of tiles.
- `VLLM_FQ_TRUST = local | signed | any` (default `signed`): `signed`
  requires a valid attestation from a key in the operator's trust list.
- Optional paranoia: `VLLM_FQ_REMOTE_DENY` pins named experts/layers to
  local-encode only.
- Existing runtime guards still apply downstream (probe + rollback bound
  the blast radius of anything that slips through).

## 5. The offline tool

`fq_encode` (lives in the exllamav3 toolchain, shares the encoder with the
runtime executor): input = HF repo or local BF16; streams per-expert via
range reads (09); emits segment files + attestations; resumable
(per-fragment, content-addressed = idempotent); `--k 3` default,
`--layers`, `--experts` subsetting; `--publish` pushes segments +
attestations; `--verify repo` mode re-encodes a random fragment sample from
someone else's repo and countersigns. One tool = first-user encoder,
rebuilder/auditor, and CI check.

M0 absorbs `fq_encode` skeleton + segment schema + attestation format;
`--verify` and Sigstore integration can trail in M5/M6.

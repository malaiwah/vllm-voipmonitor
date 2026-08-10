# Progressive Loader v2 — streaming mixed-K boot from segments + policy (no assembled checkpoint)

Date: 2026-08-10. GPUs 0-3 (RTX PRO 6000 Blackwell), GG v20-r33 rootfs env, TP4.

**Outcome: PASS.** `--load-format progressive` boots the Fruit proxy
directly from Progressive Tensors segments
(`/home/mbelleau/fq-0c/fruit-segments`) plus a fq-policy/2 document — no
`fq_assemble` run, no assembled shards on disk. Two different policies
booted back-to-back from the same segment store; per-layer tiers followed
each policy exactly; the 042-policy boot produces **greedy outputs
token-identical to the assembled fruit-mixed-042 serve** (diffed in the
same session); decode throughput is at the assembled baseline.

## 1. What was built (gg-vllm `fq/m1-stats-collector`, commit `0d6d54196`)

- `vllm/model_executor/layers/quantization/exl3_fungible/fragments.py` —
  **FragmentResolver**: `(layer, expert, K) -> fragment payload` through
  the implementation/10 §2 resolution order: local segment dirs (manifest
  dir + `VLLM_FQ_LOCAL_SEGMENTS`) → manifest `sources` chain (HF ranged
  reads: `index-k{K}.json` → header range `[0, body_offset)` → per-expert
  body range; token from `HF_TOKEN`) → error (T3 local encode out of
  scope). Content addressing per fq-attestation/1 `expert_sha256`: every
  fetched fragment is sha256-verified and cached content-addressed under
  `VLLM_FQ_CACHE` (default `~/.cache/vllm/fq`;
  fragments/headers/attestations subtrees); cache hits re-hashed on read.
  `VLLM_FQ_VERIFY` = `fetched` (default) | `all` (also verify local
  reads) | `off`. Local segments are mmapped; tensors are zero-copy views.
- `.../progressive.py` — `ProgressiveSpec` (env resolution:
  `VLLM_FQ_MANIFEST_DIR`; `VLLM_FQ_POLICY` or M2 `PolicyStore`
  current.json keyed by the manifest hash; `VLLM_FQ_DENSE_SOURCE` →
  manifest `dense_source` → `--model` dir), the
  `progressive_weights_iterator` stream synthesis, tier-bitmap +
  `--hf-overrides` synthesis, and a CLI
  (`python -m ...exl3_fungible.progressive`) that emits both.
- `.../progressive_loader.py` — `ProgressiveModelLoader`, registered as
  load format `progressive` (`LoadFormats` literal +
  `_LOAD_FORMAT_TO_MODEL_LOADER`; documented in `LoadConfig`). Per
  worker: non-expert tensors (dense, attention, shared experts, router,
  bf16 MTP experts) stream zero-copy from the dense-source shards;
  routed-expert tensors resolve per-policy through the FragmentResolver,
  **pre-filtered to the worker's TP rank** (4× less expert-tensor
  materialization per rank). Name-set parity with the source shard header
  is enforced per layer (segment mismatch fails the boot loudly), and
  per-layer `FQ progressive layer L: tiers=... bits_digest=...` log lines
  make the applied policy auditable in the serve log.
- Serve-time metadata (the fruit-mixed-report.md §2 contract, previously
  baked into the assembled checkpoint by `fq_assemble`) is synthesized at
  launch: full `hybrid_tr3_tail` patched to `bits="mixed"` + `k_values` +
  `bits_per_expert` as an **absolute-path** `file.json:field` reference
  into a tier bitmap written from the policy under
  `VLLM_FQ_CACHE/boot/`, plus the `quantization_config` stub when the
  dense source lacks one — passed as `--hf-overrides` JSON
  (`get_hf_file_to_dict` → `try_get_local_file` resolves absolute
  reference paths because `Path(model) / abs_path == abs_path`).
  Launcher: `serve-progressive.sh` (this dir).

## 2. Pre-GPU validation (CPU byte parity)

Streaming `fruit-segments` with the 042 policy reproduces the assembled
`fruit-mixed-042` checkpoint **tensor-for-tensor, byte-for-byte**:
123,915 tensors, 0 mismatches, 0 missing, 4.2 s single-threaded (2,560
fragments, all local) — `preflight_parity.py`. CPU tests: 16 new
(resolver: local hit, source fallback + verify + cache, sha-mismatch
rejection, cache reuse without refetch, unverified-fetch refusal,
`verify=all` local corruption; stream: policy-driven trellis widths,
all-K3 byte identity, rank filter, missing-K loud failure, layer-coverage
enforcement; metadata synthesis) — package suite 62 green.

## 3. Boot A — 042 policy (the fruit-mixed-042 membership)

`serve-progressive.sh /home/mbelleau/fq-0c/policy-fruit-mixed-042.json fruit-prog 8802`

- Launch → `/health` 200: **92.6 s** (see §5 for the compile-cache
  breakdown). Weights stream 3.89 s (TP0); "Model loading took 1.09 GiB /
  4.70 s"; init engine 36.72 s of which torch.compile 22.09 s (cold
  cache).
- Fragments: 2,560/rank, all local, 0 fetched, 0 sha mismatches. Loader
  tier lines == policy exactly (`3: (185,71) … 12: (115,141)`), echoed by
  the exl3 planner
  (`EXL3 mixed Trellis model.layers.12.mlp.experts: tiers=((3, 115), (4, 141))`).
- Coherence (greedy, temperature 0): all four probe prompts produce
  **token-identical text to the assembled fruit-mixed-042 serve**
  (`diff bench-A.txt bench-baseline.txt` → identical) — same weights ⇒
  same greedy path.
- First-token time (max_tokens=1 after warmup): **0.024 s**.
- Throughput (single request, 512-token greedy decode): **495.6 tok/s**.

## 4. Boot B — permuted membership (policy-at-boot flexibility)

`policy-fruit-mixed-042-rotB.json`: cyclic layer rotation of the 042
membership (layer L takes layer L+1's 256-expert bits vector, wrap;
policy digest `58bb74f44647` vs `0451f1a92098`). Booted from the SAME
segment store with only the policy path changed.

- Launch → `/health` 200: **96.0 s**; weights stream 4.02 s.
- Loader + exl3 tier lines follow the rotated policy exactly — layer 3:
  (214,42) … layer 12: (185,71), and the per-layer `bits_digest` values
  are precisely boot A's digests shifted by one layer (df2691417c30 moved
  from layer 4 to layer 3, 290afb65c3ef from layer 3 to layer 12):
  different membership AND different per-layer counts, proving the boot
  is policy-driven, not checkpoint-driven.
- Coherence: prompt-dependent, grammatical continuations (fox/tea
  continuations differ from boot A, as expected for permuted expert
  precision). Throughput: **492.8 tok/s**; first token 0.024 s.

## 5. Assembled-checkpoint baseline (same session) + boot-time analysis

`serve-fruit.sh /home/mbelleau/fq-0c/fruit-mixed-042 fruit-mixed
--kv-cache-dtype fp8_ds_mla --hf-overrides '{"use_index_cache":true}'`
(the exact fruit-mixed-report.md command):

| | assembled (warm compile cache) | progressive A (cold) | progressive B (cold) |
|---|---|---|---|
| launch → health (s) | 61.0 | 92.6 | 96.0 |
| weights stream (s, TP0) | 2.05 | 3.89 | 4.02 |
| model load (GiB / s, TP0) | 1.09 / 3.02 | 1.09 / 4.70 | 1.09 / 4.83 |
| init engine (s / compile s) | 14.54 / 0.57 | 36.72 / 22.09 | ~37 / ~22 |
| 512-tok decode (tok/s) | 490.5 | 495.6 | 492.8 |
| first token (s) | 0.024 | 0.024 | 0.024 |
| greedy outputs | reference | **identical to reference** | differ (different policy) |
| per-policy disk artifacts | 3.7 GB assembled copy | none | none |

Analysis: the **streaming assembly itself costs +1.8 s** in the weights
phase (3.89 vs 2.05 s — per-expert fragment resolution + name parity
checks vs sequential shard reads) — noise against a ~60-95 s boot. The
remaining ~30 s gap is a **torch.compile cache miss, not loader cost**:
the per-policy tier-bitmap absolute path inside `hybrid_tr3_tail` enters
the compile-cache key, so boots A and B each compiled fresh
(`torch_compile_cache/5f4fd825f4` vs `c3ee4ca495`, 22 s compile) while
the many-times-served assembled checkpoint hit a warm cache (0.57 s).
Follow-up: write the tier bitmap to a **stable per-manifest path**
(content in the file, not in the name) so repeated policy boots share the
compile cache; with a warm cache the projected progressive boot is ~63 s
vs 61 s assembled. The progressive path also removes the fq_assemble
materialization step and the 3.7 GB per-policy assembled copy entirely —
a policy change now costs one serve restart and zero disk.

Decode throughput and first-token time are at parity (±1%, within
run-to-run noise; the 501.6 tok/s in fruit-mixed-report.md was a
different session).

## 6. The seam for the M4 swap engine

The swap engine reuses exactly the loader's fragment plane:

- `FragmentResolver.resolve(layer, expert, k) -> Fragment` and
  `expert_tensors(layer, expert, k, name_filter=...)` — same resolution
  order, verification and cache at swap time as at boot time. A swap =
  resolve the target-K fragment for the incoming expert (rank-filtered
  names), copy into the tier slabs, update the tiered maps. The resolver
  is process-local, holds mmaps + header tables, and its `stats` dict
  gives the fetch/verify counters the swap telemetry needs.
- `ProgressiveSpec` — one struct binding (manifest dir, policy, dense
  source); the swap engine holds one, calls `spec.make_resolver()`, and a
  policy delta is just a new bits vector fed to the same resolver.
- The per-layer `bits_digest` log line is the shared audit trail: boot
  writes it, swaps should re-log it after each committed batch.

## 7. Files / commits

- gg-vllm `fq/m1-stats-collector`: **`0d6d54196`** ("exl3_fungible:
  Progressive Loader v2 — stream mixed-K EXL3 from segments + policy"),
  pushed to `work`. Runtime overlay: the same files synced into the
  gg-v20-r33 rootfs venv (`opt/venv/.../vllm/`) — additive only (new
  exl3_fungible modules + the progressive registry entries in
  `model_loader/__init__.py` / `config/load.py`); existing serve paths
  untouched.
- This dir: `report.md`, `README.md`, `serve-progressive.sh`,
  `preflight_parity.py`, `bench-A.txt`, `bench-B.txt`,
  `bench-baseline.txt`, `boot-A.log`, `boot-B.log`, `boot-base.log`.
  Policy B: `/home/mbelleau/fq-0c/policy-fruit-mixed-042-rotB.json`.
- End state: the assembled fruit-mixed serve is restored on port 8801
  (tmux `fq:serve`); `.serve-quad.lock` released.

## 8. Open notes

- Remote (HF) source fetches are implemented + unit-tested via a fake
  source; no public segment repo was available to exercise end-to-end in
  this run — the priming agent's uploads are the natural first target.
- `fq-manifest.json` in fruit-segments is overwritten per repack run
  (currently `k_variants=[5]`, `sources=[local:…]`); loader v2 ignores
  the stale `k_variants` and would benefit from the manifest gaining a
  `dense_source` field and a cumulative `k_variants` list.
- Stable tier-bitmap path (per-manifest, not per-policy-digest) to keep
  the torch.compile cache warm across policy boots — see §5.
- Attestation signature verification (`VLLM_FQ_TRUST`) deferred per 10
  §4; content addressing (sha256) is enforced on every fetched fragment.

# 14 — What the build taught us (addendum to 00–13)

Docs `00`–`13` were written **before** anything was built. Between them and
this file sits one night of implementation on an 8× RTX PRO 6000 (SM120)
box: the M0 artifact toolchain, M1's collector binding, the M3 live-reload
path, the M4 swap engine's T3/T4 verdicts, Progressive Loader v2, and the
0c multi-K encode campaign. Evidence lives in `../runs/*/report.md`
(index: `../runs/README.md`).

This addendum enumerates every fact a reader of 00–13 would otherwise get
wrong. **Docs 00–13 keep their original text** — they are the historical
record of what was believed pre-build. Where a statement is now *wrong*
(not merely incomplete) it is called out here by doc + section, and the
doc itself carries a one-line `**Build note (2026-08-10):**` pointer back.

Tags: **CORRECTION** = the spec statement is wrong · **SHARPENING** = right
but dangerously incomplete · **CONFIRMED** = the spec guessed right, now
measured · **NEW** = a fact no doc anticipated.

---

## 1. The capture-fn binding site is gated (M1's integration fact)

**CORRECTION — `01-artifacts-policy-stats.md` §2.** The spec says:

> Hook: `BaseRouter.set_capture_fn` — **ungated**, per-layer, fires on
> logical ids after `_compute_routing` … bound per-MoERunner at
> `gpu_model_runner.py:7906-7919`

The *hook* is ungated. The **production binding site is not**: the call
that installs a capturer (`_bind_routed_experts_capturer`) only runs when
`enable_return_routed_experts=True`, which is **off by default**. Three
consecutive T1 attempts therefore measured nothing at all and would have
"passed" on a collector that was never bound
(`../runs/t1-graph-freeze/report.md` §Rig — "discovered after three hollow
runs"; root-caused in commit `7db2089ef`).

Consequences, both live:

- **M1 must bind via its own env-gated call** (`VLLM_FQ_ENABLE`), not by
  piggy-backing on `enable_return_routed_experts`. The shipped
  `exl3_fungible/integration.py` does exactly this.
- The capture-fn slot is single-occupancy, so the FQ collector **chains**
  whatever was bound before it (01 §2 already says this; it is now load
  bearing, because with the flag on, the mtp78-collector family is the
  other occupant).
- Any future "is the collector alive?" test must assert an **absolute**
  count, never merely "nonzero" or "grew" — a hollow bind produces a
  plausible-looking zero.

**CONFIRMED** — with the flag on, the load-bearing assumption of the whole
loop holds: counts grow monotonically *inside* CUDA-graph replay
(10800 → 21520), all 10 layers agree exactly at two run lengths, and the
gap to naive `tokens × top_k` is a **constant 16 routings (2 tokens × 8)**
across a 2× change in generation length — scheduler-boundary accounting,
not a scaling leak (`t1-graph-freeze/report.md`).

## 2. The mixed-checkpoint metadata contract (three things, all mandatory)

**NEW — nothing in 00–13 describes this.** `../runs/serve-baseline/fruit-mixed-report.md` §2
derives it from `exl3.py` (`_configure_rank_sliced`,
`_load_rank_sliced_bitrates`, gg-v20-r33). A mixed-K checkpoint needs:

1. **`config.json → hybrid_tr3_tail`**: `"bits": "mixed"` (the *string*),
   `"k_values": [3, 4]` (validated ⊆ 3..6), and
   `"bits_per_expert": "tier_bitmap.json:bits_per_expert"` — a
   `"file.json:field"` **reference string, not inline data**.
2. **The referenced JSON**: `str(layer) → {…, "bits_per_expert": [256 ints]}`
   for every layer in `moe_layers`. Values must be ⊆ `k_values`.
   *Loader quirk:* an entry carrying a 256-long `tail_tr3` field and no
   bitrate field silently defaults to **all-K3** (the big-model MTP
   convention) — a policy typo degrades quietly instead of failing.
3. **`config.json → quantization_config` stub**
   (`{"quant_method": "exl3", "bits": "mixed", "codebook": "mcg", …}`).
   vLLM's `weight_utils.get_quant_config` resolves the quant class from
   `hf_config.quantization_config` **before** `Exl3Config.maybe_update_config`
   ever reads `hybrid_tr3_tail`; without it boot dies with *"Cannot find the
   config file for exl3"*. The rank-sliced format ships no
   `quantization_config.json`, so the pure `fruit-k3`/`fruit-k4` checkpoints
   have the same gap and need an `--hf-overrides` workaround.

Plus a regenerated `model.safetensors.index.json` and `MANIFEST.sha256`.
`fq_assemble.py` emits all of it automatically when a policy uses >1 K; the
Progressive Loader v2 **synthesizes** it at launch instead (absolute-path
`file.json:field` reference into a tier bitmap under `VLLM_FQ_CACHE/boot/`,
passed as `--hf-overrides`) — see §6.

**NEW, and the most expensive bug of the night: `--kv-cache-dtype fp8_ds_mla`
is required with `B12X_MLA_SPARSE`.** With the default (auto → bf16,
`B12X_NON_COMPRESSED_INDEXER` cache layout) *both* the mixed and the pure-K3
checkpoints boot cleanly and then emit **prompt-independent degenerate
text** — identical continuation regardless of prompt, i.e. attention
contributes nothing. The sparse-MLA kernel stack is only correct with a
ds_mla KV layout. Diagnosed only because a pure-K3 sanity boot was planned:
K3 garbled identically ⇒ serve config, not the mixing. Any future
mixed-K serve script must pin the KV dtype
(`fruit-mixed-report.md` §3).

## 3. Device-state facts the swap engine must honor

Source-phase verdicts: `../runs/pre-m4-checks/report.md`; GPU verdicts:
`../runs/pre-m4-checks/occupancy-gpu-report.md`, `../runs/m4-swap/report.md`.

### 3.1 Rotations are COPIES → writes target the COMBINED tensors

**CORRECTION — `02-swap-engine.md` §"K6 verdict, applied"**, which says:

> Per-expert side tensors are row-independent: suh `[E,H]`, svh `[E,H]`,
> rotations `[E,3I]`, indexed by expert id in-kernel.

They are row-independent, but they are **not indexed by expert id and not
the per-tier tensors**. `combine_trellis_rotations` **copies**
(`torch.cat(...).contiguous()`, `mixed_trellis.py:1125-1136`); the forward
binds pointers **only** from the combined struct (`:1447-1502`), and the
per-tier source tensors are **dead after prepare**. So:

- Rotation / `gate_suh` / `up_suh` / `down_svh` writes go into the
  **combined** tensors at **combined-slot** indices (tier0 slots `[0,t0)`,
  tier1 `[t0, t0+t1)`), never into the per-tier sources.
- A tier swap changes **both** experts' combined slots, so **both** experts'
  rotation/suh/svh rows must be rewritten at their new slots, plus both
  maps, inside the quiesce window. This extends 02's row inventory; the
  commit ordering (slabs → rotations → maps → memo → persist) is unchanged.
- The M4 engine honors this and the M3 reload path re-derives it
  independently: r33's `exl3.py` builds the combined tensors itself and
  hands **tier-slice views** to prepare, so combined writes propagate
  everywhere (`../runs/m3-reload/report.md` §1.2).

### 3.2 Absence must be marked in `global_to_combined` — descriptor-only is silent garbage

**NEW.** To retire a slot you must set `global_to_combined[global] = -1`
(or out of range). The route packer drops such routes
(`route_pack.py:177-178`) and `topk_sum` zeroes them
(`kernel.py:7960-7967`). Marking **only** `descriptor_map` is a
**silent-garbage bug**: the route still packs, the GEMM tiles skip, and
`topk_sum` blends never-written fc2 rows into the output. Setting
`descriptor[retired] = -1` as well is belt-and-suspenders, not the
mechanism.

The M4 engine's fail-closed check is stronger still: since v1 never varies
occupancy, the rebuilt `global_to_combined` must be a **full permutation**,
validated before the live copy (`m4-swap/report.md` §"Design consequences").

Related hard rules from the same source pass: descriptor writes must be
exactly `local` or `256+local` (tier is the *unmasked* upper bits — `>>8`
on all bits, so a stray high bit silently changes tier); map contents are
read at replay time on the compute stream, so mutation must be ordered
against replay.

### 3.3 Occupancy < capacity is safe — measured, not inferred

**CONFIRMED — `05-variable-cardinality.md` §2.1** asked for exactly this
test and it now exists and passes. C=16 (12 K3 + 4 K4), N=10 with 6
combined slots retired; full-range random int32 scribbled into the retired
`w13`/`w2` rows, NaN into their global scales and rotation/suh/svh rows.
**All 7 cases bitwise-equal** to the clean reference, which itself matches
the fresh full-map layer and a serial per-tier oracle at rel 4.7e-08.
Non-vacuous: the leakage control (same scribbles, routing that *does*
reference the retired slots) turns **1024/1024 output elements NaN**.
L2 (runtime cardinality within pre-provisioned capacity) is unblocked on
the kernel side.

### 3.4 Shared-H (`broadcast_suh`/`broadcast_svh`) layouts are out of v1

**NEW.** r19+ shared-H checkpoints have **one shared** suh/svh row per
(layer, proj, rank), so per-expert suh/svh writes do not exist there. The
M4 engine **refuses** broadcast layouts at construction; v1 targets the
per-expert layout. (Note for later: under shared-H a swap is *cheaper* —
only trellis rows + `mcg` + `rotations.intermediate` need rewriting.)

### 3.5 MCG is per-layer and pinned

**NEW.** Staged fragments must agree on the layer's codebook word (fruit
segments: `-877912083` = `0xCBAC1FED` across every expert/proj sampled;
the same constant appears in the 3.42bpw community quant's config as
`mcg_multiplier = 3417055213`). Foreign-mcg fragments are refused at
staging: r33 pins the MCG LUT ABI and there is **no per-expert mcg
plumbing in the mixed kernel**. This is a hard constraint on mixing
fragments across encodes.

## 4. Determinism is stack-scoped — attestations must name their stack

**CORRECTION — `10-shareable-segments-provenance.md` §3.3**, which claims
without qualification:

> **The killer property — deterministic re-encode**: given (BF16 tensor,
> hessian_id's statistic, encoder version, config), the trellis encode is
> bit-reproducible. So any third party holding the BF16 can re-encode any
> fragment and countersign.

Bit-reproducibility is real but **scoped to a stack**, and the boundary was
measured three different ways tonight:

1. **CUDA `pow()` is 1 ulp off CPU.** `inv_freq` must be computed on CPU
   and moved to GPU, as `from_pretrained` does. CUDA `pow()` differs from
   CPU on **3 of 32 exponents** at `rope_theta=5e5`; that alone perturbed
   cos/sin enough to **flip ~1.4 % of routings per layer**, compounding to
   an **88 % id match by layer 12**. With CPU-init rotary the pipeline is
   bit-exact (`../runs/0c-campaign/capture-stream-report.md` §"Rotary gotcha").
2. **Kernels are not row-stable across batch shapes.** Packed multi-sample
   batches (8192-token budget, block-diagonal masks) are semantically
   correct yet sdpa/GEMM reduction-order noise (~1 ulp) flips ~2 % of
   near-boundary routings per layer: ids match **97.8 % at layer 3 → 81.6 %
   at layer 12** vs the sealed reference. `grouped_mm` — and plain cuBLAS
   GEMMs including the fp32 router — return **different bits for the same
   row when the batch composition changes**. This is why the capture runs
   in exact mode (batch=1 per sample), and why a *serving engine's*
   continuous batching can never reproduce a reference capture.
3. **Cross-stack activation drift** is the same phenomenon at repo scale:
   legacy community quants recorded neither their Hessians nor their stack,
   so byte-identity with them is **honestly out of reach**
   (`0c-campaign/quant-342-layout-report.md` §Salvage class (c);
   `MULTI-K-PLAN.md` §"Attestation rung 3").

Therefore (`0c-campaign/ATTESTATION-V2.md`): an `encode-of` attestation
**must** carry its determinism scope — encoder sha, exllamav3 version,
torch+CUDA build, GPU arch, capture-methodology version — plus full
`quant_args` (K, seed_base, sigma_reg, codebook + mcg multiplier,
out_scales mode, slice-seed formula version) and capture lineage (capture
fingerprint, corpus sha, plan seed, tokens). A countersignature is a claim
about *that* scope, not a universal one.

**NEW — the predicate ladder is five rungs, not two.** 10 §3 knows
`repack-of` and `encode-of`. The full vocabulary
(`ATTESTATION-V2.md`, `MULTI-K-PLAN.md` §"Attestation rung 3"):

| Predicate | Claim | Status |
|---|---|---|
| `repack-of` | byte-identity with a pinned source (transport fidelity) | **emitted today** by `fq_repack` |
| `encode-of` | recorded recipe ⇒ independent re-encode byte-matches, within scope | designed; **not yet emitted** (see §9) |
| `derived-from` | exact deterministic view of another fragment (e.g. shared-H → per-expert expansion) | designed; in the loader's default trust set |
| `equivalence-of` | two fragments' reconstruction errors vs the *same* BF16 ground truth, side by side — validity without byte-identity, the honest rung for legacy quants | designed |
| `assembly-of` | recipe + segment shas → output shard shas; makes assembled checkpoints reproducible artifacts | designed |

The loader's **default** trusted set is `repack-of, encode-of,
derived-from` (`../runs/loader-v2/trust-and-lazy-encode.md` §2); the last
two are design-only so far.

## 5. Measured numbers that replace the spec's estimates

| Quantity | Spec said | Measured | Source |
|---|---|---|---|
| Encode, one expert, K3/K4 | **7.5 s** (07 §"Is Trellis streamable?", §Pipeline 3) | **2.55 s** K3 / **2.48 s** K4 cold; 2.39 / 2.32 s warm-H — **3× better** | `encode-bench/report.md` |
| Encode, one expert, real campaign | — | **~4.8 s** K2, **~3.4 s** K5 on real GLM-5.2 (window-1, layers 3–10, real Hessians) | `0c-campaign/glm52-encode-k{2,5}.log` |
| Per-K cost independence | — | K3 ≈ K4 **within 3 %**; **but K2 ≈ 2× K3** (DP table 16× K5's) — the "no per-K distinction" conclusion holds for K3/K4 only | `encode-bench/report.md` finding 2; `MULTI-K-PLAN.md` |
| Full K4 overlay, all routed experts | **~41 GPU-h** (07, 08 rung 3) | **~13 GPU-h** on one card (~3.3 h on an idle quad) | `encode-bench/report.md` finding 4 |
| Experts/hour at 5 % encode budget | — | **~71/h** on one GPU; a ~100-expert working set fully K4s in **~1.4 h** background | `encode-bench/report.md` finding 3 |
| ε ladder per +1 bit | qualitative | **3.91× / 3.84× / 3.79×** (mean rel-RT-MSE: K2 **0.09027**, K3 **0.02310**, K4 **0.00602**, K5 **0.00159**) — a clean geometric ladder, ~3.8×/bit | `0c-campaign/report.md`, `eps-analysis.json` |
| K2 abort criterion (04 §Abort) | might fire | **does NOT fire**: Δε/expert is uniform-ish (median CV **0.047**) but benefit = Δε·φ is concentrated — **median Gini 0.48**, median top-16-of-256 benefit share **0.318**. Per-expert allocation is the right granularity; the win comes from **routing mass**, not ε spread | `0c-campaign/report.md`, `eps-analysis.json` |
| Global solve vs uniform N_L | unknown | **+1.3–2.8 %** benefit at layer level on the 10-layer proxy; solve output genuinely non-uniform (`n_k4_per_layer` **42…152** at the 0.42 budget) | `0c-campaign/report.md` |
| M3 reload stall | "seconds-long" (04 M3) | **0.410 s / 0.466 s** total stall, **0 request drops**, on a live TP4 HTTP serve under continuous traffic — budget beaten ~10× | `m3-reload/report.md` §4 |
| M3 correctness gate | logits == fresh boot | **max \|Δlogprob\| = 0.0** on every one of 356 scored tokens, twice (cross-process and round-trip), all 4 ranks agreeing; policy sha returns exactly to the boot value | `m3-reload/report.md` §4 |
| Restart-swap floor (RUNG A) | — | **88.0 s** downtime last-healthy → first-healthy with warm compile caches | `m3-reload/report.md` §3 |
| M4 quiesce window | "< 1 engine step"; ~25–50 ms H2D at caps | **0.061 ms** (1 pair) / **0.368 ms** (8 pairs) best-case on a toy layer — this measures the **fixed** overhead (op issue + map flip + sync), not PCIe: toy payloads are ~350× smaller per expert than GLM-5.2 rank shards (46 KB vs ~7.9 MB/pair). The 02 H2D budget at 64 pairs (~504 MB) still governs | `m4-swap/report.md` §Timings |
| Mixed-K execution cost | unknown | **~0 %**: 501.6 tok/s mixed vs 503.1 tok/s pure-K3 (512-tok greedy, bs1, Fruit proxy) | `serve-baseline/fruit-mixed-report.md` §4 |
| Progressive (segment) boot cost | unknown | **+1.8 s** in the weights phase (3.89 s vs 2.05 s assembled); decode tok/s and first-token time **at parity** (±1 %) | `loader-v2/report.md` §5 |
| Assembly byte-identity | M0 gate | **79/79 layer shards sha256-identical** to `brandonmusic@9297b9f1` at full-model scale | `m0-assemble/verify.log.summary` |
| Reflink assembly | — | `fq_assemble --reflink`: a 3.7 GB mixed checkpoint in **3.8 s** (12,288 regions/layer reflinked) | `m3-reload/report.md` §2 |

### 5.1 The loader-v2 compile-cache caveat (don't misread the boot time)

Progressive boots measured 92.6 s / 96.0 s vs 61.0 s assembled — but
**~30 s of that gap is a torch.compile cache miss, not loader cost**. The
per-policy tier-bitmap **absolute path** inside `hybrid_tr3_tail` enters
the compile-cache key, so each policy boot compiled fresh (22 s) while the
many-times-served assembled checkpoint hit a warm cache (0.57 s). Fix
identified: write the tier bitmap to a **stable per-manifest path**
(content in the file, not in the name); projected warm progressive boot
~63 s vs 61 s assembled (`loader-v2/report.md` §5). Anyone quoting the
92.6 s number without this caveat is quoting a compile, not a loader.

### 5.2 Canonical model arithmetic (the docs disagree with each other)

Header-verified from the real checkpoints
(`0c-campaign/quant-342-layout-report.md`): GLM-5.2 is **78 hidden layers
+ 1 MTP layer at index 78**; MoE layers are **3..78 inclusive = 76 layers ×
256 experts = 19,456 routed experts**. Docs variously say 19,712 (07:
77×256), 19,200 (`encode-bench`: 75×256, MTP excluded) and `moe_layers:
[3, 77]` (01 §1.1). **Use 19,456 / 76 layers / `[3, 78]`.**

Measured per-layer segment sizes (76 MoE layers): K2 **2.458 GB**, K3
**3.67 GB**, K5 **6.082 GB** → full-model families ≈ K2 **187 GB**, K3
**279 GB** (278.6 GB actually uploaded), K4 ≈ **372 GB**, K5 **462 GB**.
`MULTI-K-PLAN.md`'s estimates (173 / 260 / 347 / 433 GB) run ~7 % low.

## 6. Progressive Loader v2 — boot from segments, no assembled checkpoint

**NEW capability; 08 and 10 sketched the cache tiers, nothing specified
this.** `--load-format progressive` boots directly from Progressive
Tensors segments + an `fq-policy/2` document: no `fq_assemble` run, no
assembled shards on disk, **zero per-policy disk cost** (vs a 3.7 GB
assembled copy per policy). Verified (`loader-v2/report.md`):

- **CPU byte parity first**: streaming the segments under the 042 policy
  reproduces the assembled checkpoint **tensor-for-tensor, byte-for-byte**
  — 123,915 tensors, 0 mismatches, 4.2 s single-threaded.
- **Boot A** (042 policy): greedy outputs **token-identical** to the
  assembled `fruit-mixed-042` serve; 495.6 tok/s.
- **Boot B** (rotated membership, same segment store, only the policy path
  changed): tier lines and per-layer `bits_digest` values are boot A's
  digests **shifted by exactly one layer** — proof the boot is
  policy-driven, not checkpoint-driven.
- Expert tensors are **pre-filtered to the worker's TP rank** (4× less
  materialization per rank); name-set parity with the source shard header
  is enforced per layer, so a segment mismatch fails the boot loudly.

This makes 08's "the system has no build step" concrete a milestone
earlier than planned, and it hands M4 its fragment plane: the swap engine
reuses `FragmentResolver.resolve(layer, expert, k)` with the same
resolution order, sha verification and cache at swap time as at boot time
(`loader-v2/report.md` §6, `m4-swap/report.md` §Seams).

## 7. Trust and lazy-encode as actually shipped (knob names differ from 10 §4)

**CORRECTION — `10-shareable-segments-provenance.md` §4** specifies
`VLLM_FQ_TRUST = local | signed | any` (default `signed`). What shipped
(`loader-v2/trust-and-lazy-encode.md`) is a different, finer shape:

```
VLLM_FQ_SOURCES=repoA@main,org/repoB        VLLM_FQ_SOURCES_MODE=prepend|replace|append
VLLM_FQ_TRUST_SIGNERS=<hex ed25519 pubkey>[,…]      # default: manifest signer_pubkey
VLLM_FQ_TRUST_PREDICATES=repack-of,encode-of,derived-from   # default shown
VLLM_FQ_K_FALLBACK=3      VLLM_FQ_ENCODE_QUEUE=<path>
VLLM_FQ_VERIFY=fetched|all|off      VLLM_FQ_CACHE=<dir>      VLLM_FQ_LOCAL_SEGMENTS=<dirs>
VLLM_FQ_MANIFEST_DIR  VLLM_FQ_POLICY  VLLM_FQ_DENSE_SOURCE
VLLM_FQ_BF16_DIR  VLLM_FQ_CAPTURE_DIR  VLLM_FQ_ENCODER_CMD
```

Semantics worth knowing:

- **Trust is armed only when an anchor exists** (env signers set, or the
  manifest carries `signer_pubkey`). Without an anchor, behavior is
  legacy sha-only. There is no `any`/`local` mode; **sha verification is
  unconditional** either way.
- **Countersignatures work by construction**: any allowed line in a
  source's `attestations/layer-LLL.kK.jsonl` that passes the predicate
  filter accepts; a rogue line sitting next to a trusted one does not
  block. The fetched bytes are then verified against **that trusted
  line's** `expert_sha256`.
- **Per-source attestation caches are isolated**
  (`VLLM_FQ_CACHE/attestations/<source>/…`) so mirrors carrying *different*
  encodings of the same (layer, K) cannot poison each other.
- **Boot never blocks on a missing K.** `VLLM_FQ_K_FALLBACK` walks a
  substitute ladder and returns the fragment **marked**
  (`Fragment.requested_k`, `.substituted`); the per-layer tier line and
  `bits_digest` are computed from **reality**, and `actual_bits_out` feeds
  `write_tier_bitmap(..., actual_bits=…)` so the serve's `hybrid_tr3_tail`
  records **loaded** Ks, not wishes. Every substitution *and* hard miss
  appends to a persisted encode queue (dedup by `(L,E,K)`), drained by a
  worker CLI whose `--drain` mode is a dry run by default.
  *Caveat carried from the report:* for a GPU serve with substitutions the
  bitmap must be synthesized from a **pre-flighted resolve pass**, because
  the exl3 planner sizes slabs from it before weights stream — CPU-validated
  only so far.
- Every `resolve()` emits one structured decision line
  (`MISS` / `REJECT <reason>` / `ACCEPT` / `FALLBACK K<k>` / `UNAVAILABLE`)
  with matching `resolver.stats` counters. This is the audit trail 10 §3
  wanted and never specified.

**Also new:** `VLLM_FQ_CAPACITY_UTILIZATION` (default 1.0) — the
`gpu-memory-utilization` analog for tier headroom. `C = ceil(N / util)`
bounded by E and the global byte budget; `util=1.0` reproduces v1 exactly,
`util=0.9` pre-provisions ~11 % spare upper-tier rows and unlocks
displacement-free upgrades under the two-ledger check. This turns
`05-variable-cardinality.md`'s L2 from an arithmetic exercise into one
operator slider (`MULTI-K-PLAN.md` §"Capacity as an operator knob").

## 8. `shared_h_v1` salvage: the expansion is exact, and it is a *view*

`0c-campaign/quant-342-layout-report.md`, byte-level against
`willfalco/GLM-5.2-EXL3-TR3-3.42bpw @ ae68c659` (ranged reads only, no
shard downloaded in full).

- **Layout confirmed from bytes, not metadata**: 12 per-layer shared rows
  (`experts.shared_h.{gate,up}_proj.rank{0-3}.suh`,
  `shared_h.down_proj.rank{0-3}.svh`, each `F16 [6144]`) + **36 tensors per
  routed expert** instead of `per_expert_v1`'s 48 → **9,228 EXL3 tensors
  per MoE layer**. Pattern counts of exactly 1024 (= 256×4) prove no expert
  deviates.
- **The expansion is bit-exact *by construction*, and its size signature is
  verified**: replicating the layer's shared `suh`/`svh` into each expert's
  slots yields a 48-tensor unit with the exact `per_expert_v1`
  name/shape/dtype signature — `14,168,112 + 147,456 = 14,315,568 B`,
  byte-for-byte the size of a native `per_expert_v1` K3 expert. Both
  layouts decode by the same diagonal sandwich
  `W = diag(v_out)·dequant(trellis, mcg)·diag(v_in)`, so a plain
  `per_expert_v1` loader multiplying the stored vectors decodes identical
  weights. **Pedantic caveat, still open:** the report explicitly defers a
  one-expert numeric decode-and-diff under both loaders. Treat "bit-exact"
  as *algebraically exact with the size signature verified from bytes*,
  **not yet measured end-to-end**.
- Cost of the expansion: **+147,456 B/expert** → 2.67 GiB over the family
  (292.06 GiB → 294.74 GiB). Recommendation: store class (b) (the shared-H
  family) and materialize the expanded per-expert view **on demand** — it
  is a pure function.
- Predicate is **`derived-from`, not `repack-of`** (the replication rule
  and the source shared-row digests must be recorded), because bytes are
  added.
- **8,042 K4 fragments are salvageable** (tier_bitmap: layer 3 = 206 K3 +
  50 K4; every layer 4–77 = 148 K3 + 108 K4; layer 78 = 256 K3 → **11,414
  K3 + 8,042 K4 = 19,456**), and they are **mixable with the brandonmusic
  base**: same trellis/mcg geometry, same mcg multiplier, same calibration
  corpus `cf247acc…`. Mixing is a quality decision, not a compatibility one.
- **Not salvageable:** re-basing a shared-H trellis onto different H rows
  (the codes were chosen by LDLQ *with the shared profile forced on the H
  side*; no closed-form remap), the reverse dedup direction, and sub-expert
  mixing across calibrations.

## 9. Provenance defects in the published artifact (found while auditing)

**NEW — action items, not design changes.** The window-1 K2/K5 publish
(`0c-campaign/publish_window.py`) routes fresh encodes through `fq_repack`,
which hard-codes `predicate: "repack-of"` and writes a **whole-repo**
`fq-manifest.json`. On the published repo this produced:

- K2/K5 attestations with **`predicate: repack-of`** and materials
  `{repo: "local:glm52-k5-encode-of-window1", revision: "window1",
  file_sha256: null}` — **not third-party verifiable**, and the wrong
  predicate for what are in fact fresh encodes. `MULTI-K-PLAN.md` specifies
  `encode-of` with `hessian_id` = capture manifest hash and encoder sha
  `e9a85a47…`; none of that is carried.
- The repo-root `fq-manifest.json` **overwritten** by the last repack:
  `k_variants: [5]`, `sources: ["local:glm52-k5-encode-of-window1"]`,
  `revision: "window1"`, `tensor_index: "index-k5.json"`,
  `moe_layers: [3,10]` — the brandonmusic `9297b9f1` pin that described the
  K3 base is gone from the published manifest. (`loader-v2/report.md` §8
  flagged the same overwrite on the Fruit segment store; the loader ignores
  the stale `k_variants`, so nothing breaks at boot — but the *provenance
  story* does.)
- The signer key is stable across runs (`a58b7bb7…`), so signature
  verification still works.

Fixes needed before this family is presented as `encode-of` evidence:
`fq_repack` grows `--predicate` + attestation-v2 materials; the manifest
becomes **cumulative** (merge `k_variants`, `sources`, per-K
`tensor_index`) instead of last-writer-wins; and it should gain the
`dense_source` field loader-v2 asked for.

## 10. Milestone reality vs `04-milestones.md`

| Milestone / test | 04 says | Reality 2026-08-10 |
|---|---|---|
| **M0** | gate = assemble any policy into a bootable mixed checkpoint | **DONE, twice over**: all-K3 assembly **79/79 shards byte-identical**; first true mixed-K checkpoint (`fruit-mixed-042`) boots TP4 and generates coherently, per-layer K4 counts == policy exactly |
| **M1** | gate = T1 green on 1 GPU **and TP4**, overhead < 0.5 % at cc8 | **T1 PASS on 1 GPU** under FULL cudagraph capture. **TP4 leg and the decode-overhead measurement are NOT done** |
| **M2** | dryrun loop, T2 green, 48 h shadow run | policy engine + T2 properties green as a CPU prototype (doc 13) and `policy.py`/`store.PolicyStore` exist in-tree; **the 48 h dryrun has not run** |
| **M3** | "seconds-long stall"; gate = T8 + logits == fresh boot | **DONE and beaten**: 0.41/0.47 s stall, 0 drops, logits **bit-identical** twice. Note the gate actually met was the logits equality — **T8 (kill -9 rehydration) was not run** |
| **M4** | gate = T3, T4, T5 green; stall < 1 engine step; T7 soak | **T3 PASS** (bitwise, non-vacuous, incl. cross-tier moves) and **T4 PASS ×3** (+ bitwise rollback), 41/41 CPU package tests. **T5, T6, T7 not run.** Integration into the live layer (`MixedLayerState.from_exl3_mixed_trellis()`) is a named seam, not done |
| **T3 abort signal** ("maps baked ⇒ ceiling is M3") | open risk | **RETIRED. Maps are read as data under CUDA graph** — the atomic path is live; `APPLY_MODE=reload` is not the ceiling |
| **K2 abort signal** ("homogeneous sensitivity ⇒ pivot to layer-level") | open risk | **RETIRED — does not fire** (§5) |
| Pre-M4 b12x checklist (02 §Pre-M4) | 3 source checks, ~1 d | **4/4 + the occupancy GPU test CLOSED** (§3) |

## 11. Corrections index (doc + section → what is now wrong)

| Doc §                                  | Wrong statement | Correction |
|---|---|---|
| `01` §2 | capture-fn hook is "**ungated**" and bound per-MoERunner | the *binding site* is gated on `enable_return_routed_experts` (default off); M1 binds via its own env-gated call — §1 |
| `01` §1.1 | `"moe_layers": [3, 77]`; K3 base "~260 GB" | `[3, 78]` (76 MoE layers, 19,456 experts); K3 base measured **278.6 GB** — §5.2 |
| `01` §1.1 | artifact pair as `base-k3/` + `overlay-k4/` directories | shipped form is per-layer-per-K segments + `index-kK.json` + `attestations/` (10 §1); `fq-manifest/1` as emitted also carries `predicate`, `layout`, `sources`, `signer_pubkey` |
| `02` §"K6 verdict" | side tensors "indexed by expert id in-kernel" | they live in **combined** tensors indexed by **combined slot**; per-tier sources are dead after prepare — §3.1 |
| `02` §Data flow (3), §Commit protocol (2) | "suh/svh/rotation rows" | rows of the **combined** tables at both experts' **new** slots — §3.1 |
| `02` §Pre-M4 checklist | three open questions | all answered: maps `(tier<<8)\|local` PASS, no host-side map reads PASS, `combine_trellis_rotations` **COPIES** — §3 |
| `03` T1 | "Run twice: eager and graphed; results must match"; "matches `tokens × top_k` within tolerance" | ran **graphed-only** (the eager path is broken on the bf16 stack; triton lacks MoERunner hooks) with an **absolute-count** referee and a constant 16-routing boundary offset validated at two run lengths — §1 |
| `03` T4 | "(If K6 resolves to slab-rebuild-only…)" | K6 resolved to **row-write**; T4 as run also covers rollback, mcg agreement and broadcast-layout refusal |
| `04` M3 | "Seconds-long stall"; `pause_scheduler(mode="keep")` | **0.41–0.47 s**; `mode=wait` (drain) is used — see D6 below |
| `00` D6 | quiesce via `pause_scheduler(mode="keep")` | `mode=keep` would freeze in-flight requests across the swap (**mixed-provenance KV**); the shipped path uses `POST /pause?mode=wait` (drain). `mode=wait` **cannot** be used on an inproc engine; it needs the multiproc serve **and** `VLLM_SERVER_DEV_MODE=1` |
| `05` §L5 | "`_TRELLIS256_BITS=(3,4,5,6)` blocks K2" | blocks K2 **execution** only. The **encoder accepts bits 2–5** (K2/K5 smokes PASS on SM120) and K2 segments are produced and published today |
| `06` §2 P3 | Ks the lazy encoder may target: {3,4} vs {3,4,6} | the campaign produced **{2,3,4,5}**; K5 executes on today's kernel, K2 does not |
| `07` §Streamable, §Pipeline 3, §Honest costs | 7.5 s/expert; ~41 GPU-h; 19,712 experts | **2.5 s** (K3/K4), **~13 GPU-h**, **19,456** experts — §5, §5.2 |
| `08` §Cold-boot ladder rung 3 | "~41 GPU-h across 4 GPUs ≈ 10 h" | **~13 GPU-h ≈ 3.3 h** on a quad |
| `10` §3.3 | deterministic re-encode is universally bit-reproducible | **stack-scoped**; cross-stack only `equivalence-of` (bounded ε) is honest — §4 |
| `10` §4 | `VLLM_FQ_TRUST = local \| signed \| any` | shipped as `VLLM_FQ_TRUST_SIGNERS` + `VLLM_FQ_TRUST_PREDICATES`, armed only when an anchor exists; sha verification unconditional — §7 |

Minor, not corrected in place: `00`'s architecture diagram labels the
collector/policy/swap/store components `(02)/(03)/(04)/(01)`, while the
component table directly beneath maps them to `01/01/02/01`. The table is
right.

## 12. Still unproven (do not cite these as done)

- **T1 at TP4**, and the M1 decode-overhead measurement (< 0.5 % at cc8)
  on the GLM-5.2 serve with `VLLM_FQ_ENABLE=1`.
- **T5** (torn-update fault injection) — the `step_hook` seam exists.
- **T6** (cross-rank agreement over 50 intervals), **T7** (24 h soak),
  **T8** as specified (kill -9 mid-batch rehydration), **T9** (quality
  ladder).
- The M4 engine **on a live layer**: `MixedLayerState.from_exl3_mixed_trellis()`
  and the `FragmentSource`→`FragmentResolver` wiring are seams, not
  integrations. All M4 evidence is on a toy layer (E=32, H=I=128).
- **Remote (HF) fragment fetch end-to-end** — implemented and unit-tested
  against a fake source; no public segment repo was reachable in the run.
- The **one-expert numeric check** for the shared-H → per-expert expansion
  (§8).
- Every quality number is **proxy-level**: the Fruit proxy is a 5.04B
  assistant-masked SFT model with 10 MoE layers. The 0c ε ladder, the
  solve, the mixed-boot coherence and all M3/M4 fidelity gates are on it,
  not on GLM-5.2. The GLM-5.2 leg has capture (window 1–2) and K2/K5
  encodes for layers 3–10 only.

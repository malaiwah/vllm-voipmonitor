# Changelog — Fungible Quant / Progressive Tensors

Branch: `claude/gg-overview-exploration-jchgd3`
Base: `origin/main` (merge-base `99267c23`)
Commits: 236
Scope: Research, tooling, encode campaigns, milestones (M0–M5), health monitoring, environment setup, documentation, battle tests (BT-0–BT-6c), heatmaps, and rebase/PR work for variable per-expert bit-width reallocation (EXL3 K2–K5) of GLM-5.2 on the GG vLLM stack (4× RTX PRO 6000, SM120, TP4).

---

## Research & Feasibility

### Fungible-Quant feasibility study and implementation plan
**Commits:** `c4fa1b81`, `f36875ea`, `d99c9815`
- Archived the full multi-agent research campaign on periodic workload-driven per-expert bit-width reallocation (EXL3 K3/K4) in vLLM. Produced the `PLAN.md` verdict (fixed-cardinality tier-membership swap design chosen), prior-art survey (DynaExq, MorphServe, MC-MoE, MatQuant, HOBBIT), policy design, phased plan, upstreaming strategy, and kill criteria.
- Citation verification pass: 18/18 arXiv references resolve, 0 refuted. Applied corrections (MatQuant venue, HOBBIT arXiv ID, LHU terminology). GitHub issue/RFC verification: 24/24 references resolve.
- Added upstream comment drafts for RFC #49702 and RFC #48920, grounded only in verified references.

### Implementation specification (M0–M6)
**Commit:** `62219fa3`
- `00-overview.md`: architecture, product statement (new-model day-one: one generic two-tier artifact pair, deployments self-specialize), binding decision log D1–D9.
- `01-artifacts-policy-stats.md`: FQ artifact pair layout, policy JSON schema, graph-safe stats collector, argsort policy engine with guards, env knobs.
- `02-swap-engine.md`: swap-engine design (row-write path selected).
- `03-testing-validation.md`: test ladder T1–T9 (graph-freeze, torn-update fault injection, 24h soak release gate).
- `04-milestones.md`: build order M0–M6 with acceptance gates and abort signals.

### GG integration surface audit + capture-fn chaining fix
**Commit:** `914b97f9`
- Audited `exl3.py` (2447 lines) against `dev/gilded-gnosis @ e2666d9a`: mixed-tier layer state, swap hook points, quiesce endpoints, env-knob registration, bitrate JSON loader. Found and fixed a design bug: `set_capture_fn` is single-occupancy and fires on logical IDs, so the FQ collector must chain the prior capture function rather than overwrite it.

### Swap-engine design finalized — row-write path selected
**Commit:** `682465d6`
- Resolved K6 (sparkinfer mixed-trellis code) as YES (qualified): `prepare_trellis256_moe_weights` is zero-copy/view-only, slabs are expert-major checkpoint-native EXL3 tiles, and maps are launch-argument data whose content mutation is CUDA-graph-safe. Selected the row-write swap variant; retired the slab-rebuild fallback.

### Variable per-layer cardinality design
**Commit:** `b73233bb`
- Designed a cardinality ladder from already-supported (per-layer N_L and bit-pair at startup) through runtime cardinality within pre-provisioned capacity. Capacity stays compiled state, occupancy becomes map data. The policy engine gains a slow cross-layer loop emitting byte-conserving grow/shrink transactions.

### Decision checklist and effort summary
**Commit:** `7292510a`
- P1–P3 operator decisions, accept-or-veto defaults table, 6–8 week effort breakdown with M2/M3 early-exit products.

### Lazy K4 encode-and-cache design (D3→D3'→D3'')
**Commits:** `00d4a7d3`, `2ce34868`, `253b3d82`
- Lazy K4 encode design: candidate Hessians live in DRAM (collector-v3 pattern: device ring + background drain), GPU footprint is just the ring; H visits the GPU transiently at encode time. Promotion candidates are hot experts with well-fed live Hessians, so EAQuant's under-activation failure cannot occur in the promotion direction.
- Streaming realignment audit: K3 base reframed as a warmed remote-cache seed, epsilon curves late-bound, unified T0–T3 cache hierarchy, BF16 needed only lazily per-expert at encode time.

### HF range-read de-risk
**Commit:** `297ada1e`
- Live test against `zai-org/GLM-5.2`: CDN honors Range (206) after redirect, per-expert BF16 tensors individually addressable, three projections byte-adjacent (one coalesced 75.5 MB read per expert), 4-way parallel reads scale 3.2×. Confirmed T3-remote sourcing viable today.

### Shareable encode segments on HF + provenance model
**Commit:** `90b3f16b`
- Designed the community-publishing model: per-layer per-K safetensors segment files (range-readable per expert), provenance via content addressing + DSSE/in-toto attestations, deterministic-re-encode reproducible-builds trust model, load-time gates, and an `fq_encode` offline tool.

### CPU de-risk — Phase-0a routing stability + policy prototype
**Commits:** `bef2f3f5`, `ec87e99e`
- CPU-only routing-stability analysis on the real MTP78 capture (7.29M tokens × top-8 = 58.3M routing slots over 256 experts). Verdict: aggregate top-108 allocation converges within ~2% of the corpus (τ > 0.9 after ~150k tokens), but individual ~73k-token windows sit below 0.9 — short-horizon set membership is intrinsically fuzzy.
- Policy prototype: full T2 property-test suite (10 tests) passing — determinism, budget invariance, pin/dwell/hysteresis/caps.

### Naming due diligence
**Commit:** `6b1e1a16`
- "Progressive Tensors" / "fungible quant" — no collisions found on GitHub/PyPI/HF/arXiv. Added the "Progressive Tensors" prose name (a K3 base everyone shares plus per-expert K4+ overlays).

### On-box bootstrap + interruptibility policy
**Commit:** `7d99dad9`
- Iron rules for every session on the rental box (disk survives, memory does not): work inside tmux, commit-and-push after every completed step, long jobs write progress to `state.json` every few minutes and are crash-resumable, secrets in `~/.fq_env`.

### Attestation v2 design
**Commit:** `e71859da`
- fq-attestation/2 metadata: DSSE/in-toto envelope (payloadType + Statement, subject = fragment digest), encode-of determinism scope (encoder sha, exllamav3 version, torch+CUDA build, GPU arch), ride-along per-expert `rel_rt_mse` + `routed_count` for cold-start policy, trust-graph predicates (repack-of, encode-of, derived-from, equivalence-of, assembly-of).

### Attestation rung 3 — equivalence-of predicate
**Commit:** `73a25dab`
- For legacy fragments with unrecorded recipes: encode the same expert from BF16, decode both fragments, attest their reconstruction errors against the same BF16 ground truth side by side (3-way: BF16 ⇔ our encode ⇔ community fragment). Proves a hydrated community segment is valid without claiming unattainable byte-identity.

### VLLM_FQ_CAPACITY_UTILIZATION design
**Commit:** `e2011998`
- Operator knob (default 1.0) analogous to `gpu-memory-utilization` for expert-tier headroom. At startup per-layer tier capacity C = ceil(occupancy N / util). util=1.0 reproduces v1 exactly (cap == n, trade-only); util=0.9 pre-provisions ~11% spare upper-tier rows enabling displacement-free upgrades within a two-ledger check (per-layer capacity ledger + global byte ledger).

### Topology neutrality audit
**Commit:** `05006189`
- The policy domain is topology neutral and enforced (bans rank/world_size/tp/device; T6 proves 4 independent processes agree bit-for-bit). The artifact domain is TP4-frozen: "rank_sliced_tp4" is not just a naming convention — the four rank slices are four independent EXL3 quantizations whose H-side rotations measurably differ across ranks.

### N-tier feasibility — two-tier limit is a code constant
**Commit:** `22ab440b`
- Compiled a 3-tier subclass of `W4A16MixedTrellisKernel` through stock `b12x_compile` on Blackwell. A third tier (K2/K3/K4) costs +8 registers/thread (ceiling 255) and nothing else. The finding that reframes the question: N tiers would NOT buy the K2→K3→K4 ladder — a mixed layer's total bit budget is conserved regardless of tier count.

### Tensor-level FQ feasibility — per-slice re-tiering not worth it
**Commit:** `afc523eb`
- Measured on the project's own K2/K4/K5 encodes at slice granularity. Key findings: (1) TP-rank main effect on log-error is 0.0000 — the 4 rank slices are interchangeable, so 3,072 units/layer is really 768. (2) There is NO online signal below the expert: routing mass is per-expert by construction. (3) At equal bytes, per-projection allocation is worth +0.3% (nmse) to +14.0% (proxy) — but a global three-number projection choice captures most of the value.

### Routing-flatness investigation
**Commits:** `c02792d4`, `a4f2241d`, `454f0d48`
- Two measurements confirm GLM-5.2's routing is too flat for online top-K re-tiering: (1) 98.7% of experts active, top-26 mass share 0.366 (3.6× concentration over uniform); (2) top-K Jaccard first 0.542, best 0.616, floor 0.950 — desired K4 set churns ~39% between adjacent intervals. The within-domain churn is larger than the between-domain signal.
- FLAT-1: 15× the sample buys +0.014 Jaccard. Points away from noise and toward genuine near-ties: routing probabilities around the 26th-vs-27th boundary are so close there is nothing to separate.

### Routing-flatness retraction — instability half was a guard bug
**Commit:** `3fee0be8`
- 27 of 75 layers have `n_k4 = 0`, so their desired set is empty and cannot churn. `_jaccard` clamped the empty union to 1, scored them 0.0, and averaged them in — 0.562 reported where the 48 swappable layers actually agree at 0.879. The flatness measurements stand; the churn figures and "phase change invisible" claim need re-measuring with the fixed guard.

### Selection signal — routing frequency is the right ranking
**Commit:** `4362acb1`
- Hot experts quantize BEST in every layer (Spearman −0.779, 75/75 layers negative) — but contribution = frequency × error, and a 36× frequency lever beats a 1.4× error lever. Top-26 by contribution matches top-26 by routing 96%, top-26 by error 0%. Frequency ranking IS the theoretically correct signal.

### R10 — the allocator documented
**Commit:** `00624a14`
- Documents `brandonmusic/GLM-5.2-EXL3-TR3v4-3.5bpw-MTP78` reproducibility: 1,773 prompts, 1,049,589 tokens, 75 layers. Allocation: `gain(bits) = mass × (loss_by_bits[FLOOR] − loss_by_bits[bits])` solved as exact-budget DP knapsack (384 upgrade units/layer, floor 3 bits, candidates {3,4,5}). This is exactly the objective reconstructed independently in SELECTION-SIGNAL.md.

### Language contrast — the decisive experiment
**Commit:** `fd346be3`
- Paired design on one model state driven by English then Chinese prompts (972 majority-CJK prompts from m-a-p/COIG-CQIA, 208 requests per arm). Jaccard overlap: 0.879 for EN-vs-EN (control) but only 0.321 for EN-vs-ZH, against a 0.053 chance floor. Expert sets are STABLE within a corpus and highly DISTINCT across corpora — exactly the condition under which corpus-specific expert allocation pays off. Monotone depth gradient: deeper layers are more language-specific (layers 3–17: 0.373; layers 63–77: 0.256; Spearman(depth, jaccard) = −0.56).

---

## Tooling

### fq_repack — K3 quant → attested Progressive Tensors segments
**Commit:** `37074d5d`
- Repacks a per-layer-sharded EXL3 quant into Progressive Tensors segment files: per-expert contiguous (range-readable), ed25519-signed attestation, schemas `fq-segment/1`, `fq-attestation/1`, `fq-manifest/1`. Resumable per layer, optional incremental HF publish. 5 passing tests.
- **Header-digest signing** (`4649b21d`): synced with public repo so segment header digest is included in the signed attestation envelope. 194 tests pass.

### oci_unpack — runtime-free rootfs extraction
**Commit:** `607f9175`
- Unpacks an OCI-layout image to a rootfs dir without a container runtime: overlayfs whiteout semantics, multi-arch resolution, gzip/zstd layers, writes OCI config. 104 lines + tests.

### fq_assemble — policy + segments → bootable checkpoint shards
**Commits:** `8d7e2ae`, `a6ff61d8`, `fc29dfdd`, `28f5f3c3`
- Given per-layer per-K segment files and a `bits_per_expert` policy, emits per-layer shards in the original GG rank-sliced layout. All-K3 round trip BYTE-IDENTICAL to source (layer 30, sha `a5247345`). 9 tests green.
- **Mixed-K reindex path** (`a6ff61d8`): auto-detects when segment tensor sizes differ from source slots, switches to reindex path. Emits the metadata the GG exl3 loader requires for mixed checkpoints (`tier_bitmap.json`, `config.json` hybrid_tr3_tail).
- **--reflink mode** (`fc29dfdd`): expert-tensor regions written via `os.copy_file_range` so reflink-capable filesystems can share extents. Falls back per region to ordinary mmap read+write.
- **--force safety guard** (`28f5f3c3`): `--force` can no longer eat a directory it did not create. `check_out_dir()` refuses root, `$HOME`, git trees, upstream caches, and paths overlapping `--source`/`--segments`/`--policy`. Assembly happens in a sibling staging dir swapped into place with two renames.

### fq_verify — community-reassembly proof
**Commits:** `baf182f5`, `693df858`, `bd06ed89`
- Two modes: `--identity` (byte-level reconstruction proof: local stream-reassembly, remote ranged re-reads, derived re-derivation) and `--similarity` (dequantize across families, report cosine/Frobenius/max-diff). Results: 76/76 K3 shards sha256-identical (278.5 GB), 2048/2048 K4 spans byte-equal to fresh source reads.
- **Signer pinning** (`693df858`): rewritten to add `--trust-signer`/`--trust-root`/`--allow-unpinned-signer`/`--insecure-skip-signatures`. Fails closed when nothing is pinned and the trust root cannot vouch for the claimed key.

### fq_eps — ε curves, sensitivity variance, budget solve
**Commits:** `cf69178`, `14b753c6`
- Consumes encoder done-JSONs across Ks, produces eps curves, per-layer sensitivity variance (Gini/top-16 share), K2-abort check, and global greedy budget solve → `n_k4_per_layer`. 5 tests green.
- **Budget-conserving uniform baseline** (`14b753c6`): `uniform_baseline` now distributes total budget via largest-remainder (was discarding `budget % L` remainder, inflating the reported advantage).

### fq_probe — KLD ladder reference tool
**Commit:** `33a681a`
- Builds a fixed-seed held-out probe set (~32 prompts from the 4-axis corpus) and records per-token prompt logprobs from a running serve. Reference leg for KLD comparisons. Baseline captured: mean token logprob −1.7589.

### fq_prime — community-quant priming via HF ranged reads
**Commits:** `1353f371`, `a496f3e7`
- Extracts per-expert K3/K4 fragments from published GLM-5.2 mixed quants into `fq-segment/1` WITHOUT full downloads — ranged header fetch, trellis K identification, strict-adjacency range coalescing, paced+retrying transport, preallocated pwrite segment writer. 1027 lines + tests.
- **Spot-check hardening** (`a496f3e7`): `attested_expert_sha()` now raises `TrustError` when the attestation file is absent, when no trusted line names the fragment, or when no trusted line carries a digest for the sampled expert. Three-way equality `source == segment == attestation` enforced.

### fq_fetch, fq_release, fq_trust — consumer-facing tools mirrored from public repo
**Commits:** `69daf421`, `cc8ec3ad1`, `687467106`
- **fq_fetch** (1241 lines): recipe + ordered `--source` repos → coalesced HTTP Range reads of exactly the expert spans, verified per-expert against signed attestation, resumable. **Header authentication** (`cc8ec3ad1`): `--header-trust {auto,attested,full,unsafe}` proves each source's header before any byte is fetched.
- **fq_release** (269 lines): `fq-release/1` format, one signature over every file's digest. **Atomic publish mode** (`687467106`): single `create_commit` pinned to parent HEAD with bounded rebuild-and-retry.
- **fq_trust** (403 lines): `--trust-signer` pinning, four named trust rungs, rejects placeholder signatures.

### fruit-encoder — sha-pinned K2–K5 encode driver
**Commits:** `328e1cb`, `c277a89`, `0caf97e`, `4dcd6b3`
- `fruit_encode_driver.py`: imports the sha-pinned `encode_tr3_v31.py` encoder as a library and overrides its constants, self-spawns single-GPU workers, supports `--smoke/--oracle/--encode/--assemble` modes. K3/K4 smoke passed on SM120.
- `capture_fruit.py` / `capture_hf.py`: TP8/transformers-based BF16 activation capture. Fruit capture sealed: 10 layers × 1.05M tokens, 20.11 GiB, 0/10.5M routing mismatches.
- Cgroup guard credits reclaimable page cache. CUDA 13.2 toolkit pinned explicitly (image symlink escaped to host 12.6).

### capture_stream.py — layer-streaming BF16 capture
**Commit:** `0a9dcb63`
- 1866 lines. Layer-major streaming BF16 capture: meta-skeleton + per-layer materialization from safetensors shards, boundary activations rotated on disk, DSA full/shared-indexer top-k persisted across layers and windows. Enforces exact mode (batch=1; packed/grouped MoE fail the routing-id gate because sdpa/cublas/grouped_mm are not row-stable). 100.0000% routing-id row match + byte-identical x.bin vs sealed Fruit capture on all 10 layers.

### fq_reload — FqReloadWorker + fq_converge_layers RPC
**Commits:** `a53e54c6`, `381b231`, `31b410b`, `c6e32d0`
- `FqReloadWorker` extension (collective_rpc string surface) that re-runs the content half of the exl3 mixed prepare against a new same-cardinality policy checkpoint and `copy_`s tier slabs / combined rotations / maps / host metadata in place — every CUDA-graph-baked pointer stays valid. 659 lines + tests.
- **fq_converge_layers RPC** (`31b410b`): rebuild only the named layers at the tiers convergence asks for, sourced from segments through `FragmentResolver`. Returns layer→{expert: k_installed} so a partial climb is distinguishable from a full repay.
- **Install semantics fix** (`c6e32d0`): RPC returns `"installed": False` hard-coded with resolvable tiers under `"resolvable"` key, so a caller cannot mistake "this WOULD succeed" for "this happened."

### bt_metrics — battle-test log parser
**Commits:** `b0341bbf`, `12d3abd8`
- Extracts battle-test metrics from progressive-boot serve logs. Asserts on bytes-fetched (not wall-clock) and posture equality (per-layer bits_digests folded into one digest). Tests cover false-PASS failure modes.
- **Cache-hit counting** (`12d3abd8`): adds `segments_from_cache` field — distinct from `local (no fetch)` and `prefetched from`. 13 tests passing.

### verify_retier — proving swaps move weights
**Commits:** `6f58d910`, `f86d4cd0`
- Proves a forced re-tier actually moves weights on a live serve. Asserts on the difference between `GET /fq/layer/{L}` before and after — never on the POST response. Four checks: (1) requested expert changed tier, (2) some other expert displaced, (3) per-layer K4 cardinality unchanged (D1), (4) serve still generates coherent text.
- **Auto-demotion** (`f86d4cd0`): automatically picks the coldest K4 resident to displace, so the hand-driven test evicts the same expert the policy engine would.

### render_heatmap, lang_contrast, jaccard_trace
**Commits:** `7b265a75`, `3842207f`, `d0acf490`, `0d45f0c8`
- `render_heatmap.py`: decodes the `fq-heatmap/1` endpoint (base64-encoded, layer-major, little-endian, bf16 counts + u8 tier), renders activation heat beside tier map in SVG. Finding: 98.7% of experts see traffic, top-26 mass share 0.366.
- `lang_contrast.py`: paired language-switch test (math vs code = both English; Chinese script routes differently). 972 majority-CJK prompts from m-a-p/COIG-CQIA.
- `jaccard_trace.py`: plots desired-set Jaccard over intervals to settle the guard question.
- **Heatmap format** (`0d45f0c8`): benchmarked 7 encodings. With gzip, every format lands 31–39 KB. Keep base64-in-JSON for the live endpoint; use Parquet LONG form for archive.

### Score convergence — does runtime routing rediscover a human's quant?
**Commit:** `643a152e`
- Per-layer Jaccard scoring between the experts the loop would promote and the ones a human chose for GLM-5.2-EXL3-TR3-3.42bpw. Against analytic chance floor (0.267) and human-human ceiling (0.657). Explicitly does NOT rank by `expert_rel_rt_mse`. Known-answer validated: perfect 1.0000, random 0.2500, inverted 0.0000. 9 tests.

### make_charts — dependency-free SVG evidence charts
**Commit:** `eb8526404`
- Two stacked panels sharing one time axis (throughput and expert counts at different scales) rather than a dual-axis plot. No rasterizer on the box, so 12 structural tests stand in for visual review.

### make_axis_panels — 4-panel per-axis activation figure
**Commits:** `0be12aee`, `65681fc7`
- Flagship doc image: one heat panel per MTP78 corpus axis on ONE colour scale, plus pairwise overlap matrix. `log2(share/uniform)`, fixed domain -4..+4 in 9 bands, never auto-scaled. One per-layer column permutation threaded through every panel and asserted equal at draw time.
- JSON sidecar: 75×256 integer permutation matrix at one row per line (19,503 → 227 lines).

### heatmap.html — operator heatmap page
**Commit:** `b85ef7bb`
- Single self-contained HTML file (no build step, no CDN): 75×256 = 19,200-cell FQ activation matrix on canvas, K-tier strip, derived tier↔heat mismatch panel, diverging compare panel. CIELAB-interpolated LUTs. Consumes both `fq-heatmap/1` envelope and raw `VLLM_FQ_DUMP_STATS` record.

### reap-devices — kill by GPU device
**Commit:** `c1486a03`
- Frees specific GPUs by querying `nvidia-smi` for compute-app PIDs on those devices, killing them, waiting for memory to return. Replaces argv-based matching (which failed both ways: `pkill -f vllm` matched nothing; `pgrep -f 8200` matched too much and killed a healthy serve).

### prune-fragment-cache — tiered floor, absolute cutoff
**Commits:** `4543669`, `ce792f6`
- Evicts `fragments/` only, never `segments/` (whole-layer objects a hot restart slices locally instead of re-fetching ~230 GiB). Floor 200G→90G. With `KEEP_LAYERS=1`, pruning `fragments/` during a boot is safe (each expert resolved once, payload on GPU, later miss refetches).

---

## Campaign (0c — Encode Campaign)

### Multi-K campaign plan and execution
**Commits:** `0494ba9`, `1b27bbc`, `0545a6a`, `fd0f310c`, `53fe9767`
- Encode GLM-5.2 at K∈{2,3,4,5} from a single capture and publish segments. K3+K4 is the v1 loop, K2 unlocks the fast progressive-base story, K5 gives hottest-experts headroom. Execution order: hottest-first, per-layer durable value, backfill over time.
- Campaign revised for fixed 3TB: fully streaming rolling pipeline per layer-window (capture→encode→publish→delete), community-quant priming first. Peak transient footprint ~150 GB.
- Complement-encode matrix: community primes K3 fully + K4 hot set (~108/layer); idle GPUs encode K2/K5 + K4 complement (~148/layer). ≈660 expert-encodes/layer instead of 1024.

### 0c proxy leg — eps ladder
**Commits:** `c5f4afac`, `9039728d`
- Fruit proxy K3/K4: mean eps 0.0231→0.0060 (3.8× per added bit), K2-abort does NOT fire (benefit gini 0.48 — per-expert allocation premise holds). First solve-derived mixed policy minted at 0.42 budget.
- Full 4-point eps ladder: 0.0903 / 0.0231 / 0.0060 / 0.0016 — consistent 3.8× per bit across K2–K5.

### 0c campaign — window rings and rolling publish
**Commits:** `6b460cea`, `e66e9fe4`, `5c1edd4a`, `a8b18f83`, `4b7b1ec7`, `66a7ffc1`
- Window-1 encode ring: K2+K5 on real model layers 3-10, chained on capture seal.
- Rolling publish: repacks K2/K5 encoder outputs into fq-segments (encode-of lineage), uploads to HF.
- Window-2 ring: capture layers 11-18, K2 first (operator priority), then K5, auto-publish.
- K2-to-completion ring: 8-layer windows from layer 19 to 78, each doing capture→K2 encode→publish→prune.
- Robust Python publisher (`publish_window.py`): fresh temp staging per K, no stale-glob pollution.
- Layer extraction fix: `tr3` prefix collision in `grep -oE '[0-9]+'` replaced with `sed`.

### 0c — self-driving campaign supervisor
**Commits:** `21a771b1`, `69dcbc8b`, `0c835ce4`, `13b2b864`
- Fully autonomous loop: picks next (tier, window) from encoder done-JSONs, captures when needed, encodes on idle GPUs, publishes, prunes, repeats. Tier order K2→K5→K4-complement. Disk floor triggers publish+prune. Idempotent and self-healing.
- Single-owner PID lock + concurrent-encode guard (same-tier collision detection).
- HF credential self-loading: publisher self-loads `~/.fq_env` and fails fast on missing credentials (was hanging 57 min on interactive auth prompt). Attestations re-emitted as `encode-of`. Root `fq-manifest.json` rebuilt from remote inventory after upload.
- Per-window capture prune: each window's capture pruned immediately after its segments are published (was accumulating 240 GB). Capture is deterministic, so re-capturing yields byte-identical activations.

### 0c campaign — disk, concurrency, and GPU management
**Commits:** `21727aec6`, `ca2239496`, `a4bd59634`, `9505089f1`, `f84fdca30`, `50ccc77f`, `042eab83`, `762c721a`, `469fc7f0`, `b86ca6d`
- Prune deferral: skips while an encoder is reading the capture; matched on the window's own `--layers` argument rather than globally.
- Per-tier encode guard: only SAME-tier runs collide (same work dir, same done-JSONs). Different tiers may run concurrently on disjoint GPUs.
- GPU reservation file: `.reserved-gpus` reserves GPUs 0-3 for the M5 serve; campaign keeps 4-7.
- Segment regeneration guard: never regenerate segments the remote already has (one `list_repo_files` call up front).
- Disk guard: must WAIT on its window, not advance past it (a `continue` stepped to the NEXT window, silently skipping work).
- MoE layer range: layers 3-77, not 3-78 (layer 78 is MTP). `LAST=78` made the final window fail on every pass.
- Tier order pivot: K2→K4→K5 (K5 serving blocked on SM120 shared-memory limit).
- Disk reclaim after encode: captures dropped the moment encode exits 0, gated on every layer having a done-JSON.
- Clean pause: `pause-when-k4-done.sh` waits for 75 done-JSONs, lets in-flight encode finish, removes supervisor, does final publish pass.

### 0c campaign robustness — stale state and PARTIAL captures
**Commits:** `ac870de9`, `92eb99d9`
- Clear stale capture state when a window's data was pruned: a window with NO layer dirs has stale state; clear `state.json` and `work/`.
- Recover from PARTIAL captures left by preemption: any layer dir without sealed `x.bin` or carrying `*.partial` is removed and shard state cleared.

### Campaign state preservation through preemption
**Commit:** `25a07cdd`
- Box pre-empted mid-boot (demo2 at layer 34/76). Saved campaign state, BT timelines, and decision logs so the next instance starts from a recorded state.

---

## Milestones

### M0 — Seed, Assembly Verification, Mixed-K Boot
**Commits:** `65e6fb0`, `1bede26`, `f8d565e1`, `7d89f0f5`, `026202a1`
- HF model card for the segments repo: purpose, github/branch links, layout, attestation story.
- Assembly verification COMPLETE: 79/79 shards byte-identical at full-model scale. M0 offline gate closed by identity with the serving checkpoint.
- **M0 MIXED-K BOOT PASS** (`f8d565e1`): first mixed K3/K4 Progressive Tensors checkpoint boots under GG r33. Loader plans layer-dependent partitions exactly per the solve policy (42..152 K4/layer). Generation coherent, decode 501.6 tok/s vs pure-K3 503.1 (~0% mixing cost). Key finding: `B12X_MLA_SPARSE` requires `--kv-cache-dtype fp8_ds_mla`.
- HF card v2/v3: reassembly walkthrough, public tools repo link, campaign charts embedded.

### M1/M2 — Overhead and Dryrun Evidence
**Commit:** `ddc921f0`
- M1 collector overhead: 3.7–5.0% latency cost on the proxy (GATE <0.5% NOT met as measured; proxy amplifies fixed per-step cost ~150×). Must re-measure on GLM-5.2 at cc8.
- M2 dryrun: loop drove 208 decision intervals with visible guard behavior and 4-rank agreement.

### M3 — Live Reload-Under-Quiesce
**Commit:** `a53e54c6`
- **M3 BRUTAL APPLY PASS**: live reload-under-quiesce swaps the mixed-K allocation of a running serve in 0.47s, zero drops, bit-identical to fresh boot. RUNG A: restart-swap in 88.0 s downtime. RUNG B: live swaps at 0.466 s / 0.410 s total stall under continuous traffic (93/93 + 91/91 requests, 0 drops). Post-reload prompt logprobs bit-identical to fresh-boot baseline.

### M4 — Atomic Swap Engine
**Commits:** `9ff58e86`, `994b2357`, `a2642d17`
- **T3 PASS**: maps read as data under CUDA graph — map CONTENTS mutated in place, replay output `torch.equal` to fresh-built layer. APPLY_MODE=reload is NOT the ceiling.
- **T4 PASS x3**: one expert pair swapped end-to-end by `SwapEngine.apply()`; post-swap slabs/combined tables/maps `torch.equal` to fresh-built layer; rollback bitwise. Apply window 0.061/0.368 ms.
- **T5 PASS**: no torn state observable, 6/6 abort points. Host bookkeeping commits WITH the map flip. Opt-in fail-atomic staging restores pre-swap state inside quiesce window on pre-flip abort.
- **T6 PASS**: 4 simulated ranks in independent spawned interpreters reconstruct 50 chained intervals from one seed — 283 swaps, 50 distinct digests, all four ranks byte-identical.
- **M4 swap verified on a live serve** (`a2642d17`): layer 3 after re-tier shows `n_k4=26, |K4|=26, e0=K3, e1=K4` — read from `GET /fq/layer/3`, not from the POST response.

### M5 — Serve Evidence Campaign

#### M5 serve infrastructure
**Commits:** `77f1baf69`, `2f1189aa5`, `eb8526404`, `930d182d6`, `d33e554ac`
- Swap-evidence recorder: records `fq_*` gauges alongside vLLM throughput into one JSONL. Workload shifts prompt DOMAIN on a schedule. 9 tests.
- GLM-5.2 TP4 serve script (off/dryrun/live modes). Guards at spec defaults; only decision interval shortened.
- Dependency-free SVG evidence charts (two stacked panels, no dual-axis).
- Evidence-campaign orchestrator: readiness requires a real completion, not just `/health`.
- Default serve window 32k (GPQA traces exceed 8k; truncated traces score as WRONG).

#### M5 adversarial peer review + fixes
**Commits:** `bf09da4c`, `9b9f7f37`
- Peer review (978 lines) reproduced every finding, then fixed: `build_prompt` was seeded on salted `str.__hash__` (every process drew a different corpus); `phase_end` double-counted; NaN/Inf from `/metrics` blanked SVG series; token counter `startswith` matched `_created`/`_sum`/`_count`/`_bucket`; decode rate had no counter-reset guard.
- Four criticals fixed: empty `VLLM_FQ_TRUST_SIGNERS` turns trust filtering OFF; `run-evidence.sh` guarded on nonexistent script; cleanup killed only parent (orphaning 4 TP workers); missing `set -e`.

#### M5 boot gate and convergence
**Commits:** `6d92b81a`, `21a5d5c2`, `74da1895`, `bc581872a`, `620205ac`, `c0024b0f`, `1a19b693`
- **M0 boot gate PASSES on real GLM-5.2**: checkpoint assembled by `fq_assemble` boots TP4, serves: 76.14 GiB/rank in 401s, coherent generation, 219.2 tok/s at concurrency 8 with 208/208 requests OK.
- First convergence number: mean Jaccard 0.3603 vs 0.2652 chance, 0.6710 human-human ceiling (1.36× chance, 54% of human). Preliminary — wrong corpus, wrong signal.
- Corpus convergence 0.3938 (1.48× chance, 59% of human) — **RETRACTED** (`bc581872a`): driver replayed truncated stubs (scraped display mode, replayed 160-char stubs ≈ 74 tokens/prompt).
- **Corrected code-axis convergence** (`620205ac`): verified replay, 2,631,231 prompt tokens over 3,057 prompts (861/prompt). Code axis scored 0.4176 on 19 records, beating the long synthetic run's 0.3789 on 117 records.
- **Four-axis convergence** (`c0024b0f`): 12,228 prompts, 0 failures. Findings: (1) hit COUNT beats gate MASS on all four axes (+0.007 to +0.044); (2) code axis is NOT special (legal 0.4240, reasoning 0.4223, code 0.4210, general 0.3988); (3) volume nearly irrelevant (18× tokens, worse result). Best: 0.4240 = 1.60× chance, 63% of human-human.
- **Flagship 4-axis figure** (`1a19b693`): mean pairwise Jaccard 0.424 vs 0.265 chance (1.60×). Axes agree with each other about as much as each agrees with the human reference (~0.42).

#### M5 — GSM8K quality baseline
**Commit:** `b0088a18`
- flexible-extract 0.892 ± 0.0197 on 250-item seed-1234 subsample; strict 0.116 (measures format compliance, not arithmetic). Closes the last gap: every other number measured selection overlap, none measured whether the reassembled model is competent.

#### M5 — K5 mixed tier exceeds SM120 shared-memory limit
**Commit:** `bac9d884`
- Mixed K3/K5 checkpoint loads weights but all four TP workers die building kernels: 109,568 > 101,376 bytes for the K5 tier. K3+K4 fits with zero headroom. Mechanism: `compile_mixed_trellis` instantiates EVERY tier with `force_tile_config`, bypassing the fitting machinery. Selecting against `max(tier_bits)` would fix it.

#### M5 — gate mass binding
**Commit:** `a97edd6e`
- Live GLM-5.2 stats dump had `mass` byte-identical to `count` because the weights getter was never bound. Gate mass stays OPT-IN behind `VLLM_FQ_GATE_MASS=1` against the <0.5% decode gate. Downgrades to count-only and says so (`mass_is_real=false` in the dump) rather than lying.

#### M5 — missing-K hardening
**Commit:** `008d0e4c`
- A missing K bpw never crashes the serve: staging is host-only and strictly pre-quiesce, so a missing fragment can cost a pair or an interval but can never tear a layer. Six resolver paths that escaped `resolve()` as uncaught exceptions were hardened.

#### M5 — tensor-loader compatibility
**Commits:** `352097d2`, `dd93a460`
- FQ works with every `--load-format`. Progressive stream and `fragments.py` are owning producers; `swap.ResolverFragmentSource` copies before returning. Blocker is upstream: EXL3 quant methods retain every loaded tensor until `process_weights_after_loading`, and their two "copies" are identity operations on a tensor already contiguous and on device.

#### M5 — scenario-1 boot policy (observe mode)
**Commit:** `c70fd942`
- `observe` mode: zero budget, valid all-K3 policy, loop runs as pure observer recording routing. Convergence scored OFFLINE against the reference bitmap (which needs no K4 weights). `fq-policy/2` enforces occupancy == capacity, so "all-K3 with empty K4 slots waiting" is not expressible.

#### M5 — demo-1 progressive boot
**Commits:** `4eb7a8e4`, `03d2bb41`, `4545fa89`
- Boots GLM-5.2 directly from Progressive Tensors segments with `--load-format progressive`. Partial K4 coverage is not a blocker: a requested K with no fragment walks DOWN the ladder (`VLLM_FQ_K_FALLBACK=auto`), logs loudly, and enqueues on-the-fly encode. 5,126 K4 slots across 48 covered layers, +5.632 GiB/rank.
- Demo-1 runner: discovers K4 coverage from the resolver's own rules. Two warnings prevent fake results: 8 layers saturate their fragment pool, and those pools are a SUBSET of the reference's own K4 set (circular).
- Adversarial review of demo-1 glue: 5 defects fixed with tests (exit code, checkpoint gate, zero-budget layers, etc.).

#### M5 — first complete progressive load
**Commit:** `ef40561`
- A 355B-class MoE loaded into a live TP4 engine directly from per-expert segments, no assembled checkpoint on disk: 81.86 GiB / 3700.8 s, 76 layers, 0 degraded, 0 rejects, 160 local-first hits, 219 cross-rank shares, 194.5 GiB delivered. All four TP ranks independently resolved identical mixed-K digests. Engine refused to start with "Available KV cache memory: -3.1 GiB" — an arithmetic result, not a loader fault.

#### M5 — progressive boot robustness
**Commits:** `de8056d`, `69fbef7`, `c3cfe91`, `b1d9bea`
- Degrades instead of dying: `resolve_best()` (never raises) instead of `resolve()`, so a per-expert miss degrades down the K ladder. Ranged HF reads retry transient failures with backoff.
- Deploy before booting: stale rootfs has cost 3 boots. Added idempotent deploy step to `serve-demo1.sh`.
- Refuse to boot onto occupied GPUs: pre-flight guard checks only the devices this serve will use.
- Reap stale GPU holders by DEVICE (nvidia-smi `--query-compute-apps`), not by argv pattern.

#### M5 — transfer and prefetch tuning
**Commits:** `248b056`, `8929a5c`, `c5a7ae2`, `c6d2fb4`, `b3c1c9f`, `d5ab431`
- `HF_HUB_ENABLE_HF_TRANSFER` is a no-op on `huggingface_hub` 1.27 — the live knob is `HF_XET_HIGH_PERFORMANCE`. `VLLM_ENGINE_READY_TIMEOUT_S` raised for progressive boot. `VLLM_FQ_PREFETCH_DEPTH=3`.
- Hub downloads authenticated via `~/.fq_env` subshell. Boot banner reports presence only.
- Measured bulk-fetch throughput: 142–149 MiB/s (bulk) vs 0.75 MiB/s (per-expert ranged) — ~190×. JIT dominates a cold boot at ~9 min.
- Cold progressive boot costs: K2/K4/K5 local, K3 is NOT (no index). Cold cache fetches ~375 GB. Per-layer wall clock dominated by one object — layer 4 +3 s (cached) vs layer 5 +8m49s (fresh K3).
- `VLLM_FQ_KEEP_LAYERS` exposed and reported in boot banner. Default flipped to 1 (retain whole downloaded layers for runtime re-tiering).

#### M5 — layer-78 asymmetry
**Commit:** `225aaf8c`
- Loader's domain is 76 MoE layers but the decision domain is 75, because MTP layer 78's router is never bound by the stats collector. This single fact caused four separate failures. States the constraint once with a preventive rule: documents are written in the LOADER domain; a decision-domain component merges its rows in rather than rebuilding the document.

#### M5 — growth-supported design
**Commits:** `d0acf490`, `adc1f67c`
- Unpaired promotions blocked at two independent layers: (1) no promotion proposed because `VLLM_FQ_MEMORY_BUDGET` is unset so `budget_filter` never runs; (2) no slot to promote into — `exl3.py` prepares each tier at exactly its occupancy.
- Adopt A+B: reserve only what is needed to grow ONE expert to max K, one growth at a time. One K4 slot/layer × 48 mixed layers = 218 MB/rank. Symmetric shrink reserve: one K3 slot/layer = 164 MB/rank. Both directions ~382 MB/rank total — under 0.4% of the card.

#### M5 — Scenario 2: flat-K3 policy
**Commit:** `d4a8b642`
- New policy `policy-demo2-flatk3.json`: all 19,200 routed experts at K3. Target posture: every layer growing to `((3,200),(4,56))`. Flat K3 fits (76.16 GiB, KV +6.59) and leaves 4.59 GiB above 2 GiB KV floor = 13 K4 slots/layer, not 56. Step B (full 56) needs the K3 slab to shrink as K4 grows.

---

## Battle Tests

### BT-1 PASSES — first served token from Progressive Tensors segments
**Commit:** `aabda405`
- Attempt 14: engine boots from per-expert segments plus a bitrate policy (no assembled mixed checkpoint on disk). 79.08 GiB/rank, load 1616 s, peak fetch 543 MiB/s. Two blockers fixed: stale `n_k4_per_layer` counts caught by D1 invariant, and two `bt_metrics` parser bugs.

### BT-2 PASSES + allocator-residue retraction
**Commits:** `9e7f8392`, `c75a3bbb`, `55a9a695`
- Hot restart fetched 0.0 GiB vs 295.8 GiB cold, at identical posture digest. 84 segments from cache, 40 from local, 0 from Hub. Fungible loop initialised (75 MoE layers × 256 experts instrumented).
- **Retraction**: the ~3.92 GiB "allocator residue" claim is retracted — moving the reclaim hook to the correct point measured freed 0.00 GiB. The gap came from comparing two runs with different `max_model_len` and graph capture. Corrected overhead: 7.47 GiB (was 8.06, computed against nvidia-smi card total instead of `mem_get_info` device budget).
- Zero-residue confirmed 16 times over: attempts 15–18, four ranks each, all 79.39 → 79.39 GiB.

### BT-6 FAILS → PASSES
**Commits:** `b920d2b6`, `0cd63a75`
- **BT-6 FAILS** (`b920d2b6`): decision path works (64 swaps across 39 layers, 208 of 256 displacing experts with zero routing mass — thesis confirmed), but `build_from_env` never receives an `apply_fn`, so `_maybe_apply` returns False. This is a wiring gap, not a missing mechanism.
- **BT-6 PASSES** (`0cd63a75`): 64 swaps staged off-step then INSTALLED across all four ranks. Verified from surfaces other than the log line: policy sha changed with all four ranks agreeing; cardinality 2,658 → 2,658 (D1 held); generation coherent; tier map shows 14 experts moved across 5 sampled layers. Nine defects fixed between the decision engine and this.

### BT-6c PASSES — sustained installs
**Commits:** `bc7e92d8`, `67fd20eb`, `19eb7d15`
- Two install events, all four ranks each, adoption branch firing both times, ZERO invalid swaps across 21 intervals, no NameError, no failed intervals, no worker deaths. Verified against boot policy file: 256 experts moved across 44 layers (exactly 2 installs × 64 swaps × 2 experts/swap). K4 cardinality 2,658 → 2,658 preserved.
- Final counters over ~23 minutes: 5 install events, 0 invalid swaps, 0 failed intervals, 0 worker deaths. Committed policy diffed against boot policy file so the final posture is reproducible from the repo alone.

### Live serve calibration
**Commits:** `30820f3b`, `ba26f5f0`, `5e3a4712`
- Guard fix vindicated live: jaccard 0.61 → 0.931 (the empty-layer fix: 27 layers with `n_k4=0` whose desired set cannot churn were scored 0.0 and averaged in). M4-W redesign confirmed working: staging in background thread with `all_reduce(MIN)` readiness vote.
- `VLLM_FQ_JACCARD_FLOOR` calibrated to 0.80 from measurement (within-corpus 0.90–0.95, within-corpus 4k steps 0.879, warming up 0.79, across-language 0.321, chance 0.053). Passes normal operation, holds during warm-up, blocks genuine regime change with ~3× margin.
- `CUDA_MODULE_LOADING` override-respect fix: `export CUDA_MODULE_LOADING=${CUDA_MODULE_LOADING:-EAGER}` so an explicit value wins (was unconditionally EAGER, silently discarding a deliberate LAZY).

### Concurrent instance support
**Commit:** `c2f511c0`
- `FQ_TAG` and `FQ_DEVICES_ENV` so two vLLM instances run concurrently. `FQ_TAG` separates prometheus dir, artifacts, stats, encode queue, and JIT caches. Segment cache is deliberately SHARED (read-mostly; a second copy would cost 299 GB).

---

## Loader v2 — Progressive Loader

**Commits:** `edb5a9a2`, `14af3c1c`
- Progressive Loader v2 run report: two policy boots from the same segment store — per-layer tiers followed each policy exactly; 042 boot's greedy outputs token-identical to the assembled serve; decode at parity (495.6 / 492.8 vs 490.5 tok/s assembled). CPU byte-parity preflight: 123,915 tensors, 0 mismatches. 77 CPU tests green.
- Trust + lazy-encode design: configurable multi-repo sources `VLLM_FQ_SOURCES`, attestation-based trust filtering (`VLLM_FQ_TRUST_SIGNERS` / `VLLM_FQ_TRUST_PREDICATES`), K-fallback ladder + `EncodeQueue` drain CLI.

---

## Health & Monitoring

### Health sweep — iterative rewrites
**Commits:** `4ccb64147`, `1b0e650d6`, `972beb96d`, `df0079eb`, `8a9b163`, `e782fb6`, `43c337dc`, `3307555a`, `0742dbdf`, `573119e8`, `147fc577`
- Rewritten to track supervisor/encode-k2/encode-k5/capture/publish/m5-serve, report per-tier layer coverage, and diff each log's size against the previous sweep. Honest labels: idle/exited/quiet vs STALLED.
- Track newest m5 serve log (not hardcoded tag). Watch the log the serve actually writes; probe the real port (was globbing dead logs and probing `:8000` while serve was on `:8100`).
- Liveness-based sweep: `pgrep -fc` for each job pattern + fresh-tail report + health endpoint probe.
- Count distinct layers on one rank, not matched lines (summing across 4 TP ranks reported "248/76").
- Coverage denominator corrected: 75 layers (3-77), not 76 (layer 78 is MTP).
- Warm boot progress signal: falls back to counting resolved segments by origin (warm boot emits no "FQ downloads" line).
- Report BOTH running instances separately, with "apply bound on N ranks" and "intervals INSTALLED swaps" checks.
- Count installs by distinct timestamps, not by rank log line (fourfold inflation fix).
- Surface cgroup memory (`memory.current`/`memory.max`, anon, headroom, `oom_kill`) and per-worker RSS — `free -g` reports the HOST, not the cgroup limit that kills workers.

---

## Environment & Deployment

### gg-env — runtime environment setup
**Commits:** `4ab2e7a`, `c277a89`, `2976deed`
- Persist `TORCHINDUCTOR_CACHE_DIR` under `/home` (was ephemeral `/tmp`). Autotune cache audit: vllm/triton/b12x/flashinfer-jit all persistent; flashinfer lazy autotune identified as per-process tax on bf16 engines only.
- Pin CUDA 13.2 toolkit explicitly (image `/usr/local/cuda` symlink escapes to host 12.6). `nvcc` on PATH for DeepGEMM JIT.
- Deploy script: the serve runs the ROOTFS copy, not the source tree. `deploy-fq.sh` diffs, copies, clears stale `__pycache__`, verifies all 12 modules import inside the rootfs. Also deploys `entrypoints/serve/__init__.py` (without it, every `/fq/*` route is silently absent).

---

## Documentation

### HF model card — iterative corrections
**Commits:** `65e6fb0`, `7d89f0f5`, `026202a1`, `51bc231f`, `b29c0621`, `809c59e3`, `957d8e33`, `e037570cf`, `c1348f309`, `d75ab5e`
- HF model card for the segments repo: purpose, provenance, repo layout, attestation story.
- v2: reassembly walkthrough + public tools repo link. v3: campaign charts embedded.
- Corrected against evidence: title/intro names K2/K5 (not just K3); bootability scoped to proxy; "verified on all 79" specifies which 79; encode-of described as stack-scoped.
- All four tiers in the title (K4 arrives via community priming). Maturity split: assembly verified vs runtime experimental.
- Retired stale known-issues (manifest last-writer-wins fixed, K2/K5 mislabeling fixed). Assembly examples pinned to signer.
- HF card is the front door: per-recipe `--include` sets with disk measured from real inventory (all-K3 279 GB, fast-load K2 269 GB, hot-K5 298 GB, primed-K4 294 GB).
- Manifest MISMATCH that incremental publishing causes: named explicitly so users don't conclude artifacts were tampered with.
- Rewritten as version-controlled source for `malaiwah/GLM-5.2-EXL3-FQ-segments` README. Evidence-backed inventory snapshot, proven numbers, limitations up front.

### Build findings and spec corrections
**Commits:** `87961f3b`, `df4d7e6f`, `ea578fe2`, `46e51046`
- `14-build-findings.md` (497 lines): consolidates every fact a reader of the pre-build spec would otherwise get wrong, cross-referenced to run reports. Key corrections: capture-fn binding gated on `enable_return_routed_experts`, rotations are COPIES targeting COMBINED tables, measured numbers replacing spec estimates (2.5 s/expert not 7.5; ~13 GPU-h not ~41).
- Additive-only build notes appended to all 14 existing spec docs (00-13) so every original claim keeps its text but is corrected where the build proved it wrong.
- Normalized relative paths in build notes. `runs/README.md` rewritten as evidence index.

### Admin API spec
**Commits:** `47c3babe`, `9f9b2650`
- Implementation spec for `POST /fq/retier` and friends: `adjust_k` disambiguation, fixed cardinality enforced in four places, memory guard (UNIT = 1,179,648 B/rank/expert/bit), missing fragments reject atomically after whole-batch pre-flight.
- "As implemented" section: documents deviations from design (router lives in FQ package, not `vllm/entrypoints`; `adopt_policy`/`explain_forced` in `admin.py`).

### FQ heatmap spec
**Commits:** `585289af`, `358c559e`, `405e4b08`
- Design decision for live MoE expert heatmap: two panels sharing one layout (magnitude + K-tier strip) plus derived tier-vs-heat mismatch panel. CVD-validated palettes. `log2(share/uniform)`, fixed domain ±4.
- `GET /fq/heatmap` endpoint spec: base64 bf16 + gzip puts default sample at 32,785 B (25× smaller) at 0.389% worst-case relative error. Single-flight with TTL.
- Reset control reconciled with DESIGN.md: scope "heatmap" + include=cum meets every requirement better than "collector" scope.

### Progressive download path documentation
**Commit:** `44f82c5`
- Answers the four-level concurrency question with the log lines that produced each finding. Marks end-to-end wall-clock comparison TBM rather than estimating.

### Upstream issues filed
**Commit:** `5739cb6`
- **#282**: EXL3 retains every loader tensor until `process_weights_after_loading`; its two apparent copies are no-ops, so borrowed-buffer load formats can corrupt any EXL3 checkpoint.
- **#283**: Mixed-Trellis prefill block/tile is chosen by a GLM-5.2 tier-signature allowlist, never from `max(tier_bits)`; a K3+K5 split exceeds SM120's 101,376-byte opt-in limit.

### Handoff documentation
**Commits:** `21cb2749`, `7fe003c9`, `ccaf2d4e`, `67fd20eb`, `19eb7d15`
- `SESSION-STATE.md`: resumable snapshot before context compaction — live state, launch incantation, what is proven, three retractions, traps this box has sprung.
- `HANDOFF.md` (657 lines): vocabulary from scratch, every path absolute, two-instance GPU discipline, proven vs claimed, three retractions. Diagnoses concurrent-boot cgroup-OOM: `memory.max` = 1280 GiB, `oom_kill 2` — two workers killed by kernel with no traceback while `free -g` showed ~1 TB.
- Complete open-work register: all 39 items in full, grouped (in-flight, correctness+observability debt, memory+growth, headline demos, battle tests, measurement, publication, campaign). Preserves numbers that would die with the session (89.2% GSM8K baseline, K5 SM120 shared-memory figures, exact swap.py/admin.py line numbers).
- Final state: BT-6c PASSED, binding constraint moved from correctness to memory and IO. "START HERE" block names the single measurement that unblocks the most (malloc_trim on next boot).

---

## Rebase & PR

### Upstream overlap analysis
**Commits:** `264165fbb`, `b19c37020`, `39adf045c`
- Branch is 0 behind `dev/gilded-gnosis` (rebase is a no-op). `main` is a TRAP: 416 commits behind. PR #280 preserves the hybrid_tr3_tail path. Replaying our 11 commits onto #280 gives 0 conflicts.
- r34's vLLM base is `dev/gilded-gnosis@e2666d9a` (unchanged since 2026-08-07). r34 is an isolated composition of base + 21 open PR heads (no tags or releases). `malaiwah/vllm-voipmonitor` forks `vllm-project/vllm` directly, so GitHub pre-fills PR base as upstream `main` — must be retargeted by hand.

### PR submission materials
**Commits:** `72c08938`, `dace9cb5`
- Four documents under `runs/pr/` so a human can review and submit; PR deliberately NOT opened. `PR-BODY.md` leads with what this is (runtime re-tiering: stats → policy → atomic swap) rather than what it contains. Every number cites a file, including the ones that hurt (RETRACTED corpus convergence, count beating gate mass, K5 hardware-blocked, M1 decode cost).
- Test number pinned to committed tree: 493 passed, 10 skipped, 1 warning in 10.23s (re-ran in detached worktree at exactly the commit hash).

---

## Serving Baseline

### Full GLM-5.2 K3 boots and serves
**Commit:** `ca22437`
- First boot of full GLM-5.2 K3 on the box — without container runtime (JarvisAI managed container, namespaces disabled), using extracted r33 rootfs + `gg-run.sh`. Serves on GPUs 0–3 at 37.7 tok/s single-request MTP0, health+generation probes green. Added `health/sweep.sh`.

---

## On-Box Bootstrap

### Resumable downloads + runtime-free ghcr puller
**Commits:** `a10fa464`, `24eb1cbe`, `7eda019`
- Resumable download launchers for GLM-5.2 original (1.5 TB, 283 files verified), K3 quant (295 GB, 82 safetensors verified), GG OCI images. Each with `state.json` checkpointing.
- `ghcr_pull.py`: dependency-free (stdlib `urllib`) ghcr.io puller that fetches images to OCI layout dir without docker/podman, doing token exchange, skipping blobs already present with correct sha256.

### Pre-M4 checks
**Commits:** `87874c61`, `29655563`, `f591b88`, `90e188a7`
- Fresh-source drift check: GG HEAD == audited `e2666d9a`, exl3 HEAD == r33 pin, b12x +3 dense-gemm-only commits (zero overlap with moe/trellis/maps). Fresh b12x master contains `mixed_trellis.py` (1524 lines) — audit gap closed.
- 4 adversarial verdicts over `mixed_trellis.py`: maps PASS, launch PASS, rotations COPY, occupancy PASS.
- Occupancy < capacity GPU test PASS: 7/7 bitwise-equal with garbage in all unreferenced row classes + leakage control.
- Encode bench: ~2.5 s/expert (3× better than planned), K3==K4 cost, ~71 experts/h at 5% budget.

---

## K2/K5 Encode Cost Corrections

### K2 encode cost corrected in campaign ledger
**Commits:** `df0079eb`, `1b27bbc`
- K2 encode cost corrected in the 0c campaign ledger (part of health sweep rework commit).
- Disk expansion economics: growing `/home` from 3 TB to ~5 TB covers the ~4.2 TB working set, removing the storage-driven half-campaign split.

---

## Retractions

Three claims were retracted during the work and are recorded here so they are not re-asserted:

1. **Allocator residue** (`9e7f8392`): the ~3.92 GiB "allocator residue" claim is retracted. Moving the reclaim hook to the correct point measured freed 0.00 GiB. Corrected overhead: 7.47 GiB (was 8.06).
2. **Corpus convergence numbers** (`bc581872a`): the 0.3938 corpus convergence number is retracted. The replay driver scraped a human display mode and replayed 160-char truncated stubs (≈74 tokens/prompt). Corrected numbers in `620205ac`: code-axis 0.4176.
3. **Routing-flatness instability** (`3fee0be8`): the "routing is too unstable" half of the flatness finding is retracted. 27 of 75 layers with `n_k4=0` were scored 0.0 and averaged in — the 48 swappable layers actually agree at 0.879. The flatness measurements (98.7% active, top-26 = 36.6% mass) stand.

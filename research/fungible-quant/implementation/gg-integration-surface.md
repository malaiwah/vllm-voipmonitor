# GG vLLM Fork: Integration Surface for the Fungible-Quant Engine

Branch: `gg/dev/gilded-gnosis` @ `e2666d9a65f41fc376607531453cbd57c4c71016` (2026-08-07, "refactor: complete b12x rename").
All `file:line` cites below are against that ref (`git show gg/dev/gilded-gnosis:<path>`).
Primary file: `vllm/model_executor/layers/quantization/exl3.py` (2447 lines). The B12X kernel
package (`b12x.moe.*`) is **external** — imported lazily (exl3.py:201-247), not vendored in the tree
(only tests/tools reference it by name).

---

## 1. Full state inventory of a mixed-tier FusedMoE layer after startup

The module owning EXL3 MoE state is `RoutedExperts` (`vllm/model_executor/layers/fused_moe/routed_experts.py:93-102`
sets `self.layer_name`, `self.local_num_experts`), wrapped by `MoERunner`
(`vllm/model_executor/layers/fused_moe/runner/moe_runner.py:240`, holds `self.router` :279,
`self.routed_experts` :285, `layer_id` property :934). The quant method is `Exl3MoEMethod`
(exl3.py:1276).

### 1a. Scalar/config attrs stamped on the layer (module attrs, created in `create_weights`, exl3.py:1283-1377)

| Attr | Where set | Notes |
|---|---|---|
| `layer.exl3_tp_rank` / `exl3_tp_size` | exl3.py:1305-1306 | checkpoint TP must equal runtime TP (:1312-1317) |
| `layer.exl3_hidden_size` | exl3.py:1307 | H |
| `layer.exl3_intermediate_size_per_partition` | exl3.py:1308 | I_pp |
| `layer.exl3_params_dtype` | exl3.py:1309 | bf16/fp16 only (:1293-1296) |
| `layer.exl3_max_num_batched_tokens` | exl3.py:1342-1344 | from scheduler_config; hard error if absent (:1333-1341) |
| `layer.exl3_is_draft` | exl3.py:1354-1356 | authoritative stamp: `runner_type == "draft"`; read by `_is_draft_layer` (:91-112). NOTE: despite the docstring at :96-97, `load_eagle_model` (`vllm/v1/worker/gpu/spec_decode/eagle/utils.py:97`) does NOT set it on this ref — the create_weights stamp + name-substring fallback (:108-112) are what exist |
| `layer.exl3_layer_bitrates` | exl3.py:1357-1359 | per-expert bitrate tuple from the JSON (see §7) |
| `layer.exl3_mixed_bitrate` | exl3.py:1360 | `len(set(bitrates)) > 1` selects the mixed path |
| `layer.exl3_trellis_tile_config` | exl3.py:1701 (mixed) / :1816 (uniform) | from `_mixed_trellis_tile_config` (:1556-1570): (128,128,32,512) when H%512==0, else (128,128,64,256)/(128,128,128,128) |

### 1b. Registered parameters (loading-time only in mixed mode)

`create_weights` registers `w13_{suh,svh,trellis,mcg,mul1}` and `w2_{...}` as `Exl3MoEParameter`
(exl3.py:1361-1377; class :1177-1251). **Mixed-bitrate mode preallocates GPU slabs only for
`suh`/`svh`** (`preallocate=rank_sliced and suffix in {"suh","svh"}` when `exl3_mixed_bitrate`,
:1369-1375); trellis payloads stay as per-expert dict entries in `param.exl3_tensors`
(keyed `(expert_id, shard_id)`, :1200,1214-1216). Slab allocation on first load: :1228-1238;
row aliasing validated in `_rank_sliced_backing` (:1508-1531).

**All of these are freed after prepare**: `_prepare_mixed_rank_sliced_weights` ends with
`param.exl3_tensors.clear(); param.exl3_backing = None` for every `w13_*`/`w2_*` param
(exl3.py:1706-1710). After startup the registered params are empty husks — the checkpoint-form
per-expert tensors (trellis rows, suh/svh rows) **no longer exist on device**. A swap engine
must re-source them (from disk or a CPU-side retention cache it adds).

### 1c. Persistent per-layer device state: `layer.exl3_mixed_trellis` (module attr dict, exl3.py:1692-1700)

Built once in `_prepare_mixed_rank_sliced_weights` (exl3.py:1572-1715), called from
`process_weights_after_loading` (:1408-1410 → `_prepare_rank_sliced_weights` :1717-1720):

- `"tiers"`: tuple of 2 opaque B12X prepared-weight objects, one per tier, from
  `api.prepare_weights` = `b12x.moe._shared.kernels.w4a16.prepare.prepare_trellis256_moe_weights`
  (binding exl3.py:243; call :1665-1686 with `w13_layout="trellis3_t256_proj"`, `trellis_bits=bits`,
  `codebook="mcg"`, per-tier `gate_suh`/`up_suh`/`intermediate_rotations`/`down_svh`,
  `workspace=w13.view(torch.int32).reshape(-1)[:1]` :1684). Input tier slabs are stacked in
  **tier-membership order** (`expert_ids` ascending within tier): w13 stack :1617-1627 shape
  `(2, n_tier, H/16, I_pp/16, 16*bits)` (:1633-1639), w2 stack :1628-1632 shape
  `(n_tier, I_pp/16, H/16, 16*bits)` (:1640-1645). Tier row index = position of the expert in
  `tier_ids[t]` — this ordering is the membership contract the maps encode.
- `"tier_ids"`: tuple of 2 tuples of global expert ids (:1687, :1694) — host-side Python, the
  source of truth for membership.
- `"tier_bits"`: e.g. `(3, 4)` (:1695).
- `"global_to_combined"`, `"descriptor_map"`: device tensors from
  `api.build_tiered_maps(tier_ids[0], tier_ids[1], device)` (:1689-1691, stored :1696-1697) —
  passed as **runtime args** on every kernel launch (:1961-1962).
- `"rotations"`: `api.combine_trellis_rotations(*prepared_tiers)` (:1698) — combined
  suh/svh/rotation slab in tier-then-membership order.
- `"tile_config"` (:1699).

Per-tier rotation inputs consumed at prepare: `gate_suh`/`up_suh` = `index_select` rows of the
w13_suh slab (:1653-1655), `intermediate_rotations` = cat(gate_svh, up_svh, down_suh) per tier
(:1656-1663), `down_svh` (:1664).

### 1d. Process-global runtime cache: `_MIXED_TRELLIS_RUNTIMES` (module dict, exl3.py:74)

Created lazily on first eager forward by `_mixed_rank_sliced_runtime` (exl3.py:1828-1935). Key
(:1844-1856): `(_runtime_owner_token(quant_config, layer), device_index, x.dtype, topk_ids.dtype,
H, I_pp, tier_signature, topk, max_decode_m, max_batched_tokens, tile_config)` where
`tier_signature = tuple((bits, len(ids)) per tier)` (:1840-1843) and the owner token isolates
target vs draft (:115-121, scope id :124-153). **Fixed cardinality ⇒ `tier_signature` is
constant ⇒ membership swaps never miss this cache** — that is the load-bearing invariant of the
whole design. Compilation after capture begins is a hard error:
`"Mixed-bitrate EXL3 runtime must be compiled during the eager profile pass before CUDA graph capture"`
(:1860-1864). Value dict (:1916-1922): `api`, `decode`/`prefill` states, `max_decode_m`
(`VLLM_EXL3_TRELLIS_MAX_M`, default 32, clamped to max_batched_tokens :1835-1838),
`max_batched_tokens`. Each state (`make_state`, :1876-1908):
- `"launch"`: `api.compile_mixed_trellis(size_m, hidden_size, intermediate_size,
  tier0_num_experts, tier1_num_experts, tier0_bits, tier1_bits, top_k, max_m_blocks,
  moe_block_size=8, sms, max_shared_mem, force_tile_config, rotation_input_dtype,
  route_ids_dtype)` (:1882-1899). Depends only on tier **counts/bits**, never on membership.
- `"buffers"`: `api.make_mixed_trellis_buffers(launch, device, sms)` (:1903-1907) — mutable
  scratch; byte-counted for the log at :1924-1934.
Route-slot sizing via `api.max_packed_route_slots` with block size 8 (:76, :1877-1881).

### 1e. Per-forward read path and identity stability

`apply` (exl3.py:2342-2370) → `_apply_rank_sliced` (:2200-2213) → `_apply_mixed_rank_sliced`
(:1937-1967): re-reads `layer.exl3_mixed_trellis` each call (:1954) and passes
`(x, tiers[0], tiers[1], topk_weights, topk_ids, global_to_combined, descriptor_map, rotations,
launch, buffers)` to `api.run_mixed_trellis` (:1955-1966). Capacity guard :1946-1950; decode vs
prefill state pick by `m <= max_decode_m` (:1951-1953).

**Stability:** all tensors are allocated once at startup and reused every step; nothing on the
hot path reallocates. Under CUDA-graph capture the device pointers of tier weights, maps,
rotations, launch tables and buffers are baked into the graph, so a swap engine must mutate
**contents in place** (`copy_`) and must never replace these tensors with new allocations. The
Python dict indirection at :1954 only helps in eager/pre-capture execution.

Guards relevant to the engine: `expert_map` must be None (:2356-2357 — "EXL3 MoE expert
maps/EPLB are not supported"), EP hard-rejected in create_weights (:1297-1300 —
`NotImplementedError("EXL3 correctness MoE currently supports TP but not expert parallelism")`),
SiLU only (:2352-2355), no bias (:1301-1304), exactly two tiers (:1591-1595). So topk_ids reaching
`run_mixed_trellis` are **global logical expert ids** — membership maps are the only translation
layer, owned entirely by `exl3_mixed_trellis`.

(For contrast, the uniform-bitrate path keeps live slabs + pointer tables:
`layer.exl3_trellis_weights` :1787-1797, `layer.exl3_pointer_tables` :1810,
`layer.exl3_expert_map` = arange :1811-1815, runtime scratch dict in `_RANK_SLICED_RUNTIMES`
:2103-2177.)

## 2. Swap hook points

### (a) Writing new rows into tier slabs
The tier weight bits live inside the two opaque prepared-tier objects
(`layer.exl3_mixed_trellis["tiers"]`, produced at exl3.py:1665-1686). vLLM-side code never
touches their internals after prepare; the natural hook is a new method on `Exl3MoEMethod`
sitting next to `_prepare_mixed_rank_sliced_weights` (exl3.py:1572) that, for a swap
(expert_out, expert_in) at tier-row r: re-runs the per-expert half of the prepare pipeline
(B12X would need either a row-granular `prepare_weights` variant or an exposed
"repack one expert into slab row r" entry point) and `copy_`s the packed rows into the tier
object's storage in place. Source data problem: the checkpoint-form trellis/suh/svh tensors are
freed at :1706-1710 — the engine must retain CPU copies at prepare time (cheapest: hook just
before :1706) or reload from safetensors. Row order contract: tier row index == index in
`tier_ids[t]` (stacking loops :1617-1632).

### (b) Rebuilding global_to_combined / descriptor_map in place
Regenerate with `api.build_tiered_maps(new_tier0_ids, new_tier1_ids, device)` (same call as
exl3.py:1689-1691), then `old.copy_(new)` into the existing tensors stored at :1696-1697
(in-place, since their addresses are graph-baked as launch args :1961-1962), and update the
host-side `mixed["tier_ids"]` tuples (:1694) so `tier_signature` recomputation (:1840-1843) and
any future re-prepare see the new membership. Cardinality per tier must not change or the
signature — and therefore the compiled launch (:1885-1890) — is invalidated.

### (c) suh/svh/rotation rows
Combined into `mixed["rotations"]` by `api.combine_trellis_rotations(*prepared_tiers)`
(exl3.py:1698); per-tier inputs were built by `index_select` over the (now freed) suh/svh slabs
(:1653-1664). A swap must write the incoming expert's gate_suh/up_suh/gate_svh/up_svh/down_suh/
down_svh into the corresponding tier-row slice of the combined rotations tensor (and any
rotation copies held inside the prepared tier objects). Same retention requirement as (a).

### Locks / streams / async precedent
- `exl3.py` has **zero locking**: `_MIXED_TRELLIS_RUNTIMES` / `_RANK_SLICED_RUNTIMES` are plain
  module dicts (:73-74) mutated only during the eager profile pass; forwards run on the main
  compute stream. Any background writer must add its own synchronization.
- **No GG-specific async worker exists for EXL3.** The in-tree precedent is upstream's EPLB
  async worker, fully present on the branch: `vllm/distributed/eplb/async_worker.py` —
  `start_async_worker` :24-47 (daemon `threading.Thread` :45, dedicated
  `torch.cuda.Stream(device_index)` :35), `run_rebalance_experts` :50 (all work under
  `torch.cuda.stream(cuda_stream)`), `transfer_run_periodically` :76-129
  (`state.rearrange_event.wait(stream=...)` :82, `communicator.set_stream` :92, per-layer
  `transfer_layer` :129 from `rebalance_execute.py`). Handshake state in
  `vllm/distributed/eplb/eplb_state.py`: `is_async` :266, `rearrange_event` :272,
  `async_worker` thread :274, worker device idx :280, buffer_ready/layer_transferred flag
  semantics :182-208, main-thread gating :631-650, :744-750, :817, :860. This
  thread+stream+event pattern is the template for the fungible-quant background engine.
- EPLB cannot be reused directly (`layer.expert_map` rejected, exl3.py:2356-2357), but its
  double-buffer + per-layer quiesce choreography is the model to copy.

## 3. Quiesce integration

The GG fork **carries the full upstream pause/RPC surface**:
- Engine core: `EngineCore.pause_scheduler(mode: "abort"|"wait"|"keep", clear_cache=True)`
  `vllm/v1/engine/core.py:850-880`; `resume_scheduler` :881-884; `is_scheduler_paused` :885-888;
  `sleep(level, mode)` :889-926 (level 0 = pause only; 1+ delegates to executor);
  `wake_up` :927+; proc-level `pause_scheduler` :1793; `collective_rpc` :974-981.
- AsyncLLM: `pause_generation` (deprecated alias) → `engine_core.pause_scheduler_async`
  `vllm/v1/engine/async_llm.py:786`; `resume_generation` :795-798; `is_paused` :799+;
  `collective_rpc` :964-974; weight-update RPCs :1070-1095. Client plumbing:
  `vllm/v1/engine/core_client.py:1130-1133` (`pause_scheduler_async`), :935-942, :1188-1196
  (collective_rpc sync/async). Abstract: `vllm/engine/protocol.py:227`.
- HTTP endpoints (dev router): `POST /pause`, `POST /resume`, `GET /is_paused` —
  `vllm/entrypoints/serve/dev/rlhf/api_router.py:29-154`; `POST /collective_rpc` —
  `vllm/entrypoints/serve/dev/rpc/api_router.py:23-43`; `POST /sleep`, `POST /wake_up` —
  `vllm/entrypoints/serve/dev/sleep/api_router.py:21-39`; attached in
  `vllm/entrypoints/serve/__init__.py:59-61`.
- Worker-side RPC targets reachable via collective_rpc: `Worker.reload_weights`
  `vllm/v1/worker/gpu_worker.py:470-471`; chunked weight-transfer session
  `start_weight_update`/`start_draft_weight_update`/`update_weights`/`finish_weight_update`
  :1277-1356. `LLM.collective_rpc` `vllm/entrypoints/llm.py:560-590`, weight-update wrappers
  :861-896.
- Sleep/CuMem: present but abstracted behind `SleepModeBackend`
  (`vllm/device_allocator/sleep_mode_backend.py`: ABC :37, `CuMemBackend` :109 wrapping
  `CuMemAllocator` — `vllm/device_allocator/cumem.py:82`, singleton :107-119 —, factory :142/:176).
  Worker wiring: gpu_worker.py:112, :198-209 (lazy backend), `sleep()`→`suspend` :227,
  `wake_up()`→`resume` :247, gated by `model_config.enable_sleep_mode` :277. **Nothing in the
  EXL3/GG serving path requires sleep mode; it is orthogonal and off unless
  `enable_sleep_mode` is set.** For fungible-quant, `pause_scheduler(mode="keep"|"abort")` +
  a worker-side collective_rpc method (e.g. `apply_fungible_swap`) is the intended quiesce
  choreography; a fully in-stream swap (EPLB-style, no pause) is the more ambitious alternative.

## 4. Startup plan / caching precedent

`vllm/v1/worker/startup_plan.py` (191 lines) — the pattern to imitate for persisting the
fungible policy + repacked-slab cache:
- Schema/version: `PLAN_SCHEMA_VERSION = 1` :38; plan rejected on schema or fingerprint
  mismatch (`_load_plan` :87-103).
- Fingerprint: `compute_plan_fingerprint(vllm_config, rank, world_size)` :41-74 — sha256 of
  `{schema, vllm version, vllm_config.compute_hash(), device_name, device_total_memory,
  device_capability, torch.__version__, torch.version.cuda, rank, world_size}`, truncated to
  16 hex chars :73-74.
- Location: `{VLLM_CACHE_ROOT}/startup_plan/startup_plan_{fingerprint}.json` (`_plan_path`
  :77-84) — "regenerable derived state, alongside the torch.compile cache".
- Apply-time safety gate: recorded value used only if current free memory ≥ recorded baseline
  (`_applicable_kv_cache_memory_bytes` :106-131); any mismatch falls back to full profiling.
- Rehydrate: `maybe_apply_startup_plan(worker)` :134-164, called at the top of
  `determine_available_memory` — gpu_worker.py:513. Persist: `maybe_save_startup_plan`
  :167-191 (atomic `os.replace` of a `.tmp.{pid}` file :185-188; failures logged, never
  raised), called at gpu_worker.py:920.
- Gate env: `VLLM_ENABLE_STARTUP_PLAN` — envs.py:289 (typed stub), :2032-2033 (registration),
  and listed in `compile_factors().ignored_factors` :2389 with the comment "Runtime memory-plan
  persistence; does not affect compiled graphs" — the exact precedent for keeping a
  `VLLM_EXL3_FUNGIBLE*` knob out of the torch.compile cache key.
A fungible-quant plan file would fingerprint additionally on: bitrate-JSON content hash,
tier_signature per layer, tile_config, B12X version — and store the policy (per-layer
tier_ids) plus optionally packed tier-slab blobs for fast rehydrate.

## 5. Stats hook (router capture)

- `FusedMoERouter` ABC: `vllm/model_executor/layers/fused_moe/router/fused_moe_router.py:12`;
  abstract `set_capture_fn(capture_fn: Callable[[torch.Tensor], None] | None)` :23-27; public
  `select_experts` :44+ calls `_select_experts` then writes the int16 routing-replay buffer if
  bound (:67-81).
- `BaseRouter` (`vllm/model_executor/layers/fused_moe/router/base_router.py:159`):
  `self.capture_fn` init :183; `set_capture_fn` :185-187 (exact signature:
  `def set_capture_fn(self, capture_fn: Callable[[torch.Tensor], None] | None) -> None`).
  Call-site order in `_select_experts` (:288-305): (1) `_validate_eplb_state` :288, (2)
  `_compute_routing` :291, (3) **`self.capture_fn(topk_ids)` :296-297 — logical ids, BEFORE
  EPLB mapping**, (4) `_apply_eplb_mapping` :300 (:204-213), (5) dtype convert :303. No GG
  modification vs upstream semantics; GG's `_select_experts` runs from `MoERunner.forward`'s
  modular branch (moe_runner.py:610-624 — router first, then
  `routed_experts.forward_modular` → `Exl3MoEMethod.apply`). This is exactly where per-layer
  expert-frequency stats for the fungible engine come from for free.
- Binding: `GPUModelRunner.init_routed_experts_capturer` gpu_model_runner.py:7864-7904
  (device buffer sized `max_num_batched_tokens`; pinned CPU mirror :7876-7884);
  `_bind_routed_experts_capturer` :7906-7919 iterates `self.model.modules()`, matches
  `isinstance(module, MoERunner) and isinstance(module.router, BaseRouter)`, closes over
  `module.layer_id` (property moe_runner.py:934) and calls
  `module.router.set_capture_fn(_capture_fn)` :7919. Consumers: buffer read :3765/:4823,
  clear :4188, replay wiring :4812-4823. Capturer:
  `vllm/model_executor/layers/fused_moe/routed_experts_capturer.py` — `RoutedExpertsCapturer`
  :58 (`capture` :110, `clear_buffer` :207, `get_device_buffer` :214) and
  `RoutedExpertsManager` :223 (KV-slot-indexed store; `store_batch` :298, `get` :307).
  Tests: `tests/model_executor/test_routed_experts_capture.py:76,102`.
  **Caveat:** only one capture_fn slot per router — the fungible stats hook must either chain
  with the capturer's fn or reuse `RoutedExpertsCapturer`'s buffer.
- **GLM-5.2 on this branch** is `GlmMoeDsaForCausalLM`
  (`vllm/model_executor/models/registry.py:116` → `deepseek_v2.py`; class is a bare subclass of
  `DeepseekV2ForCausalLM`, deepseek_v2.py:2184-2185; HF model_type `glm_moe_dsa`, layer-type
  patch `deepseek_sparse_attention` in `vllm/transformers_utils/config.py:139-144`). Its MoE
  gate is `DeepseekV2MoE.gate` = `GateLinear` (deepseek_v2.py:310-321, fp32-capable router
  dtype :123-131/:309) feeding `FusedMoE(..., use_grouped_topk=True, num_expert_group,
  topk_group, scoring_func, e_score_correction_bias=...)` :364-388; forward computes
  `router_logits, _ = self.gate(hidden_states)` :417. Router selection:
  `create_fused_moe_router` (`router/router_factory.py:39`, priority list :66-75) →
  **`GroupedTopKRouter`** (`router/grouped_topk_router.py:246`, a `BaseRouter`, so
  set_capture_fn works). (The glm4_moe.py family — Glm4MoeForCausalLM registry.py:114 —
  is the non-DSA sibling, same FusedMoE/grouped-topk shape, glm4_moe.py:192-227.)
- **MTP layer-78 path**: spec model `DeepSeekMTPModel` → `deepseek_mtp.py` `DeepSeekMTP`
  (registry.py:630); `mtp_start_layer_idx = config.num_hidden_layers` (deepseek_mtp.py:487,
  also :149) — for GLM-5.2 num_hidden_layers=78, so the MTP head is `model.layers.78.*`,
  named exactly like a target layer (why name-inference fails; exl3.py:96-103). Its bitrate-map
  entry uses the `tail_tr3` fallback → all-K3 vector (exl3.py:540-543), i.e. the draft layer is
  uniform-K3 and runs the uniform path, isolated from the target by
  `_runtime_owner_token` (exl3.py:115-121) + draft stamp (:1354-1356). Draft/target CUDA-graph
  parity test: `tests/quantization/test_exl3.py:484-533`.

## 6. Config / env-knob surface

Registration pattern (all in `vllm/envs.py`):
1. Passthrough lambda in `environment_variables` with a group comment — the EXL3 block at
   envs.py:2281-2294: `"VLLM_EXL3_TRELLIS_MIN_M": lambda: os.getenv("VLLM_EXL3_TRELLIS_MIN_M")`
   etc. (MIN_M :2286, MAX_M :2287, BLOCK_M :2288, PREFILL_CHUNK :2289, PREFILL_TRELLIS :2290,
   PREFILL_BLOCK_M :2291, EXT_PATH :2293, ABI_SHIM :2294). The comment (:2281-2285) states the
   contract: registered *only* so `validate_environ` (envs.py:2363-2369 — warns/hard-fails on
   unknown `VLLM_*`) doesn't flag them; consuming code applies its own defaults.
2. Consumption is direct `os.environ` in exl3.py via `_positive_env_int(name, default)`
   (exl3.py:267-284 — blank-string = unset, positive-int validation) and plain
   `os.environ.get` (:172,:176,:1993). Reads happen at runtime-plan time (:1835,:1987-1994),
   not import time.
3. torch.compile cache-key interaction: `compile_factors()` (envs.py:2372+) hashes **every**
   registered var except `ignored_factors` (:2380-…, filter :2457). The VLLM_EXL3_* knobs are
   NOT in ignored_factors, so they participate in the compile hash; `VLLM_ENABLE_STARTUP_PLAN`
   IS ignored (:2389). For new knobs — `VLLM_EXL3_FUNGIBLE` (enable),
   `VLLM_EXL3_FUNGIBLE_INTERVAL`, `VLLM_EXL3_FUNGIBLE_MAX_SWAPS` (caps) — imitate the block at
   :2281-2294 and add them to `ignored_factors` (runtime behavior, not compiled-graph shape),
   citing the startup-plan precedent comment style.
4. Typed attribute stubs at the top of envs.py (cf. `VLLM_ENABLE_STARTUP_PLAN: bool = False`
   :289) are optional for passthrough-str knobs — the EXL3 block skips them.

## 7. The bitrate JSON loader

Trigger chain: `hf_config.hybrid_tr3_tail` dict with `format == "exl3-trellis"`
(`_RANK_SLICED_FORMAT` exl3.py:156; override hook :389-402; `maybe_update_config` :404-424) →
`_configure_rank_sliced` (:452-508) validates required metadata keys
`{bits, codebook, experts_per_layer, moe_layers, tensor_schema, tp}` (:452-460), `codebook ==
"mcg"` (:466-470), `moe_layers == [first, last]` (:471-478), exact tensor_schema
`model.layers.{L}.mlp.experts.{E}.{proj}.rank{r}.{trellis|suh|svh|mcg}` (:479-486). Mixed mode:
`bits == "mixed"` + `k_values ⊆ {3,4,5,6}` (:489-497) + `bits_per_expert` must be a string
(:498-501), stored as `rank_sliced_k_values` (:503).

`_load_rank_sliced_bitrates(model_name, revision)` (exl3.py:510-557):
- Reference syntax `"file.json:field"` (rsplit on `:`; :517-524).
- Fetch: `get_hf_file_to_dict(filename, model_name, revision)` (:525) —
  `vllm.transformers_utils.repo_utils`, i.e. HF-hub/local-dir **read-only** resolution.
- Schema of the payload: top-level dict keyed by **string layer index** for every layer in
  `[first, last]` (:533-538); each entry is a dict whose `field` key holds a list of exactly
  `experts_per_layer` ints (:539-548); special fallback: if `field` missing and
  `len(entry["tail_tr3"]) == experts`, synthesize `[3]*experts` (:540-543 — the GLM-5.2 MTP
  overlay); every value must be in the declared `k_values` (:549-555). Result:
  `self.rank_sliced_bits_by_layer: dict[int, tuple[int,...]]` (:556-557, init :361).
- Per-layer read: `rank_sliced_layer_bitrates(layer_name)` (:559-576) regexes
  `layers.(\d+)` and indexes the dict; stamped once onto the layer at create_weights
  (:1357-1360).

**(a) Runtime reload**: everything downstream of the JSON is burned in at startup —
`layer.exl3_layer_bitrates` (:1357), tier partition (:1583-1595), slabs/maps (:1615-1700).
A reload path needs: re-run `_load_rank_sliced_bitrates` (needs `model_name`+`revision`, which
the config currently receives only inside `maybe_update_config` :420-423 — persist them on the
config), diff `rank_sliced_bits_by_layer` per layer, verify per-layer multiset of bits is
unchanged (fixed cardinality; otherwise tier_signature/launch invalidated), then drive hooks
§2(a-c) per changed layer. **(b) Write-back**: no writer exists anywhere on the branch;
`get_hf_file_to_dict` is read-only. The natural write-back is a sidecar under
`VLLM_CACHE_ROOT` following the startup_plan pattern (§4 — atomic tmp+`os.replace`,
fingerprint keyed) rather than mutating the checkpoint repo; emit the same
`{"<layer>": {"<field>": [k,...]}}` schema so the file can be promoted to a checkpoint
`bits_per_expert` reference verbatim.

---

### Cross-cutting constraints recap
- Fixed per-layer tier cardinality is what keeps `tier_signature`, the compiled launch
  (exl3.py:1885-1899) and the CUDA graphs valid across swaps; membership is the only degree of
  freedom (maps + slab rows + rotation rows, all in-place).
- No EP (:1297-1300), no expert_map/EPLB on the EXL3 path (:2356-2357), TP fixed to checkpoint
  TP (:1312-1317) — the swap engine is purely per-rank-local, one slab set per TP rank.
- Checkpoint-form tensors are freed post-prepare (:1706-1710); the engine must add retention.
- Quiesce exists and is HTTP-reachable (`/pause` + `/collective_rpc`); sleep mode is available
  but unnecessary for swaps.

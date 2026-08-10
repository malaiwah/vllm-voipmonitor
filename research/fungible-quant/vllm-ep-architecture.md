Now I have everything needed. Here is the dense, code-grounded reference report.

---

# Expert Parallelism (EP) in vLLM — Code-Grounded Reference

Repo state: `/home/user/vllm-voipmonitor`, HEAD `99267c23ca51` (2026-03-18).

## 1. Config surface

**`vllm/config/parallel.py`**
- `ParallelConfig.enable_expert_parallel: bool = False` — parallel.py:139-140. Docstring: *"Use expert parallelism instead of tensor parallelism for MoE layers."*
- `ParallelConfig.enable_ep_weight_filter: bool = False` — parallel.py:141-147 (skip non-local expert weights on disk read when EP active; no effect on 3D fused-expert checkpoints like GPT-OSS).
- `ParallelConfig.all2all_backend: All2AllBackend = "allgather_reducescatter"` — parallel.py:161-171 (default backend, doc-string enumerates all options).
- `All2AllBackend` literal — parallel.py:40-51: `naive, pplx, deepep_high_throughput, deepep_low_latency, mori, nixl_ep, allgather_reducescatter, flashinfer_all2allv (alias), flashinfer_nvlink_two_sided, flashinfer_nvlink_one_sided`. Note: `pplx` is in the literal but is dead — validator at parallel.py:370-375 warns and silently forces it to `allgather_reducescatter` (**pplx backend has been removed**).
- `EPLBConfig` — parallel.py:54-94 (`window_size`, `step_interval`, `num_redundant_experts`, `log_balancedness[_interval]`, `use_async`, `policy`), plus `ParallelConfig.enable_eplb`, `eplb_config`, `expert_placement_strategy: "linear"|"round_robin"` — parallel.py:148-160.
- **EP size derivation**: EP size is *not* a standalone knob — it is derived. `ParallelConfig` has no `expert_parallel_size` field; EP size is computed at the `FusedMoEParallelConfig` / process-group layer as `TP_size × DP_size` (× PCP_size if used). This is explicit in `docs/serving/expert_parallel_deployment.md:35-39`: `EP_SIZE = TP_SIZE × DP_SIZE`.
- Validators of interest: `enable_eplb` requires `enable_expert_parallel=True` and `TP*DP > 1` (parallel.py:388-401); `num_redundant_experts` must be 0 unless EPLB enabled (403-409); `enable_elastic_ep` requires `enable_eplb=True`, forbids `pipeline_parallel_size>1`, forbids `data_parallel_external_lb`/`data_parallel_hybrid_lb` (parallel.py:699-712); `all2all_backend in ("allgather_reducescatter","naive")` forces `eplb_config.use_async=False` with a warning (820-829, "Async EPLB causes hangs with the ... all2all backend").
- `ParallelConfig.use_sequence_parallel_moe` property (parallel.py:585-600): true when `all2all_backend` is one of the "sequence-parallel-friendly" backends (`allgather_reducescatter, naive, deepep_high_throughput, deepep_low_latency, mori, nixl_ep`) **and** `enable_expert_parallel` **and** `TP>1` **and** `DP>1`. Rationale documented at parallel.py:577-584: attention's `o_proj` all-reduce replicates tokens across the TP group; with EP + DeepEP-style all2all, feeding replicated tokens into dispatch wastes compute/comm, so MoE input is made sequence-parallel instead.

**`vllm/engine/arg_utils.py`**
- CLI-mirroring fields: `enable_expert_parallel` (421), `enable_ep_weight_filter` (422), `all2all_backend` (424).
- `--enable-expert-parallel` / `-ep` argparse registration — arg_utils.py:901-905; `--enable-ep-weight-filter` — 906-909; `--all2all-backend` — 910-912.
- Threaded into `ParallelConfig(...)` construction: `is_moe_model=model_config.is_moe` (1737), `enable_expert_parallel=self.enable_expert_parallel` (1738), `enable_ep_weight_filter` (1739), `all2all_backend=self.all2all_backend` (1740), `enable_elastic_ep` (1741), plus DBO/EPLB knobs (1742-1749).

## 2. Process groups (`vllm/distributed/parallel_state.py`)

`initialize_model_parallel()` (parallel_state.py:1478 onward) lays the global rank space out as a 5‑D tensor:
```python
all_ranks = torch.arange(world_size).reshape(
    -1, data_parallel_size, pipeline_model_parallel_size,
    prefill_context_model_parallel_size, tensor_model_parallel_size)
# layout: ExternalDP x DP x PP x PCP x TP   (parallel_state.py:1538-1553)
```
- **TP group**: `all_ranks.view(-1, tp_size)` — 1558.
- **DCP group**: carved out of the TP axis by reshaping the whole tensor to chunks of `decode_context_parallel_size` (1579); doc-note at 1572-1578 says DCP "reuses the GPUs of TP group," `tp_size` must be divisible by `dcp_size`.
- **PCP group**: transpose(3,4) then reshape (1596-1601).
- **PP group**: transpose(2,4) then reshape (1616-1619).
- **DP group**: transpose(1,4) then reshape to `data_parallel_size` (1631-1650); can be a stateless (elastic) group via `_init_stateless_group` if `enable_elastic_ep`.
- **EP group** (parallel_state.py:1652-1714):
  ```python
  if config.model_config is None or config.model_config.is_moe:
      group_ranks = all_ranks.transpose(1, 2).reshape(
          -1, data_parallel_size * prefill_context_model_parallel_size * tensor_model_parallel_size
      ).unbind(0)
      ...
      _EP = init_model_parallel_group(group_ranks, ..., group_name="ep")
  ```
  So **EP group size = DP × PCP × TP**, held fixed within a PP stage and within an "ExternalDP" (outer, non-model DP used e.g. by verl integration) group. **The EP group is only created for MoE models** (`config.model_config.is_moe` check, line 1655) — dense models get `_EP = None`.
  - An **EPLB group** with the *same membership as EP* but a separate process group/communicator is created immediately after, only if `enable_eplb` (1684-1713), explicitly to avoid deadlocks between MoE-forward collectives and EPLB rebalancing collectives.
  - `enable_elastic_ep` routes both EP and EPLB group creation through `_init_stateless_group` using ports pre-allocated by `ParallelConfig.allocate_elastic_ep_ports()` (parallel.py:468-527) instead of the normal `init_model_parallel_group`.
- Accessors: `get_ep_group()` (1251-1257) asserts `_EP is not None` with message *"EP group is only created for MoE models with num_experts > 0."* Related: `get_tp_group()` (1216), `get_dcp_group()` (1224, aliased as `get_context_model_parallel_group`), `get_pp_group()` (1235), `get_dp_group()` (1243), `get_pcp_group()` (1275).
- Teardown: `destroy_ep_group`-equivalent happens inside a shared teardown at 1885-1888 (`_EP.destroy(); _EP = None`), and a group-swap helper at 1192-1210 destroys/replaces `_DP/_EP/_WORLD/_EPLB` together (used for elastic EP scale up/down).
- Relationship summary: **EP ⊆ (DP × PCP × TP) within one PP stage**; **DCP is a sub-partition of the TP axis** (orthogonal concern — attention KV/context splitting — not folded into the EP reshape at all); **PP is a completely separate axis**, so EP groups exist per-PP-stage.

## 3. MoE dispatch under EP vs TP — `vllm/model_executor/layers/fused_moe/`

**`config.py` — `FusedMoEParallelConfig`** (config.py:924-1148):
```python
use_ep = (dp_size_ * pcp_size_ * tp_size_ > 1) and vllm_parallel_config.enable_expert_parallel   # config.py:1083-1086
...
if not use_ep:
    # TP path: tp_size, tp_rank flattened across DP×PCP×TP; ep_size=1, ep_rank=0
    tp_size, tp_rank = flatten_tp_across_dp_and_pcp(...)   # 991-1000, 1092-1110
else:
    # EP path: tp_size forced to 1 (no intra-expert sharding); ep_size = flattened tp_size, ep_rank = flattened tp_rank
    ep_size = tp_size; ep_rank = tp_rank                    # 1114-1130
```
The docstring (1022-1081) gives a canonical table, e.g. TP=2,DP=2,EP=True → `TP={1,0} DP={2,dp_rank} EP={4,ep_rank}` — i.e., with EP on, tensor-parallel sharding of expert weights is disabled and replaced by an EP group that spans DP×TP.

- `use_all2all_kernels` property = `dp_size > 1 and use_ep` (config.py:944-946) — **the all2all dispatch/combine kernels are only engaged when both EP and DP>1**; pure EP with DP=1 falls back to a communication-free "no-op dispatch" path (see below).
- Backend gating properties, all built on `use_all2all_kernels`: `use_deepep_ht_kernels` (948-953), `use_deepep_ll_kernels` (955-957), `use_fi_nvl_two_sided_kernels` (959-964), `use_fi_nvl_one_sided_kernels` (966-971), `use_naive_all2all_kernels` (977-981, covers both `naive` and `allgather_reducescatter`), `use_mori_kernels` (983-985), `use_nixl_ep_kernels` (987-989), `use_batched_activation_format` = `use_deepep_ll_kernels` (973-975).

**`layer.py` — `FusedMoE`**:
- `determine_expert_map(ep_size, ep_rank, global_num_experts, expert_placement_strategy, ...)` (layer.py:67-153): splits experts as evenly as possible (`base_experts = global_num_experts // ep_size`, remainder assigned to low-rank ranks), returns an `expert_map` tensor mapping global→local expert index (`-1` = not owned). `"linear"` placement gives contiguous blocks (117-119); `"round_robin"` gives a strided assignment (120-127) — restricted (see `determine_expert_placement_strategy`, 156-190) to models with `num_expert_group>1`, no redundant experts, no EPLB, and only usable with `deepep_low_latency` or `nixl_ep` backends (falls back to `"linear"` with a warning otherwise).
- Constructor (`__init__`, layer.py:301 onward): builds `self.moe_parallel_config = FusedMoEParallelConfig.make(...)` (369-375); if `self.use_ep` (425-468), asserts `global_num_experts % ep_size == 0` when EPLB is on, else asserts `num_redundant_experts == 0` (426-434), computes `local_num_experts`/`expert_map`/`expert_mask` via `determine_expert_map`.
- **Where TP-sharded vs EP whole-expert weights diverge**: `assert intermediate_size % self.tp_size == 0; self.intermediate_size_per_partition = intermediate_size // self.tp_size` (layer.py:486-487). Under **TP** (`use_ep=False`), `tp_size>1` shards every expert's intermediate dimension — each rank holds a slice of *every* expert. Under **EP** (`use_ep=True`), `FusedMoEParallelConfig.make` forces `tp_size=1`, so `intermediate_size_per_partition == intermediate_size` (whole expert width) but `self.local_num_experts < global_num_experts` (whole-expert, subset-of-experts sharding).
- Weight loading enforces this at load time: `_map_global_expert_id_to_local_expert_id` (1016-1019) uses `self._expert_map`; `weight_loader` (1060-1136) computes `expert_id = self._expert_map[global_expert_id]` and **returns `False`/skips the copy if `expert_id == -1`** (1089-1091) — i.e. under EP, a rank simply never loads/copies weight tensors for experts it doesn't own, while `_load_w13`/`_load_w2` (937-990) do the TP-style column/row `narrow()` split using `tp_rank`/`tp_size` (which are 1/0 under EP, i.e. no-ops).
- `use_ep`/`ep_size`/`ep_rank`/`tp_size`/`tp_rank` are exposed as properties delegating to `moe_parallel_config` (layer.py:720-738).
- **Reduction semantics without a2a kernels (EP + DP=1, or non-DP TP)**: `DefaultMoERunner.maybe_all_reduce_tensor_model_parallel` (`vllm/model_executor/layers/fused_moe/runner/default_moe_runner.py:331-338`) calls `tensor_model_parallel_all_reduce` (= `get_tp_group().all_reduce`, `vllm/distributed/communication_op.py:12-14`) whenever `self.moe_config.tp_size > 1 or self.moe_config.ep_size > 1` and shared-expert outputs don't already reduce (default_moe_runner.py:371-376). This is the mechanism for **EP without DP**: every EP/TP-group rank still receives the *same* (TP-replicated) full token batch, computes only its local experts (others contribute nothing), and the physical **TP process group** all-reduce sums the partial per-expert contributions — no dispatch/gather communication is used at all, just a final reduce. This is implemented via the no-op prepare/finalize path (`MoEPrepareAndFinalizeNoDPEPModular`, see §5) rather than any all2all backend.

## 4. All2all backends — modular kernel / prepare-finalize layer

**Selection point** — `vllm/distributed/device_communicators/base_device_communicator.py:161-176`:
```python
use_ep = config.parallel_config.data_parallel_size > 1
all2all_backend = config.parallel_config.all2all_backend
self.is_ep_communicator = unique_name.split(":")[0] == "ep"
self.use_all2all = self.is_ep_communicator and use_ep
```
i.e. the all2all manager is constructed **only on the `"ep"` device communicator, and only when `data_parallel_size > 1`** — direct evidence for the DP+EP coupling (§5).

**Manager construction** — `vllm/distributed/device_communicators/cuda_communicator.py:117-172` dispatches on `self.all2all_backend`:
| CLI value | Manager class (`all2all.py`) | Comm primitive / pattern | Hardware assumption |
|---|---|---|---|
| `naive` | `NaiveAll2AllManager` (all2all.py:40-146) | Broadcast-based "multicast" via `get_ep_group().broadcast/all_reduce`; dispatch = per-rank broadcast loop, combine = EP-group all-reduce (69-143) | Any interconnect; for testing/debugging only |
| `allgather_reducescatter` | `AgRsAll2AllManager` (149-248) | `dist_group.all_gatherv` for dispatch (router logits+hidden+topk), `reduce_scatterv` for combine (232-245), over EP or DP group depending on `is_sequence_parallel` | Generic NCCL allgather/reduce-scatter; works over NVLink or plain TCP/IB, no special kernel |
| `deepep_high_throughput` | `DeepEPHTAll2AllManager`(305-360)/base `DeepEPAll2AllManagerBase`(251-302) | DeepEP high-throughput buffer (`deep_ep.Buffer`), grouped-GEMM continuous layout; allocates `num_rdma_bytes` only if `self.internode` (313-335) | Intranode: NVLink; internode: **RDMA/InfiniBand** via DeepEP (`num_qps_per_rank = num_sms//2`); requires `deep_ep` package (`has_deep_ep()` assert, 257-260) |
| `deepep_low_latency` | `DeepEPLLAll2AllManager` (363-427) | DeepEP low-latency masked-layout buffer, `low_latency_mode=True`; sizes RDMA buffer via `Buffer.get_low_latency_rdma_size_hint` (392-397); `allow_mnvl=envs.VLLM_DEEPEP_LOW_LATENCY_USE_MNNVL` | **RDMA/NVSHMEM**-based low-latency path (DeepEP LL uses IBGDA); `max_sms_used()` returns `0` (425-427) since RDMA needs no SMs — supports CUDA graphs (masked/static shapes) |
| `mori` | `MoriAll2AllManager` (762-853) | MoRI EP dispatch/combine op; registers process group with `mori.shmem.shmem_torch_process_group_init` (773-774); intra-node vs inter-node kernel types (`IntraNode`/`InterNodeV1`) | AMD ROCm only, gfx942/gfx950 (793-795); `rdma_block_num` set for internode |
| `nixl_ep` | `NixlEPAll2AllManager` (430-543) | NIXL EP `Buffer` with dynamic `connect_ranks`/`disconnect_ranks` — supports **elastic EP** scale up/down (431-433); RDMA-sized buffer (`Buffer.get_rdma_size_hint`) | RDMA transport via NIXL; `max_sms_used()` = 0 (540-542) |
| `flashinfer_nvlink_two_sided` (alias `flashinfer_all2allv`, deprecated) | `FlashInferNVLinkTwoSidedManager` (545-646) | FlashInfer/TensorRT-LLM `MnnvlMoe` workspace, two-sided A2A | **Multi-node NVLink (MNNVL)** — `Mapping`/`MnnvlConfig` |
| `flashinfer_nvlink_one_sided` | `FlashInferNVLinkOneSidedManager` (649-759) | FlashInfer `MoeAlltoAll`/`trtllm_moe_alltoall`, one-sided put-based A2A | **Multi-node NVLink (MNNVL)**, newer/faster kernel than two-sided |

**Env vars** (`vllm/envs.py`): note the repo has **no `VLLM_ALL2ALL_BACKEND` env var** (searched the whole tree — zero hits); backend selection is purely `--all2all-backend` / `ParallelConfig.all2all_backend`, not an env var, in this codebase version. Related tunables that *are* env vars: `VLLM_DEEPEP_BUFFER_SIZE_MB=1024` (envs.py:220), `VLLM_DEEPEP_HIGH_THROUGHPUT_FORCE_INTRA_NODE` (221), `VLLM_DEEPEP_LOW_LATENCY_USE_MNNVL` (222), `VLLM_NIXL_EP_MAX_NUM_RANKS=32` (248), `VLLM_MOE_DP_CHUNK_SIZE=256` (139), `VLLM_RANDOMIZE_DP_DUMMY_INPUTS` (141), `VLLM_DEEPEPLL_NVFP4_DISPATCH` (148), `VLLM_MOE_ROUTING_SIMULATION_STRATEGY` (1341-1342).

**Prepare/Finalize dispatcher** — `vllm/model_executor/layers/fused_moe/all2all_utils.py:maybe_make_prepare_finalize` (89-274) is the true selection function tying `FusedMoEParallelConfig` booleans to a `FusedMoEPrepareAndFinalize` subclass:
- `not use_all2all_kernels` (i.e. DP≤1, or EP off): `dp_size>1` (plain DP, no EP) → `make_moe_prepare_and_finalize_naive_dp_ep` (AllGather+ReduceScatter fallback, with an explicit log: *"Detected DP deployment with no --enable-expert-parallel. Falling back to AllGather+ReduceScatter dispatch/combine."*, 116-120); else → `make_moe_prepare_and_finalize_no_dp_ep` (no-op prepare/finalize, `prepare_finalize/no_dp_ep.py:39-141` — just local quantization, no comm at all, matching §3's TP-all-reduce-only path).
- Otherwise branches on `moe.use_deepep_ht_kernels` → `DeepEPHTPrepareAndFinalize` (135-145), `use_deepep_ll_kernels` → `DeepEPLLPrepareAndFinalize` (147-180), `use_mori_kernels` → `MoriPrepareAndFinalize` (181-211), `use_fi_nvl_two_sided_kernels`/`use_fi_nvl_one_sided_kernels` → FlashInfer P&F classes (213-230), `use_naive_all2all_kernels` → naive/AGRS P&F (232-237), `use_nixl_ep_kernels` → `NixlEPPrepareAndFinalize` (239-272). All obtain the concrete comm handle from `get_ep_group().device_communicator.all2all_manager`.
- Documented feature matrix in **`docs/design/moe_kernel_features.md`**: activation format (`standard` vs `batched`), supported quant types/formats, async (DBO) support, and "Apply Weight On Input" per backend (lines 33-47). Also states (line 19): *"All backends except `flashinfer` only work with EP+DP or EP+TP. `Flashinfer` can work with EP or DP without EP."* Kernel "families" table (104-112) maps each backend's P&F class to compatible `FusedMoEExpertsModular` implementations (e.g. `deepep_low_latency` → `DeepEPLLPrepareAndFinalize` + `BatchedDeepGemmExperts`/`BatchedTritonExperts`/`CutlassBatchedExpertsFp8`/`BatchedMarlinExperts`).

## 5. DP+EP coupling

Direct evidence chain:
1. **Config docstring** — `ParallelConfig.data_parallel_size` (parallel.py:107-109): *"MoE layers will be sharded according to the product of the tensor parallel size and data parallel size."*
2. **All2all manager gating** — `base_device_communicator.py:170`: `use_ep = config.parallel_config.data_parallel_size > 1` — the EP communicator's all2all manager is built **only if DP>1**.
3. **`FusedMoEParallelConfig.use_all2all_kernels`** (config.py:944-946): `dp_size > 1 and use_ep`. With DP=1, even if `enable_expert_parallel=True` and TP>1 (so EP is structurally active, `ep_size=tp_size`), no dispatch/combine kernel runs — instead the "no-op prepare/finalize + TP-group all-reduce" path is used (§3), meaning every EP rank redundantly processes the *entire* replicated batch and only saves FLOPs on the expert FFN matmul, gaining none of the memory-bandwidth/communication benefits of real token routing.
4. **Why couple with DP for attention**: `docs/serving/data_parallel_deployment.md:7-15` — for MoE models (esp. MLA models like DeepSeek), it's "advantageous to use data parallel for the attention layers and expert or tensor parallel (EP or TP) for the expert layers." With DP driving attention, each DP rank processes an independent request batch (no o_proj all-reduce replication across the whole EP-eligible axis), and the *real* per-token routing only needs to happen at the MoE layer via all-to-all, which is exactly what unlocks efficient wide-EP: many small per-DP-rank batches get gathered/dispatched into large expert-local batches on the EP group, keeping expert GEMMs efficient despite low per-DP-rank concurrency. `docs/serving/expert_parallel_deployment.md:35-68` states plainly: *"EP is typically coupled with Data Parallelism (DP). While DP can be used independently of EP, EP is more efficient when used in conjunction with DP,"* and gives the topology table: with EP, expert layers form an EP group of size `TP×DP` while attention is either TP-sharded (TP>1) or fully replicated/DP-parallel (TP=1) per DP group.
5. **Deployment implication**: this is why the reference multi-node example (`docs/serving/expert_parallel_deployment.md:96-118`) sets `--tensor-parallel-size 1 --enable-expert-parallel --data-parallel-size 16` — attention is data-parallel-replicated per GPU (DP=16), and the entire 16-way group becomes the EP group for a wide, DeepEP-backed expert-parallel deployment across nodes. Same rationale drives `data_parallel_deployment.md:15`: with EP+DP, forward passes and expert-layer collectives across ranks must stay synchronized every step (dummy forward passes are injected on idle DP ranks via a DP Coordinator) — a direct operational cost of the DP+EP coupling.

## 6. Explicit constraints / asserts / incompatibilities

- **EPLB requires EP**: `enable_eplb` ⇒ `enable_expert_parallel` must be True, else `ValueError("enable_expert_parallel must be True to use EPLB.")` (parallel.py:394-395); also requires `TP*DP > 1` (396-401) and CUDA/ROCm only (`current_platform.is_cuda_alike()`, 389-393).
- **EPLB quant-method support**: `FusedMoE.__init__` raises `NotImplementedError(f"EPLB is not supported {self.quant_method.__class__.__name__}.")` if `enable_eplb and not self.quant_method.supports_eplb` (layer.py:601-611) — not all quantization methods implement EPLB weight-redistribution.
- **EPLB even split**: with EPLB, `assert self.global_num_experts % self.ep_size == 0` (layer.py:427-430, *"EPLB currently only supports even distribution of experts across ranks."*); without EPLB, `assert num_redundant_experts == 0, "Redundant experts are only supported with EPLB."` (432-434).
- **Round-robin placement** restricted to `deepep_low_latency`/`nixl_ep` backends, needs `num_expert_group>1`, no redundant experts, no EPLB — else silently falls back to `"linear"` with a warning (layer.py:156-190).
- **Async EPLB**: `use_async` only valid with `policy=="default"` (parallel.py:88-91, model_validator); forced off (with warning) when `all2all_backend in ("allgather_reducescatter","naive")` because *"Async EPLB causes hangs"* with those backends (parallel.py:820-829).
- **Elastic EP** (`enable_elastic_ep`): requires `enable_eplb=True` (699-701); incompatible with `pipeline_parallel_size>1` (`"Elastic EP is not supported with pipeline parallelism"`, 702-706); incompatible with `data_parallel_external_lb`/`data_parallel_hybrid_lb` (`NotImplementedError`, 707-712, "Elastic EP relies on a single API server and core client to coordinate scale up/down").
- **LoRA + EP unsupported**: `vllm/lora/layers/fused_moe.py:49-51` — `assert not self.base_layer.use_ep, "EP support for Fused MoE LoRA is not implemented yet."`
- **DCP orthogonality**: DCP splits the TP axis for attention-context parallelism (`tp_size % decode_context_parallel_size == 0`, parallel.py:416-420) and is structurally independent from EP's DP×PCP×TP axis reshaping in `parallel_state.py`; no explicit code-level incompatibility was found between DCP and EP (they compose because DCP acts purely inside attention/KV-cache handling while EP acts on the MoE/FFN sublayer), but the two features have not been observed to interact in any special-cased assert in this codebase snapshot.
- **`pplx` backend removed**: any config requesting `all2all_backend="pplx"` is silently downgraded to `allgather_reducescatter` with a warning (parallel.py:370-375) — a hard behavioral note for anyone porting old configs.
- **`is_moe_model=False` + DP>1** (offline mode): `ValueError("Offline data parallel mode is not supported/useful for dense models.")` (parallel.py:746-750) — reinforces DP's primary use case being MoE/EP.
- **CUDA graphs**: `deepep_low_latency` explicitly supports CUDA graph capture (masked/static layout, doc table `expert_parallel_deployment.md:23`; `DeepEPLLAll2AllManager.max_sms_used()` returns `0` since it's RDMA-only, all2all.py:425-427); `deepep_high_throughput` is documented as optimized for (variable-shape) prefill and is not listed with CUDA graph support. The doc explicitly recommends `--compilation_config '{"cudagraph_mode": "FULL_DECODE_ONLY"}'` for disaggregated decode instances (expert_parallel_deployment.md:320).
- **DBO SM control**: only `deepep_high_throughput` supports dynamic SM control for communication/compute overlap; `gpu_ubatch_wrapper.py:135-152` looks up `all2all_manager.max_sms_used()`/`set_num_sms()` only when `enable_expert_parallel` is set, and other backends return `None`/no-op for those hooks.
- **Weight loading / EP filter**: `enable_ep_weight_filter` is skipped entirely when `enable_eplb` is on (`default_loader.py:_init_ep_weight_filter`, "redundant physical expert slots may map to logical experts that belong to other ranks... skip the filter entirely").
- **GPTQ/Marlin WNA16 act-order under EP**: needs the *full* pre-sharded `intermediate_size` (`moe_quant_params["intermediate_size_full"] = intermediate_size`) for `GPTQMarlinMoEMethod`, `CompressedTensorsWNA16MarlinMoEMethod`, `CompressedTensorsWNA16MoEMethod` (layer.py:622-628), since EP's per-rank whole-expert weights still need this metadata to reconstruct act-order permutations correctly.
- **`use_overlapped` disabled** (shared-expert/combine overlap) when `enable_eplb and backend != "allgather_reducescatter"` (correctness issue) or when using `use_fi_nvl_two_sided_kernels` (nothing to gain with FlashInfer+DP) or marlin kernels (layer.py:633-644).
# EPLB (Expert Parallelism Load Balancer) — Technical Report

Repo: `/home/user/vllm-voipmonitor` (a vLLM fork). All paths below are absolute; line numbers are as of the current working tree.

Note up front: this checkout's EPLB code has diverged noticeably from vanilla upstream vLLM. There is **no `rebalance_algo.py`** file (the file the prompt asked me to look at) — the rebalancing algorithm has been refactored into a policy-plugin layout at `vllm/distributed/eplb/policy/{abstract.py,default.py,__init__.py}`, and a new async/non-blocking rearrangement path (`async_worker.py`) has been added alongside the original synchronous path. I describe the code as it actually exists here and flag every place it differs from what the prompt assumed.

---

## 1. Concepts: logical / physical / redundant experts

Defined in the module docstring `vllm/distributed/eplb/eplb_state.py:6-27`:

- **Logical expert**: a slot in the model's logical MoE structure (e.g., DeepSeek-R1 has 256).
- **Physical expert**: an actual instantiated copy of a logical expert's weights on some device. `num_physical_experts = num_logical_experts + num_redundant_experts`.
- **Redundant expert**: extra physical copies created for popular logical experts so multiple ranks/replicas can serve the same logical expert concurrently, spreading load.
- **Local physical expert**: a physical expert instantiated on the current rank (`num_physical_experts / num_ep_ranks`).

Example from the docstring: DeepSeek-R1, 256 logical experts + 32 redundant → 288 physical experts; with 32 EP ranks, 9 local physical experts/GPU.

`num_redundant_experts` (`EPLBConfig.num_redundant_experts`, `vllm/config/parallel.py:68-69`) is the knob controlling how many extra physical copies exist across the whole fleet (not per rank) — see the doc's clarification `docs/serving/expert_parallel_deployment.md:177-187`: "each EP rank has `(NUM_TOTAL_EXPERTS + NUM_REDUNDANT_EXPERTS) / NUM_EP_RANKS` experts" and it also drives GPU memory overhead (`NUM_MOE_LAYERS * BYTES_PER_EXPERT * (TOTAL+REDUNDANT)/EP_RANKS`, ~2.4GB per redundant expert on DeepSeek-V3).

### Mapping tables — owned by `EplbModelState` (`vllm/distributed/eplb/eplb_state.py:88-247`)

| Tensor | Shape | dtype | Semantics |
|---|---|---|---|
| `physical_to_logical_map` | `(num_moe_layers, num_physical_experts)` | int64 (built via `torch.tensor(list_of_int)`, `eplb_state.py:392-395`) | For each physical slot, which logical expert it currently holds. Example given in docstring, `eplb_state.py:100-107`. |
| `logical_to_physical_map` | `(num_moe_layers, num_logical_experts, num_redundant_experts+1)` | int64 (inferred; `logical_replica_count` explicitly uses `torch.long` but this tensor's `torch.full(..., -1, device=...)` call at `eplb_state.py:405-409` does not pass `dtype=`, so it inherits PyTorch's integer-fill inference — I could not execute torch in this sandbox to double-check the exact inferred dtype, but it is consumed later purely as an index tensor, and after the first rearrangement it is explicitly `torch.int64` from `np.int64` numpy arrays, `policy/default.py:364-366,373-376`) | Sparse reverse map: for each logical expert, the list of physical slot indices holding it, padded with `-1`. |
| `logical_replica_count` | `(num_moe_layers, num_logical_experts)` | `torch.long` (explicit, `eplb_state.py:410-414`) | Count of non‑`-1` entries per logical expert in `logical_to_physical_map`; i.e. how many replicas exist. |

Ownership:
- `EplbState` (one instance per `GPUModelRunner`, `vllm/v1/worker/gpu_model_runner.py:4512-4513`) owns a dict `model_states: dict[str, EplbModelState]` keyed by `model_config.compute_hash()` — this lets one `EplbState` track both a main model and (optionally) a speculative-decoding drafter model (`gpu_model_runner.py:4529-4555`).
- The three tensors above live once per `EplbModelState` and are **views shared with the model** via `MixtureOfExperts.set_eplb_state()` (`vllm/model_executor/models/interfaces.py:877-907`), which in turn calls each `FusedMoE` layer's `set_eplb_state()` (`vllm/model_executor/layers/fused_moe/layer.py:1451-1466`) to slice out that layer's `[moe_layer_idx]` row and store it on `self.eplb_state: EplbLayerState` (`layer.py:395`, `EplbLayerState` dataclass at `eplb_state.py:1091-1097`). Because these are tensor *views*, in-place updates made by the EPLB algorithm (`.copy_()`) are automatically visible to the router without extra plumbing.

The **maximum replica slot count** (last dim of `logical_to_physical_map`) is fixed at model-add time to `MAX_EXPERT_REDUNDANCY + 1 = 1024` (`eplb_state.py:399-409`), independent of the actual `num_redundant_experts`, so that later rearrangements (which can change relative replica counts) never need to reallocate/resize this tensor — new maps are just `torch.nn.functional.pad`-ed to fit (`eplb_state.py:778-794`, `async_worker.py:92-97`). Comment at `eplb_state.py:396-399` notes this supports up to 128 nodes assuming 8 GPUs/node.

`build_initial_global_physical_to_logical_map` (`eplb_state.py:319-337`) builds the *initial* (pre-rebalance) placement: physical slots `[0, num_routed_experts)` map 1:1 to logical experts, and redundant slots `[num_routed_experts, num_routed_experts+num_redundant_experts)` map to logical experts `i % num_routed_experts`, i.e. redundant copies of experts 0, 1, 2, ... round-robin.

---

## 2. Config flags

`EPLBConfig` — `vllm/config/parallel.py:54-94`:

```
window_size: int = 1000            # steps of load history kept (parallel.py:58-59)
step_interval: int = 3000          # steps between rearrangements (parallel.py:60-66)
num_redundant_experts: int = 0     # Field(ge=0) (parallel.py:68-69)
log_balancedness: bool = False     # parallel.py:71-75
log_balancedness_interval: int = 1 # parallel.py:76-79
use_async: bool = False            # non-blocking EPLB (parallel.py:80-83)
policy: EPLBPolicyOption = "default"  # parallel.py:85-86; EPLBPolicyOption = Literal["default"] (parallel.py:38)
```

`_validate_eplb_config` model_validator (`parallel.py:88-94`): rejects `use_async=True` with a non-`"default"` policy, and rejects `log_balancedness=True` with `log_balancedness_interval <= 0`.

`ParallelConfig` top-level flags — `parallel.py:148-151`:
```
enable_eplb: bool = False
eplb_config: EPLBConfig = Field(default_factory=EPLBConfig)
```

Validation in `ParallelConfig` (`parallel.py:388-409`):
- `enable_eplb=True` requires a CUDA/ROCm platform (`current_platform.is_cuda_alike()`), else `ValueError`.
- Requires `enable_expert_parallel=True` (`"enable_expert_parallel must be True to use EPLB."`).
- Requires `tensor_parallel_size * data_parallel_size > 1`.
- Conversely, if `enable_eplb=False` but `eplb_config.num_redundant_experts != 0`, raises `ValueError` (redundant experts are meaningless without EPLB).

Additional derived behavior (`parallel.py:820-829`): if `all2all_backend` is `"allgather_reducescatter"` or `"naive"` and `use_async=True`, vLLM force-disables async EPLB with a warning ("Async EPLB causes hangs with the '%s' all2all backend").

Elastic EP interaction (`parallel.py:699-712`): `enable_elastic_ep=True` requires `enable_eplb=True`, is incompatible with pipeline parallelism and with `data_parallel_external_lb`/`data_parallel_hybrid_lb`.

CLI wiring — `vllm/engine/arg_utils.py`:
- `enable_eplb: bool = ParallelConfig.enable_eplb` (`arg_utils.py:434`), `eplb_config: EPLBConfig = get_field(ParallelConfig, "eplb_config")` (`arg_utils.py:433`).
- Arg registration: `parallel_group.add_argument("--enable-eplb", ...)` / `("--eplb-config", ...)` (`arg_utils.py:933-934`).
- `eplb_config` can be passed as a JSON string and is coerced: `if isinstance(self.eplb_config, dict): self.eplb_config = EPLBConfig(**self.eplb_config)` (`arg_utils.py:631-632`).
- Propagated into `ParallelConfig(..., eplb_config=self.eplb_config, ...)` (`arg_utils.py:1747-1748`).

Documentation: `docs/serving/expert_parallel_deployment.md:135-224` covers `--enable-eplb`, the `--eplb-config` JSON (and equivalent dotted CLI args), the expert-distribution formula, and the memory-overhead formula (quoted above).

**Per-layer support gate**: `FusedMoE.__init__` raises `NotImplementedError(f"EPLB is not supported {self.quant_method.__class__.__name__}.")` if `enable_eplb and not self.quant_method.supports_eplb` (`vllm/model_executor/layers/fused_moe/layer.py:601-611`). `supports_eplb` defaults to `False` in `FusedMoEMethodBase` (`fused_moe_method_base.py:104-106`) and is overridden to `True` only in `UnquantizedFusedMoEMethod` (`unquantized_fused_moe_method.py:103-105`), `fp8.py:956`, `modelopt.py:1411`, and three places in `compressed_tensors_moe.py:1005,1904,2538`. Notably **mxfp4** (`vllm/model_executor/layers/quantization/mxfp4.py`) and **quark** (`quark_moe.py`) do not override it, so EPLB is unsupported for those quant methods by default.

---

## 3. Load statistics collection

**Where counters live**: `EplbModelState.expert_load_pass` — shape `(num_moe_layers, num_physical_experts)`, dtype `int32` (`eplb_state.py:448-452`), and `expert_load_window` — shape `(window_size, num_moe_layers, num_physical_experts)`, dtype `int32` (`eplb_state.py:453-462`). Both zeroed/created in `EplbState.add_model` (`eplb_state.py:375-515`).

**Increment site (per forward pass)**: `eplb_map_to_physical_and_record` in `vllm/model_executor/layers/fused_moe/router/base_router.py:14-96` (CUDA path; there's also a CPU no-op fallback at `base_router.py:89-96`), `torch.compile`-wrapped (`@torch.compile(dynamic=True, ...)`, `base_router.py:16`). Called from `BaseRouter._apply_eplb_mapping` (`base_router.py:156-168`) inside `select_experts()` (`base_router.py:203-249`), which is the common template method every concrete router (`grouped_topk_router.py`, `fused_topk_router.py`, `custom_routing_router.py`, etc.) goes through.

Mechanics (`base_router.py:40-86`):
1. `topk_ids` (logical expert ids from the router) are converted to `.long()`.
2. A **replica** for each logical expert is chosen pseudo-deterministically: `replica_count = logical_replica_count[topk_ids_long]`; `replica_indices = (flatten_position_index % replica_count)`; `physical_ids = logical_to_physical_map[topk_ids_long].gather(-1, replica_indices)`. This is not random — it's a deterministic hash based on flattened token position, chosen so token load is spread evenly across replicas of the same logical expert without extra RNG state.
3. `expert_load_view.scatter_add_(dim=0, index=topk_ids_flatten (now physical ids), src=ones)` — increments the physical-expert counter for every routed token (each of the `top_k` selections counts once). `torch.bincount` is explicitly avoided because it isn't `torch.compile`-compatible (comment `base_router.py:65-75`).

Note recorded in the `expert_load_window` field docstring (`eplb_state.py:160-168`): load is recorded for **all physical experts**, not just local ones, to keep stats consistent across dispatch backends (naive all-to-all vs DeepEP); with naive all-to-all + data-parallel, recorded load is inflated by `dp_size` since every DP rank contributes the same token set (linked PR discussion cited in the comment).

**Rolling window semantics** (`EplbState.step`, `eplb_state.py:517-641`, window portion at `601-611`):
```python
if not is_dummy:
    for eplb_model_state in self.model_states.values():
        eplb_model_state.expert_load_window[self.expert_load_window_step] = (
            eplb_model_state.expert_load_pass.clone()
        )
        eplb_model_state.expert_load_pass.zero_()
    self.expert_load_window_step += 1
    if self.expert_load_window_step >= self.expert_load_window_size:
        self.expert_load_window_step = 0
```
This is a circular buffer of size `window_size` (default 1000). Every non-dummy engine step, the just-accumulated `expert_load_pass` (a single forward pass's counts) is snapshotted into the current window slot and the accumulator is reset; the write cursor wraps mod `window_size`. `is_dummy` steps (CUDA-graph capture, empty-batch dummy runs for DP synchronization) instead just zero `expert_load_pass` without touching the window (`eplb_state.py:547-550`) — they don't pollute stats, but the code comment stresses the rearrangement-step counter is still advanced on dummy steps to keep all ranks' collective calls synchronized (`eplb_state.py:613-617`).

**All-reduce across ranks**: happens in `EplbState.rearrange()` (not every step — only when a rearrangement is triggered). Per-layer physical loads are first scattered into **logical** expert space via `scatter_add_` using `physical_to_logical_map` as the index (`eplb_state.py:676-701`), summed over the window (`.sum(dim=0)`), then all-reduced across the EP group: `self._allreduce_list(global_expert_load_windows)` (`eplb_state.py:702-703`, implementation `eplb_state.py:999-1024`, using `torch.distributed.all_reduce(..., group=get_ep_group().device_group)`, with a batched code path that concatenates multiple models' load tensors into one `all_reduce` call when there's more than one tracked model, e.g. main model + spec-decode drafter).

**Balancedness logging** (`EplbState.step`, `eplb_state.py:552-599`), gated by `eplb_config.log_balancedness` and `log_balancedness_interval`: separately all-reduces a *snapshot* of `expert_load_pass` (via `_sync_load_pass`, `eplb_state.py:1026-1034`, which does **not** mutate the real accumulator), reshapes per-layer physical loads into `(num_moe_layers, num_ranks)` and computes `balancedness = avg_tokens_per_rank / max_tokens_per_rank`, logged from rank 0 only (`eplb_state.py:587-599`).

---

## 4. The rebalancing algorithm — `vllm/distributed/eplb/policy/default.py` (formerly `rebalance_algo.py`)

The algorithm is pluggable: `AbstractEplbPolicy` (`policy/abstract.py:9-43`) defines the `rebalance_experts()` contract; `EPLB_POLICIES = {"default": DefaultEplbPolicy}` (`policy/__init__.py:10`), with an assertion tying the registry keys to `EPLBPolicyOption`'s `Literal` values (`policy/__init__.py:13`). Only one concrete policy, `DefaultEplbPolicy`, exists today, adapted from DeepSeek's public EPLB algorithm (docstring credit, `policy/default.py:6-12`, linking `github.com/deepseek-ai/EPLB`).

### `balanced_packing` (`policy/default.py:22-73`)
Greedy bin-packing: given `weight[num_layers, num_groups]`, pack `num_groups` items into `num_packs` bins so each bin gets exactly `num_groups/num_packs` items, minimizing max bin weight. Special-cased for `groups_per_pack == 1` (identity assignment, `default.py:42-45`). Otherwise, for each layer, sort groups by descending weight, then for each group greedily assign it to the currently-lightest non-full pack (`np.argmin` over `pack_weights`, with full packs masked to `inf`, `default.py:57-72`). This is a classic longest-processing-time-first (LPT) greedy heuristic — not optimal but fast and good in practice, run in pure NumPy per layer (Python loop over `num_layers`, vectorized within a layer only over `argmin`).

### `replicate_experts` (`policy/default.py:75-104`)
Greedily grows a logical expert set from `num_log` to `num_phy` replicas: at each step, picks the logical expert with the current highest **per-replica** load (`weight / logcnt`, i.e. the most "starved for replicas" expert) and gives it one more replica (`np.argmax(weight/logcnt)`, then increments its `logcnt`). This is repeated `num_phy - num_log` times. Vectorized across all layers (`n`) simultaneously via `arangen` fancy indexing.

### `rebalance_experts_hierarchical` (`policy/default.py:106-202`)
This is the "hierarchical" (as opposed to flat/global) load balancer, structured in three stages, driven by the **group-limited routing** assumption (each logical expert belongs to one of `num_groups` groups, and groups can be affinitized to nodes — this mirrors DeepSeek's "group-limited routing" MoE design where the router first picks top groups, then experts within them, so keeping a group's experts co-located on one node minimizes cross-node dispatch traffic):

1. **Step 1 — pack groups to nodes** (`default.py:148-163`): sum load per group (`tokens_per_group`), then `balanced_packing` groups into `num_nodes` bins. Builds a permutation `log2mlog`/`mlog2log` mapping logical expert index → a node-local reordered index, so all experts of the same group end up contiguous within one node's logical range.
2. **Step 2 — replicate within node** (`default.py:165-172`): reorders the load into node-local layout (`tokens_per_mlog`), then calls `replicate_experts` per node's slice to build `num_physical_experts/num_nodes` physical replicas per node, minimizing intra-node load imbalance. Produces `phy2mlog`, `mlogcnt`.
3. **Step 3 — pack physical experts to GPUs** (`default.py:174-181`): computes effective per-replica load (`tokens_per_mlog / mlogcnt`), then `balanced_packing`s those replicas onto `num_gpus/num_nodes` GPUs within each node.

The remaining lines (`default.py:183-202`) undo all the intermediate index permutations to produce final outputs in the original logical-expert numbering: `pphy2log` (physical→logical), `pphy_replicas_idx` (replica rank per physical slot), and `logcnt` (final replica count per logical expert, restored to original ordering via `mlog2log`).

### `preserve_intragpu_slots` (`policy/default.py:204-294`)
A postprocessing pass (not present in upstream DeepSeek EPLB — appears to be a vLLM-specific optimization) that, when the GPU count and slots-per-GPU are unchanged between old and new mappings, reorders the *new* per-GPU expert assignment so that experts that are staying on the same GPU keep their old slot position where possible. This minimizes unnecessary intra-GPU weight *shuffling* (row copies within the same tensor) even when logically nothing needs to cross the network — pure optimization to reduce `move_to_buffer`/`move_from_buffer` work, not correctness-affecting. Two-pass algorithm per GPU: first pass matches unchanged logical experts to their old slot; second pass fills remaining slots with whatever's left over.

### `rebalance_experts` — the entry point (`policy/default.py:296-376`)
```python
if num_groups % num_nodes == 0:
    # hierarchical
    ... = cls.rebalance_experts_hierarchical(weight_np, num_replicas, num_groups, num_nodes, num_ranks)
else:
    # "global" load-balance — degenerates to the hierarchical routine with num_groups=1, num_nodes=1
    ... = cls.rebalance_experts_hierarchical(weight_np, num_replicas, 1, 1, num_ranks)
```
**Important correction vs. the prompt's framing**: there is no separate "global" algorithm implementation — the so-called global policy is literally the same `rebalance_experts_hierarchical` function called with `num_groups=1, num_nodes=1` (i.e., one giant "group" and one "node", degrading hierarchical packing to flat packing). This is used whenever `num_groups` doesn't evenly divide `num_nodes` (`default.py:338,345-351`; also triggered from the caller when `num_gpus % num_nodes != 0`, `eplb_state.py:727-733`, which forces `num_nodes=1`).

Everything runs on **CPU/NumPy**: `weight.float().cpu().numpy()` (`default.py:331`), and results are converted back with `torch.from_numpy(...).to(device)` (`default.py:373-376`). This is deliberate — the packing algorithm is inherently sequential/greedy and doesn't parallelize well on GPU; doing it on CPU also avoids interfering with the GPU compute stream.

`old_global_expert_indices` (the current `physical_to_logical_map`), when supplied, feeds `preserve_intragpu_slots` (`default.py:358-361`) to minimize needless data movement.

Final packaging (`default.py:362-376`): builds the dense `log2phy` (i.e. `logical_to_physical_map`) sparse table with shape `(num_layers, num_logical_experts, num_redundant_experts+1)` — note this differs from the padded-to-1024 shape stored persistently in `EplbModelState`; the caller pads it (`eplb_state.py:778-794`).

---

## 5. Rearrangement execution — `vllm/distributed/eplb/rebalance_execute.py`

### Buffers
`rearrange_expert_weights_inplace` (`rebalance_execute.py:550-649`) allocates **one set of scratch buffers per weight tensor, shared across all MoE layers** (not one buffer per layer): `weights_buffer = [torch.empty_like(w) for w in expert_weights[0]]` (`rebalance_execute.py:604-610`) — sized off layer 0's tensors, with the code explicitly assuming (`NOTE` comment, `rebalance_execute.py:606-607`) that the same weight tensor (e.g. `w13_weight`) has the same shape across all layers. It then loops `for layer_idx in range(num_moe_layers)` and reuses the same buffer for each layer sequentially (`rebalance_execute.py:630-649`) — this bounds the **extra memory overhead to roughly one layer's worth of local expert weights**, not the whole model's.

In async mode, an equivalent per-model buffer (`EplbModelState.expert_buffer`, allocated once in `add_model`: `expert_buffer = [torch.empty_like(w) for w in model.expert_weights[0]]`, `eplb_state.py:481`) is reused layer-by-layer by the background thread, protected by `EplbModelState.buffer_lock` (a `threading.Lock`) plus a pair of CUDA events (`buffer_ready_event`, `buffer_consumed_event`) to hand off producer/consumer ownership between the async worker's CUDA stream and the main thread's stream without a full device sync (`eplb_state.py:171-247`, `async_worker.py:158-182`, `eplb_state.py:906-989`).

### `move_to_buffer` (`rebalance_execute.py:154-386`) — the core P2P engine
Given per-layer `old_indices`/`new_indices` (numpy arrays of length `num_physical_experts = ep_size * num_local_experts`), for the calling rank's local expert rows:
1. Computes `is_unchanged` (same logical expert before/after — no data movement needed) and `is_received_locally` (needed expert already exists somewhere *on this same rank*, so it can be a plain local tensor copy instead of a network transfer) (`rebalance_execute.py:200-210`).
2. Builds a **send map** (unique experts this rank currently holds, and their first local row) and a **receive map** (unique experts this rank needs but doesn't have, and their destination row) (`rebalance_execute.py:212-236`).
3. **Local moves** (`rebalance_execute.py:238-251`): copies rows within the same rank straight into the scratch buffer with `.copy_(..., non_blocking=True)` — no network.
4. **Remote moves**: uses `get_ep_ranks_with_experts_batch` (`rebalance_execute.py:47-151`, a vectorized NumPy routine using `np.unique`/`np.lexsort`) to compute, for every expert being exchanged, the full list of ranks that currently hold it (`ranks_to_send`) and ranks that need it (`ranks_to_recv`). It then does a **round-robin fan-out/fan-in assignment** splitting receivers evenly across senders (`num_dst_per_sender = len(ranks_to_recv)//len(ranks_to_send)`, `rebalance_execute.py:287-295` for sends, `rebalance_execute.py:335-341` for recvs) — i.e., this is a **peer-to-peer (isend/irecv) pattern, not a broadcast/all-gather**, deliberately load-balancing the fan-out of popular experts across all ranks that hold a copy, rather than having one source rank serve every request.
5. **Batched P2P execution** (`rebalance_execute.py:360-376`): all queued `P2POp`s (`isend`/`irecv`) for the layer are submitted together via `torch.distributed.batch_isend_irecv(p2p_ops)`, then each request's `.wait()` is called (`reqs = batch_isend_irecv(p2p_ops); for req in reqs: req.wait()`), i.e. this **blocks the calling thread until all P2P ops for the layer complete** in synchronous/main-thread mode (`cuda_stream=None` path, `rebalance_execute.py:369-375`). In async-thread mode, the same ops are enqueued `with torch.cuda.stream(cuda_stream):` (`rebalance_execute.py:361-368`) so they run on the dedicated async CUDA stream instead of blocking the main compute stream, though `.wait()` is still called from within the async worker's own coroutine/thread.
6. There's a **stateless-group branch** (`is_stateless`, `rebalance_execute.py:253-259,297-303,342-348`): when `get_ep_group()` is a `StatelessGroupCoordinator` (used for elastic EP, where ranks are not part of a normal `torch.distributed` `ProcessGroup`), P2P ops are hand-constructed with `object.__new__(P2POp)` and dispatched via `ep_group.device_communicator.batch_isend_irecv(p2p_ops)` instead of `torch.distributed.batch_isend_irecv`.

`move_from_buffer` (`rebalance_execute.py:389-465`) then copies data back out of the scratch buffer into the real `expert_weights` tensors: for rows that were locally moved or were a remote-recv "primary" landing spot, and additionally **duplicates** received rows to any other local destination rows that wanted the same expert (avoiding redundant network transfers when a rank needs the same incoming expert in multiple local slots) (`rebalance_execute.py:432-465`).

### Is it blocking?
- **Synchronous (default) path**: `EplbState.rearrange()` (`eplb_state.py:643-832`), when `not self.is_async` (or during a profile run), computes the new mapping and calls `rearrange_expert_weights_inplace(...)` **inline on the main thread**, looping over all `num_moe_layers` synchronously (`rebalance_execute.py:630-649`), each layer's P2P ops being `.wait()`-ed before moving to the next layer. This is a genuine **stall of the serving loop** — no requests can make forward progress on this rank while this runs (it's literally invoked from the model-runner's `execute_model`, see §6). `rearrange()` even measures and logs the elapsed wall time via CUDA events (`start_event`/`end_event`, `eplb_state.py:662-669,798-808`, only measured on rank 0 / `is_main_rank`).
- **Async path** (`eplb_config.use_async=True`): `EplbState.rearrange()` instead just computes the new expert-load snapshot (`EplbStats`), records a "window ready" CUDA event, sets `model_state.rebalanced=True`, and signals a `threading.Event` (`self.rearrange_event.set()`, `eplb_state.py:809-831`). A background daemon thread (`async_worker.py:25-56`, started once via `start_async_worker`) picks this up in `transfer_run_periodically` (`async_worker.py:103-188`): it first computes the algorithm result on CPU (`run_rebalance_experts`, `async_worker.py:59-100`, called from the worker thread, not blocking the main thread), then transfers **one layer at a time** (`transfer_layer`, `rebalance_execute.py:468-547`) using its own CUDA stream and its own `_EPLB` process group (see §6), interleaving with `move_to_workspace` on the main thread (`eplb_state.py:906-989`) which is invoked from `EplbState.step()` every step while a transfer is pending (`eplb_state.py:619-632`) — so the actual weight-buffer consumption is spread across many subsequent engine steps rather than stalling one step. The main thread only briefly blocks acquiring `buffer_lock` (with a 6×10s retry/timeout before raising, `eplb_state.py:914-928`), which should normally be uncontended and fast since the async worker holds it only around each buffer write.
- The comment at `rebalance_execute.py:623-625` ("`NOTE(bowen)`: We need this synchronize to run, but I don't know why...") flags an unresolved `torch.accelerator.synchronize()` call inserted defensively before the sync rearrangement loop — an honest unresolved mystery left in the vLLM codebase itself.

### Elastic EP rank remapping
`_map_old_expert_indices_with_rank_mapping` / `_map_new_expert_indices_with_rank_mapping` (`rebalance_execute.py:652-737`) let `rearrange_expert_weights_inplace`/`transfer_layer` handle scale-up/scale-down of the EP group (a `rank_mapping: dict[old_rank -> new_rank]`, with `-1` meaning "being shut down"), remapping the physical-expert index space before running the normal P2P logic.

### Memory overhead of rearrangement
Roughly **one extra copy of one MoE layer's local expert weights** (the shared `weights_buffer`, sized from layer 0, `rebalance_execute.py:604-610`), reused across all layers — bounded, not `O(num_layers)`. In async mode this is `EplbModelState.expert_buffer`, likewise one-layer-sized and persistent for the model's lifetime (`eplb_state.py:481`). This is on top of the standing per-GPU memory overhead of holding `num_redundant_experts` extra physical experts (see §1/§2 memory-footnote from the docs).

### `is_profile` special case
When `is_profile=True`, `rearrange_expert_weights_inplace` **does not move any real data**; it only does a minimal dummy `all_gather` on layer-0's weights with a `torch.distributed.barrier()` beforehand (`rebalance_execute.py:611-621`), purely to force PyTorch/NCCL to allocate and reserve whatever communication buffers the real transfer would need later, so `profile_run()`'s memory accounting captures that overhead upfront (see §6).

---

## 6. When it triggers

### The counters
`EplbState` maintains (`eplb_state.py:263-289`):
- `expert_load_window_step` / `expert_load_window_size` — the circular window cursor/size (from `eplb_config.window_size`).
- `expert_rearrangement_step` / `expert_rearrangement_step_interval` — steps-since-last-rearrangement / trigger threshold (from `eplb_config.step_interval`).

`expert_rearrangement_step` is deliberately **not initialized at 0** — it starts at `max(0, step_interval - step_interval // 4)` i.e. **75% of the way to the first trigger** (`eplb_state.py:464-469`, comment: "Set the initial progress of rearrangement to 3/4"). This means the very first rebalancing happens after only 1/4 of the configured interval — presumably to correct an initially-uniform-but-likely-wrong placement quickly, then settle into the full interval cadence thereafter.

### `EplbState.step()` (`eplb_state.py:517-641`)
Called once per engine forward step (see call sites below), with `is_dummy` and `is_profile` flags:
1. If `is_profile`: immediately calls `self.rearrange(is_profile=True)` and returns — a one-off, no counters touched (`eplb_state.py:543-545`).
2. If `is_dummy`: zeroes `expert_load_pass` for all tracked models without recording it into the window (§3).
3. Optional balancedness logging (§3), gated on `log_balancedness` and only every `log_balancedness_interval` rearrangement-steps.
4. Window update (only if `not is_dummy`, §3).
5. `self.expert_rearrangement_step += 1` — **incremented every step regardless of `is_dummy`**, with an explicit comment explaining this is required so all EP ranks execute the same number of collective calls even when some ranks are on dummy/empty batches in a DP setting (`eplb_state.py:613-617`).
6. Async-mode bookkeeping: if async transfer is pending and all ranks report their buffer ready, drains one layer via `move_to_workspace` (`eplb_state.py:619-631`).
7. **Trigger check**: `if self.expert_rearrangement_step >= self.expert_rearrangement_step_interval:` — resets the counter to 0 and calls `self.rearrange()` (`eplb_state.py:633-641`). In async mode, if a previous rearrangement is still `model_state.rebalanced` (i.e., still mid-transfer), the new trigger is simply skipped for that pass (`eplb_state.py:634-639`) — it doesn't queue up a second overlapping rearrangement.

### Call sites (`vllm/v1/worker/gpu_model_runner.py`)
- `eplb_step()` wrapper (`gpu_model_runner.py:2881-2895`): no-ops if `not enable_eplb or self.eep_eplb_suppressed` (the latter flag is set during elastic-EP scale-up to temporarily suppress normal rearrangement, `vllm/v1/worker/gpu_worker.py:341`); otherwise asserts `self.eplb_state is not None` and the model `is_mixture_of_experts`, then calls `self.eplb_state.step(is_dummy, is_profile, log_stats=parallel_config.eplb_config.log_balancedness)`.
- **Real-request path**: `execute_model()` calls `self.eplb_step()` (default `is_dummy=False`) after bookkeeping and after finalizing the KV connector, inside `record_function_or_nullcontext("gpu_model_runner: eplb")` (`gpu_model_runner.py:4074-4075`). This means EPLB stats are collected — and periodically the (possibly blocking) rearrangement is triggered — **as an ordinary part of every real forward pass**, i.e. "mid-serving" as the prompt asks: yes, in sync mode a triggering step genuinely stalls request processing on that rank for the duration of the collective weight exchange.
- **Dummy/graph-capture path**: `_dummy_run()` calls `self.eplb_step(is_dummy=True, is_profile=is_profile)` (`gpu_model_runner.py:5300-5308`), guarded by a `skip_eplb` parameter. The surrounding comment explains the DP-synchronization rationale directly: dummy batches on idle DP ranks must still call `eplb_step` (with `is_dummy=True`) so that all ranks execute the same rearrangement-trigger collective calls in lockstep and don't hang waiting on ranks that skipped it.
- **`skip_eplb=True` call sites** (deliberately bypass even the dummy-step counting): CUDA-graph/kernel warmup and compile-warmup batches in `gpu_worker.py:599` and `gpu_worker.py:699` ("We skip EPLB here since we don't want to record dummy metrics"), and in `vllm/model_executor/warmup/kernel_warmup.py:74,105`.
- **Profile run**: `profile_run()` calls `self._dummy_run(self.max_num_tokens, is_profile=True)` (`gpu_model_runner.py:5541-5543`) purely to pre-allocate the communication buffers needed for the largest possible transfer (§5's `is_profile` no-op-data-move path), so the memory profiler's peak-memory estimate correctly accounts for EPLB's buffer overhead before real serving starts.
- **EPLB state construction**: `load_model()` creates `self.eplb_state = EplbState(self.parallel_config, self.device)` if `enable_eplb` (`gpu_model_runner.py:4512-4513`), then calls `add_model()` for the drafter model if present (`gpu_model_runner.py:4547-4555`) and for the main model after weight loading (`gpu_model_runner.py:4610-4622`); if `eplb_state.is_async`, also calls `eplb_state.start_async_loop()` there (`gpu_model_runner.py:4621-4622`) to spin up the background thread.

### Separate process group for EPLB
vLLM creates a **dedicated `_EPLB` process group**, distinct from `_EP` (the MoE forward-pass expert-parallel group), specifically to isolate EPLB's own collectives from the model's regular collectives — the comment is explicit (`vllm/distributed/parallel_state.py:1684-1687`): *"Create EPLB group with the same ranks as EP if EPLB is enabled. This is a separate process group to isolate EPLB communications from MoE forward pass collectives and prevent deadlocks..."* Accessors: `get_ep_group()`/`get_eplb_group()` (`parallel_state.py:1251-1269`), constructed conditionally on `parallel_config.enable_eplb` (`parallel_state.py:1688-1713`, supports both a normal `init_model_parallel_group` and, for elastic EP, a `_init_stateless_group`). The async worker specifically pulls `get_eplb_group().device_group` (`async_worker.py:29`), while the synchronous `rearrange()` path uses `get_ep_group().device_group` (`eplb_state.py:542,659`).

### `eplb_utils.py` — environment override
`override_envs_for_eplb` (`eplb_utils.py:13-55`) force-sets `NCCL_MAX_CTAS=8` when `data_parallel_size>1`, `enable_eplb`, `all2all_backend=="deepep_low_latency"`, and `use_async` are all true, to avoid a documented deadlock between NCCL (used by async EPLB's weight exchange) and DeepEP low-latency's cooperative kernel launch competing for SMs (references `github.com/deepseek-ai/DeepEP/issues/496`). Called from `gpu_worker.py:1044`.

---

## 7. Model support — `MixtureOfExperts` protocol

`vllm/model_executor/models/interfaces.py:836-919` (`@runtime_checkable class MixtureOfExperts(Protocol)`), fields:
```
expert_weights: MutableSequence[Sequence[Tensor]]
num_moe_layers, num_expert_groups, num_logical_experts,
num_physical_experts, num_local_physical_experts,
num_routed_experts, num_shared_experts, num_redundant_experts: int
moe_layers: Iterable[nn.Module]
```
Methods:
- `set_eplb_state(expert_load_view, logical_to_physical_map, logical_replica_count)` (`interfaces.py:877-907`) — has a **default (non-abstract) implementation** that iterates `self.moe_layers`, appends each layer's `get_expert_weights()` to `self.expert_weights`, and calls each layer's own `set_eplb_state(moe_layer_idx=..., ...)`. The docstring explicitly instructs implementers to collect `expert_weights` here rather than in the weight loader, because post-load processing (e.g. quantization) may still transform the weights afterward (`interfaces.py:890-892`).
- `update_physical_experts_metadata(num_physical_experts, num_local_physical_experts) -> None: ...` (`interfaces.py:909-913`) — abstract-ish (body is `...`), used for elastic EP resizing.
- `is_mixture_of_experts(model)` helper (`interfaces.py:916-919`) checks `isinstance(model, MixtureOfExperts) and getattr(model, "num_moe_layers", 0) > 0`.

**Models implementing it** (28 files reference `MixtureOfExperts` under `vllm/model_executor/models/`): `transformers/moe.py`, `sarvam.py`, `step3p5.py`, `qwen3_next_mtp.py`, `qwen3_next.py`, `qwen3_vl_moe.py`, `qwen3_5_mtp.py`, `qwen3_5.py`, `qwen3_moe.py`, `openpangu.py`, `nemotron_h.py`, `mixtral.py`, `mllama4.py`, `mimo_v2_flash.py`, `kimi_linear.py`, `lfm2_moe.py`, `llama4.py`, `interns1_pro.py`, `hunyuan_v1.py`, `glm4_moe_lite_mtp.py`, `glm4_moe_lite.py`, `glm4_moe_mtp.py`, `glm4_moe.py`, `deepseek_v2.py`, `ernie45_moe.py`, `deepseek_mtp.py`, `AXK1.py`, plus `interfaces.py` itself.

**DeepSeek** (`deepseek_v2.py`, which also backs DeepSeek-V3/R1): `DeepseekV2MoE.__init__` sets `n_routed_experts`, `n_shared_experts` (`deepseek_v2.py:252-253`), `n_redundant_experts = eplb_config.num_redundant_experts` (`deepseek_v2.py:279`), `n_logical_experts = n_routed_experts` (`deepseek_v2.py:280`), `n_physical_experts = n_logical_experts + n_redundant_experts` (`deepseek_v2.py:281`), `n_local_physical_experts = n_physical_experts // ep_size` (`deepseek_v2.py:282`). A `DeepseekV2MixtureOfExperts` mixin (`deepseek_v2.py:1256-1294`) implements `extract_moe_parameters()` and `update_physical_experts_metadata()`. `DeepseekV2ForCausalLM(nn.Module, SupportsPP, DeepseekV2MixtureOfExperts, SupportsLoRA, SupportsEagle, SupportsEagle3)` (`deepseek_v2.py:1297-1304`) wires it up in `set_moe_parameters()` (`deepseek_v2.py:1361-1380`): walks `self.model.layers`, picks the last `DeepseekV2MoE` instance found as `example_moe`, populates `self.moe_layers`/`self.moe_mlp_layers`, calls `extract_moe_parameters(example_moe)`.

**GLM4-MoE** (`glm4_moe.py`) — structurally identical pattern: `Glm4MixtureOfExperts` mixin (`glm4_moe.py:597-622`), `Glm4MoeForCausalLM(nn.Module, SupportsPP, SupportsLoRA, Glm4MixtureOfExperts)` (`glm4_moe.py:625`), same `n_logical_experts`/`n_physical_experts`/`n_local_physical_experts` bookkeeping directly on the `Glm4MoE` mlp module.

So: **yes**, both DeepSeek-style and GLM-style MoE fully implement the interface, and share essentially the same mixin pattern (this pattern is copy/pasted across most of the 28 listed models).

`FusedMoE.get_expert_weights()` (`layer.py:1381-1449`) is the concrete per-layer weight-collection routine invoked from `MixtureOfExperts.set_eplb_state`'s default body: it fixes up any non-contiguous `weight_scale` tensors (transposed last-2-dims case, `layer.py:1382-1419`), excludes `NON_EXPERT_WEIGHTS = {"e_score_correction_bias","w13_input_scale","w2_input_scale"}` (global/broadcast tensors that aren't per-expert, `layer.py:1428-1432`), asserts contiguity for the rest, and returns each remaining parameter reshaped to `(local_num_experts, -1)` — this flattened-per-expert-row view is exactly what `move_to_buffer`/`move_from_buffer` operate on (indexing by expert row).

---

## 8. Docs, warnings, limitations

Docs mentioning EPLB: only `docs/serving/expert_parallel_deployment.md` (EPLB section: lines 135-224) and an unrelated `docs/governance/committers.md` name-drop. There is **no dedicated `docs/design/eplb.md`** or similar deep-dive doc — the design detail lives only in code docstrings/comments.

Explicit warnings/limitations found in the doc and code:
- "EP is an experimental feature. Argument names and default values may change in the future." (`expert_parallel_deployment.md:30-31`).
- Memory-footnote: EPLB's redundant experts must fit in GPU memory, "may not be a good fit for memory constrained environments or when KV cache space is at a premium" (`expert_parallel_deployment.md:182-187`), with the DeepSeek-V3 ≈2.4GB/redundant-expert figure.
- Recommendation for large multi-node deployments: set `num_redundant_experts: 32` "so the most popular experts are always available" (`expert_parallel_deployment.md:203`).
- `window_size` vs `step_interval`: "if [step_interval] is greater than the EPLB window size, only the metrics of the last `lb_window_size` steps will be used for rearranging experts" (`EPLBConfig.step_interval` docstring, `parallel.py:60-66`) — i.e. if you rebalance less often than your window covers, you effectively discard older load data rather than accumulating a longer history.
- Async EPLB is explicitly documented as risky with certain all2all backends: forced off for `allgather_reducescatter`/`naive` (`parallel.py:820-829`), and there's a documented NCCL/DeepEP-low-latency deadlock worked around via `NCCL_MAX_CTAS` (`eplb_utils.py:24-54`).
- An unresolved/undocumented `torch.accelerator.synchronize()` call is left with a "don't know why we need this" comment right before the synchronous rearrangement loop (`rebalance_execute.py:623-625`) — an honest maintainer admission of an unexplained requirement.
- EPLB is gated to CUDA/ROCm only (`parallel.py:389-393`) — not supported on CPU/TPU/etc.
- Quant-method gating: EPLB unsupported for MoE quant backends that don't set `supports_eplb=True` (mxfp4, quark, and any custom/未-listed quant method) — raises `NotImplementedError` at layer construction (`layer.py:601-611`).
- Troubleshooting notes for the underlying EP/DeepEP transport (`non-zero status: 7 cannot register cq buf`, `init failed for transport: IBGDA`, NVSHMEM peer disconnect) are documented in `expert_parallel_deployment.md:213-217`, though these are EP/DeepEP-transport issues in general, not EPLB-specific.

---

## Things I could not fully verify

1. **Exact dtype of `logical_to_physical_map` at initial construction** (`eplb_state.py:405-409`): the `torch.full((...), -1, device=self.device)` call has no explicit `dtype=` kwarg. I could not run PyTorch in this sandbox (`ModuleNotFoundError: No module named 'torch'`) to confirm PyTorch's dtype-inference for an integer fill value with no `dtype=`. Based on documented PyTorch ≥1.5 semantics it should end up `torch.int64`, and it's certainly treated as an integer index tensor everywhere downstream (indexing/gather in `base_router.py`, later overwritten with explicit `int64` numpy-derived tensors after the first rearrangement, `policy/default.py:364-376`), but I did not empirically confirm the exact dtype at the moment of initial (pre-first-rearrangement) construction.
2. I did not trace every one of the 28 files reporting `MixtureOfExperts` matches individually (e.g., did not read `mixtral.py`, `llama4.py`, `qwen3_moe.py` in full) — I confirmed the pattern in depth for DeepSeek and GLM4 (the two models the prompt specifically asked about) and take the grep hits as reasonable evidence the same mixin pattern is used broadly, but cannot vouch for every listed file's completeness/correctness of implementation without reading each one.
3. Elastic EP (`vllm/distributed/elastic_ep/*`) interacts with EPLB (`rank_mapping`-driven `eplb_state.rearrange()` calls, `elastic_execute.py:422-451`) — I traced this only far enough to confirm it's a distinct, on-demand trigger path (not part of the periodic `step_interval` cadence) and did not do a full study of elastic EP itself, since it was outside the prompt's explicit scope.
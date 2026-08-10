# Fungible Quant: Technical Assessment and Implementation Plan

**Target:** GLM-5.2 / EXL3 on 4× RTX PRO 6000 Blackwell (SM120, TP4+DCP4, EP=1)
**Verdict date:** 2026-08-10
**Author:** lead architect review, based on four prior-art sweeps, five adversarially-verified vLLM feasibility lenses, and direct source verification against `gg/dev/gilded-gnosis`.

---

## 1. VERDICT

**Is it novel? Mostly no. Is it feasible? Yes — but only a specific, constrained version of it, and the version that is feasible is not the version that was proposed.**

### Already solved elsewhere, by name

- **The whole runtime loop** — EMA hotness from live router traces → budget-constrained top-N precision set → asynchronous promote/demote → atomic handoff — is **DynaExq**, arXiv:2511.15015 (Chu, Xiang, Shen, Yang, Lin, Zhang; UConn/UCSC, Nov 2025). It is a one-for-one match to the proposal, including hysteresis to prevent thrashing, a dedicated migration CUDA stream, and "stable expert handles" (pointer indirection so the forward pass never sees a partially-materialized tensor). Published nine months ago.
- **The hot-swap mechanism inside a live serving engine** — **MorphServe**, arXiv:2506.02006: pre-staged quantized layer variants in pinned CPU memory, written into pre-reserved GPU slots on async CUDA streams, **~6 ms per layer, fully overlapped with decode**. This is a direct, measured rebuttal to "vLLM loads weights once at startup, so runtime replacement needs heavy lifting."
- **The fragmentation and swap-reserve objections** — solved by DynaExq's partitioned fixed-block pools (constant-time freelist, blocks aligned to format granularity, no runtime `cudaMalloc`).
- **The offline "which tensor deserves the bits" problem** — thoroughly solved, and it is *not* an open research question. HAWQ → HAWQ-V2 → HAWQ-V3 established sensitivity-under-ILP. For MoE specifically: **MC-MoE** (arXiv:2410.06270, ICLR 2025) solves per-expert bit allocation as an LP over (activation frequency φ, routing weight w, reconstruction error ε) and reports **"the solution takes approximately one second to compute."** **BitsMoE** (arXiv:2606.00079), **GEMQ** (arXiv:2605.23078), **MoPEQ** (arXiv:2509.02512) refine it.
- **The premise that allocation requires BF16 activation measurement on rented B300s is false.** Three independent calibration-free/cheap signals exist: **AlphaQ** (arXiv:2606.04980) uses HT-SR weight spectra only; **router-norm allocation** (arXiv:2604.06515, RPI/IBM) "requires no GPU and negligible computation" and *beats* activation-frequency baselines; **KL-lens** (arXiv:2604.13440) is forward-only, no backprop, no Hessian.
- **Per-expert mixed precision at runtime** — HOBBIT (arXiv:2411.01433) / **MoE-APEX** (ASPLOS 2026, DOI 10.1145/3779212.3790187), DyMoE (arXiv:2603.19172), FlexQuant (arXiv:2501.07139), CXL-MoE (arXiv:2512.04476).

### What is genuinely novel

1. **Runtime precision reallocation under tensor parallelism.** Every published system is single-GPU (DynaExq: RTX 5090/A6000; MoE-APEX: llama.cpp/Jetson; MorphServe: single replica). DynaExq's own future work names distributed integration as undone. TP4+DCP4 with EP=1 is unexplored territory. This is real and defensible.
2. **Bit allocation for an MTP speculative-decoding draft head, with acceptance length as the objective.** Nothing in the literature addresses this. The `glm52-mtp78-collector` work is ahead of published research here.
3. **Long-horizon allocation stability.** No paper measures it — not even DynaExq, whose entire premise depends on it and which publishes no rank-correlation-across-windows, no hotness variance, and does not disclose its EMA α or period T. The 7.3M-token layer-78 trace is plausibly **the best dataset in existence** to answer this.
4. **Online re-encoding.** No system re-quantizes at runtime; all of them page pre-built variants. But this is novel because it is a *bad idea*, not because nobody thought of it (see §3). Drop it.

### Feasibility, stated plainly

The proposal as written — free per-tensor bpw, online re-encode from BF16, memory reclaimed on downgrade — is **not feasible on this stack**, for structural reasons: (a) all-K4 routed experts alone need **86.62 GiB/rank against a ~86 GiB budget** at `util=0.90`, so any max-capacity arena is dead on arrival; (b) EXL3 trellis codes are **not successively refinable**, so every bit-width change is a full re-encode requiring a Hessian that the conversion pipeline computes and then throws away; (c) the CUDA-graph pointer-baking constraint is enforced systemically throughout vLLM.

A **constrained version is feasible in weeks, not months**, and I verified the enabling fact directly in the fork's own code (§3).

### The judgment

**Build the offline half first, prove the value, and only then consider runtime.** The runtime rebalancer is the least valuable and most expensive component. The valuable, cheap, genuinely novel part is *workload-derived bit allocation applied at startup*, with the policy persisted and rehydrated. Two published results predict the runtime loop buys little on top: **VBQ** (arXiv:2607.02893) found learned allocation "stabilizes early enough to freeze into a fixed recipe," and **AWQ** (MLSys 2024) claims activation salience generalizes across domains *without* calibration overfitting. If those hold here, Phase 1 captures ~all the value and Phases 3–5 are wasted work.

**The go/no-go is not an engineering question.** It is: *does workload-specialized allocation beat generic allocation at equal bpw?* That is measurable offline in about a day with tooling exllamav3 already ships, and it must be answered before a line of runtime code is written.

---

## 2. THE DECOMPOSITION

Seven components, not six — the research consistently omitted the kernel one, which is the first thing a vLLM reviewer will raise.

| # | Component | Difficulty | Independently useful? | Home |
|---|---|---|---|---|
| a | Online quantization at startup + local disk cache | **Easy** (half already exists) | **Yes, to every vLLM user** | vLLM core |
| b | Statistics collection (per-expert + per-tensor) | **Easy** (per-expert) / Moderate (per-tensor) | Yes | Plugin now, vLLM core later |
| c | Bit-allocation decision policy | **Moderate** — solved offline, unsolved online | Yes, standalone | exllamav3 / llm-compressor |
| d | Runtime hot-swap | **Hard** | Only with (c) | Fork + b12x |
| e | JIT streaming of variants from HF | **Easy** | Marginally | Plugin |
| f | Persistence / rehydration | **Easy** (precedent exists in-fork) | Yes | Plugin, then vLLM core |
| g | **Mixed-bit-width MoE kernel** | **Hard, and already shipped in this fork** | Yes | b12x / SparkInfer |

### (a) Online quantization at startup + local disk cache — *half of this is already built upstream*

Online quantization **already exists** as a first-class, memory-lean, layer-at-a-time meta-device path: `QuantizeMethodBase.uses_meta_device` (`vllm/model_executor/layers/quantization/base_config.py:21-24`), `Fp8OnlineLinearMethod` (`fp8.py:519-523`, JIT materialize `:544-559`, quantize-on-complete `:573-579`), and **`Mxfp8OnlineMoEMethod` (`mxfp8.py:224-227`)** — the user's proposed "online MXFP8 for MoE at startup" is not hypothetical, it ships today. It is even designed to **re-run at runtime**: the reload path deliberately deletes the `_already_called_process_weights_after_loading` guard (`reload/layerwise.py:219-222`) so online quantization re-executes on freshly loaded BF16 weights.

**What is missing is exactly the cache.** Verified: nothing under `vllm/model_executor/` writes quantized results to disk; the only weight-writing paths are the operator-invoked `ShardedStateLoader.save_model` and a `.bin`→safetensors converter. Every restart re-quantizes from scratch (cf. issue #46611, 60 s broadcast-timeout reports where vLLM's own error message names weight/KV-cache quantization among likely causes).

This is the single best first upstream contribution: small, uncontroversial, benefits every FP8/MXFP8 user, requires no RFC, and builds the exact infrastructure the policy artifact needs.

### (b) Statistics collection

**Per-expert is nearly free.** `BaseRouter.set_capture_fn` (`base_router.py:132-134`, invoked unconditionally at `:239-241` *before* EPLB remap) is an **ungated** per-layer hook, independent of `enable_eplb` and of expert parallelism. Bind it by walking `compilation_config.static_forward_context` exactly as `gpu_model_runner.py:6583-6596` does.

**Two traps.** (1) MoE ops are *not* CUDA-graph splitting ops (`compilation/config.py:1006`, `splitting_ops = list(self._attention_ops)`), so the router runs **inside** captured graphs. A callback containing `.item()`, host branching, list appends, or per-step Python constants **silently freezes at capture time and never updates again.** It must be a pure tensor op into a persistent buffer — which is precisely why EPLB's recorder is a single `scatter_add_` of ones (`base_router.py:79-86`). (2) The in-tree `RoutedExpertsCapturer` is unavailable here: `enable_return_routed_experts` is hard-asserted off under DCP>1 (`scheduler.py:249-253`), which DCP4 trips. The *hook* is not gated; only the shm/slot-mapped return path is, and the policy does not need per-token attribution.

**Per-tensor (dense) is harder.** Dense `Linear` layers are not in `static_forward_context` — only attention, MoE, and mamba register there. Enumerate via `model.named_modules()` filtered on `LinearBase` (which also gives you stable dotted names that double as topology-neutral policy keys). Instrumentation goes through `PluggableLayer.register_oot` (`custom_op.py:83-100`); `ColumnParallelLinear`/`RowParallelLinear`/`ReplicatedLinear` are all `PluggableLayer`s. `FusedMoE` is **not**, hence the `set_capture_fn` route for MoE.

There is **no general activation-statistics infrastructure** in vLLM — the only `register_forward_hook` in the tree is the NVTX tracer, explicitly documented as broken under CUDA graphs and under `STOCK_TORCH_COMPILE`. Do not use nn.Module hooks.

### (c) The bit-allocation policy

This is where the intellectual value lives, and **it does not belong in vLLM.** It belongs in exllamav3's conversion toolchain (or, for upstream credibility, `vllm-project/llm-compressor`). exllamav3 already ships `measure_model.py` (per-tensor KL-delta curves at `-l3`), `optimize_model.py` (greedy knapsack maximizing `1e10·dkld/(dbits+1)` with concavity correction `-((-dkld)**0.69)`), and `compile.py` (assembles a mixed checkpoint by **selecting** tensors from pre-quantized directories — *no re-encoding*). The workload term is the only thing missing.

### (d) Runtime hot-swap — see §3. Fork + b12x. This is the expensive part and the least justified.

### (e) JIT streaming of bit-width variants from HF

Nearly free: safetensors' 8-byte header-length + JSON header gives per-tensor byte offsets, so a single named tensor is an HTTP range read; vLLM already uses `safe_open` + `get_tensor` per tensor (`weight_utils.py:854-859`). **One negative finding:** HF Xet content-defined chunking will not dedupe across bit-widths — re-encoding at a different K changes every byte, so a multi-bpw repo costs the full sum of its variants, not less.

### (f) Persistence / rehydration — *the artifact format already exists in this fork*

Verified: the fork already loads a per-expert bitrate map from a JSON reference `"file.json:field"` in the HF repo, keyed by layer index, one integer per expert per layer, with `k_values` restricted to `{3,4,5,6}` (`exl3.py:_load_rank_sliced_bitrates`, ~`:507-556`). **This *is* the policy artifact.** It is machine-readable, regenerable, and already the input to the mixed-tier path. That eliminates a whole design question.

Two caveats. The metadata schema restricts K to 3..6, so **K2 downgrade is not even representable today** — the budget-freeing half of the proposal is capped at "don't upgrade." And the checkpoint declares `tp` and uses `tensor_schema = "model.layers.{L}.mlp.experts.{E}.{proj}.rank{r}.{trellis|suh|svh|mcg}"`, hard-failing when `checkpoint_tp != runtime tp` — the storage is topology-baked (see §4).

For the cache side, imitate `torch_compile_cache`: `VLLM_CACHE_ROOT/…/{hash}` with the key built by a `compute_hash()` that puts `parallel_config` in `ignored_factors` (`config/compilation.py:697-727` is the pattern). The fork's `startup_plan.py` is a working precedent for fingerprinted, persisted, boot-rehydrated state — but its fingerprint includes `rank` and `world_size` and must not be copied verbatim.

### (g) Mixed-bit-width MoE kernel — *already solved here, and this is a genuine asset*

Upstream vLLM **cannot run a statically mixed-bit MoE layer**: issue #41955, "Mixed INT4/INT8 GPTQ MoE models crash on initialization (AssertionError in fused_marlin_moe)" — fused kernels assume one bit-width per layer. MxMoE (arXiv:2505.05799) shows autogenerated mixed-precision GroupGEMM is worth 29.4% over uniform at matched accuracy. The GG fork's b12x two-tier `mixed_trellis` megakernel already does this. That is a real, differentiated capability and the strongest thing to lead an RFC with.

---

## 3. THE HARD PART, SOLVED OR NOT

### The constraint, stated precisely

A captured CUDA graph bakes device pointers into kernel launch parameters. vLLM's replay-time address check (`compilation/cuda_graph.py:276-279`, `:338-347`) covers **only positional tensor args, never module parameters**, and is gated on `VLLM_LOGGING_LEVEL == "DEBUG"` (`:191`) — i.e. **off in production**. A weight replaced at a new address therefore yields silent wrong output, not an exception. This was independently confirmed by three feasibility lenses, and vLLM itself concedes it in a code comment: `reload/layerwise.py:240` — *"# Copy processed values into original tensor storage (preserves cudagraph refs)"*.

So every runtime weight-mutation path in vLLM is same-shape `copy_` into pre-existing storage: EPLB (`rebalance_execute.py:251, 427, 465`), layerwise reload (`layerwise.py:244`), `update_weights` (`gpu_worker.py:1006`), `reload_weights` (`gpu_model_runner.py:4746`). Note also that `copy_` **silently value-casts on dtype mismatch** — it fails loudly on shape and quietly on dtype.

Worse, at the data-structure level: `FusedMoE.get_expert_weights()` returns `weight.view(self.local_num_experts, -1)` with a contiguity assert (verified at `fused_moe/layer.py:1430-1449`). Experts are **rows of one tensor with uniform stride**. Per-expert heterogeneous bit-width is not "hard" in this layout — it is unrepresentable. EPLB's buffer is `torch.empty_like(w)` under the note *"we assume the same weights across different layers have the same shape"* (verified at `rebalance_execute.py:604-610`).

### The five candidates

**1. Fixed-capacity arena sized for K_max, under-filled for lower K — REJECT.**
This is the workaround three independent feasibility lenses converged on (it is LoRA's `max_lora_rank` pattern: `lora/layers/base_linear.py:65-85`, `copy_` into a padded prefix at `:110-120`). **The arithmetic kills it.** All-K4 routed experts = 256 experts × 77 layers × 4.5 MiB = **86.62 GiB/rank**, against a ~86 GiB budget at `util=0.90` on a 96 GiB card — before dense/attention weights, the MTP draft, the ~1 GiB Trellis arena, activations, NCCL, CUDA graph pool, or a single KV block. It does not fit. And even if it did, reserving K4 capacity costs the same memory as simply quantizing at K4, so the scheme would buy nothing.

**2. Nested / prefix codes — REJECT for EXL3 as it stands; keep as the best long-term research direction.**
EXL3 is not successively refinable, and this is provable from source, not inferred. Each weight decodes from a **16-bit sliding window** of the tile bitstream; only the *stride* changes with K (`exl3_dq.cuh:16-31`). Reading a K4 stream at stride 3 produces entirely different windows, and the codebook is a multiply-and-bit-twiddle hash (`x *= 0xCBAC1FED`, `codebook.cuh:67-75`) with zero locality — so truncation yields **uncorrelated noise, not a coarse approximation**. Additionally `suh`/`svh` change value on every K change, because the global scale is found by test-quantizing at the target K (`quantize.py:1045-1071, 1215-1220`). Classical coding theory agrees: embedded TCQ exists precisely because plain TCQ is not refinable (US Patent 6,125,149).

Every free-truncation scheme in the literature relies on structure EXL3 lacks: **MatQuant** (arXiv:2502.06786, ICML 2025; oral at ICLR 2025 SLLM workshop) on integer MSB nesting *plus co-training*; **Any-Precision LLM** (arXiv:2402.10517, ICML 2024) on bit-plane layout *plus a clustering backbone* (its Table 2 shows uniform quantizers give INF perplexity under incremental upscaling).

**But the additive-residual family does not require MSB structure.** **RRQ** (arXiv:2608.04048) explicitly decouples stages so "the low-bit foundation and the residual stages can be constructed with different quantizers" and existing low-bit checkpoints are reusable as the base; **Drop-by-Drop** (arXiv:2606.12876) proves Gaussian sources are successively refinable under weighted MSE and needs no re-encoding to drop a codebook; **AnyBCQ** (arXiv:2510.10467) is cheaper at runtime than Any-Precision LLM. A K3 trellis base plus a cheap additive residual plane would make upgrade = *materialize one plane*, downgrade = *free it*, at constant base-tensor size and O(1) runtime cost. **That is the single highest-leverage change available, and it belongs in exllamav3, not vLLM.** It is out of scope for v1 but should be on the roadmap.

**3. VMM remapping via CuMemAllocator — REJECT for v1 on two independent grounds.**
The mechanism is real and shipped: `sleep()` calls `cuMemUnmap` + `cuMemRelease` but deliberately **not** `cuMemAddressFree` (that appears only in `my_free`, `csrc/cumem_allocator.cpp:528`), and `wake_up()` re-maps fresh physical pages at the **same `d_mem`** (`cumem.py:238-241`). Stable-VA-with-swapped-physical-backing is production capability.

It fails here for two reasons nobody connected. (i) **Granularity.** CUDA minimum allocation granularity is the 2 MiB class (`cumem_allocator.cpp:313-319`), and the per-expert-per-layer K3→K4 delta is **1.125 MiB/rank — below one granule.** Per-expert remapping reclaims nothing. You would need ≥2 experts bundled per granule, and the reservation size is frozen at malloc (`:319-323`) so growth requires patching the allocator to over-reserve VA. (ii) **It is broken on this exact hardware:** issue #21336, *"vLLM crashes when using --enable-sleep-mode with Blackwell PRO 6000 GPUs"*, **open since 2025-07-21**, plus #48680 (sleep-mode OOM on NVFP4/SM120 from cumem MemPool overhead).

**4. Double-buffer + graph re-capture — REJECT for periodic use; retain as a rare maintenance path.**
Re-capture is possible — elastic EP does it: clear entries, `torch.compiler.reset()`, `reset_compile_wrapper`, `unlock_workspace()`, `compile_or_warm_up_model()`, `lock_workspace()` (`elastic_ep/elastic_execute.py:389-414`). Cost: **5–20 s** for capture alone (`gpu_model_runner.py:5768-5775`) plus a full Inductor recompile, plus prefix-cache destruction. Not a rebalance-interval operation. *(Correction to one input claim: capture is not literally once-per-process — `profile_cudagraph_memory()` at `gpu_worker.py:398-399` already performs a full capture→discard→re-capture cycle on every non-eager boot, and `set_cudagraph_capturing_enabled` is a plain toggle, `monitor.py:98-100`. The guard is a tripwire, not a lock. The cost, not the guard, is what rules this out.)*

**5. Fixed-cardinality tiers + in-place membership swap at a quiesced point — SELECT.**

### The chosen design, and why it works

**I verified the enabling fact directly in the fork's source.** In `_prepare_mixed_rank_sliced_weights` (`exl3.py:1572`) and `_apply_mixed_rank_sliced` (`:1937`):

- `api.compile_mixed_trellis(...)` (`:1885-1901`) takes **`tier0_num_experts`, `tier1_num_experts`, `tier0_bits`, `tier1_bits`** — i.e. **cardinalities and bit-widths are baked into the compiled launch.**
- `api.run_mixed_trellis(x, tier0, tier1, topk_weights, topk_ids, global_to_combined, descriptor_map, rotations, launch, buffers)` (`:1955`) takes **`global_to_combined` and `descriptor_map` as runtime device-tensor arguments on every call.** These are produced by `api.build_tiered_maps(tier_ids[0], tier_ids[1])` (`:1689`) purely from the two lists of global expert IDs.

**Tier membership is runtime data. Tier cardinality is compiled state.** So:

> **Hold the per-layer tier cardinalities fixed at startup (|K4| = N_L, |K3| = E − N_L) and express every rebalance as a membership permutation.**

Consequences, each of which dissolves a stated blocker:

| Blocker | Resolution under fixed cardinality |
|---|---|
| Tensor size changes | Tier slab shapes `(2, n_tier, H/16, I_pp/16, 16·bits)` and `(n_tier, I_pp/16, H/16, 16·bits)` (verified `:1631-1645`) are **constant forever**. Every write is a fixed-shape row `copy_` — exactly what EPLB already does safely. |
| CUDA-graph pointer invalidation | No parameter is ever rebound. Same discipline as `layerwise.py:240-246`. |
| `tier_signature` cache miss → b12x recompile + new ~1 GiB arena + leak in the never-evicting `_MIXED_TRELLIS_RUNTIMES` dict | `tier_signature` = `((3, E−N), (4, N))` is **invariant** (`:1840-1856`), so the cache hits forever. The `RuntimeError("Mixed-bitrate EXL3 runtime must be compiled during the eager profile pass before CUDA graph capture")` at `:1862` is never reached. |
| Memory reserved for atomic swap | Bounded and small (below). |
| Fragmentation | Zero. No allocation occurs on the swap path. |
| Online re-encode cost + Hessian persistence (~2.35 GiB/layer, ~210 GiB model-wide) | **Eliminated.** Page pre-built K3/K4 artifacts, as every published system does. |

### Memory arithmetic (H=6144, E=256, top-8, I=2048, 77 MoE layers, TP4)

Per-expert-per-layer-per-rank bytes = `3·H·I/8/TP · bits` = **1,179,648 × bits**.

| Quantity | K3 | K4 | Δ |
|---|---|---|---|
| One expert, one layer, one rank | 3.375 MiB | 4.500 MiB | 1.125 MiB |
| All routed experts, per rank | 64.97 GiB | 86.62 GiB | 21.66 GiB |
| One routed bpw point (+0.01) | — | — | 221.8 MiB/rank |

At the current 3.42 bpw routed operating point: **N = 108 experts/layer at K4, 148 at K3**, total **74.07 GiB/rank** — leaving ~12 GiB/rank for dense/attention, MTP draft, the ~1 GiB Trellis arena, activations, NCCL, graphs and KV. A 108-slot swappable population per layer is more than enough for the policy to matter.

**Swap cost.** One expert exchanged per layer = read 4.5 MiB (incoming K4) + 3.375 MiB (outgoing K3) = **7.875 MiB/rank/layer**; ~48 KiB of `suh`/`svh` rows is noise. A full-model pass over all 77 layers moves **~606 MiB/rank** — ~11–24 ms of H2D on PCIe Gen5/Gen4, ~87 ms from NVMe at 7 GB/s. Fully overlappable on a side stream. For reference, MorphServe measured ~6 ms per layer swap on PCIe Gen4.

**Reserved staging.** Best case: one expert double-buffered = **9 MiB/rank**. Reserve it at profile time using EPLB's exact trick — allocate inside `profile_run` under an `is_profile` flag so `memory_profiling` sees it and the KV budget shrinks honestly (`rebalance_execute.py:611-621`).

**Offline storage.** Full K3 + full K4 routed-expert artifacts ≈ 260 + 347 = **~607 GB on NVMe.** Trivial. This buys away Viterbi encode cost, Hessian persistence, stale-Hessian quality risk, and the entire online-quantization subsystem.

### The one remaining unknown, now narrowed

`api.prepare_weights(..., w13_layout="trellis3_t256_proj", ...)` (`:1675-1687`) probably **repacks** the stacked slab into an mma-friendly permuted layout, and the source tensors are then freed (`param.exl3_tensors.clear(); param.exl3_backing = None`, `:1704-1709`). Whether b12x exposes a per-expert-row write into a prepared tier object is not visible from this repo.

**The design degrades gracefully, which is why I am willing to select it before that answer arrives:**

- **Best case** — per-expert row write exposed: swap = two row `copy_`s + rotation-row updates + `build_tiered_maps` into the existing `global_to_combined`/`descriptor_map` tensors in place. Sub-millisecond of GPU work. Reserved memory 9 MiB/rank.
- **Fallback** — only whole-tier `prepare_weights` available: batch **all** swaps for one layer and rebuild that layer's two tier slabs. At 3.42 bpw that is 108×4.5 + 148×3.375 = **~0.96 GiB/rank**, ~1.92 GiB double-buffered, **one layer per engine step** — EPLB's exact cadence (`async_worker.py:118-186`; `eplb_state.py:619-631`). Bounded regardless of how many experts move. Cardinality still fixed, so still no recompile and no arena change.

Both variants are viable. Answering the b12x question chooses between 9 MiB and 2 GiB of reserved memory — it does not change whether the design works.

### Quiesce

Apply metadata updates between engine steps. vLLM already ships the primitive: `pause_scheduler` with `mode ∈ {abort, wait, keep}` (`entrypoints/serve/rlhf/api_router.py:30-72`; `v1/engine/core.py:643-679`). `keep` freezes requests in queue rather than dropping them.

### Failure modes of the chosen design

1. **Torn update** between the trellis rows and `suh`/`svh`/`rotations` produces silent corruption with no error. Mitigation: perform the entire metadata update inside the quiesce window, or double-buffer the map tensors and flip one index.
2. **Strict permutation.** Every upgrade costs a downgrade. This is a real expressiveness loss — and also a feature: the memory budget becomes a hard structural invariant rather than a checked constraint.
3. **Two-tier ceiling.** `len(tiers) != 2` raises (`:1591`), and the metadata schema restricts `k_values` to 3..6. K2 is unavailable. The proposal's downgrade path is capped at "stay K3."
4. **Cross-rank divergence is fatal.** The tier map must be bit-identical on all 4 ranks. A per-logical-expert policy applied uniformly gives this for free. Note also that `num_gpu_blocks` is unified downward across workers (`kv_cache_utils.py:1527-1528`), so any asymmetry would penalize every GPU's KV cache.
5. **`FusedMoEQuantConfig` staleness.** It is memoized on the quant method and captures direct tensor references (`fused_moe/layer.py:1468-1479`). Any swap must null it to force rebuild.
6. **Cardinality is a restart-level decision.** Changing N_L requires a `compile_mixed_trellis` + fresh arena, refused under capture. Treat as a maintenance operation, not a rebalance.

---

## 4. THE POLICY PROBLEM

### Signal — three terms, all cheap, all topology-neutral

**Do not use raw hotness.** Two independent papers found it is the *weakest* available signal: the RPI/IBM router-norm method (arXiv:2604.06515) explicitly **beats** activation-frequency and activation-weight baselines, and MoPEQ uses Hessian trace "instead of relying on the activation frequency of the expert." vLLM's EPLB counter records literally nothing else — `scatter_add_` of `torch.ones_like` (`base_router.py:79-86`).

1. **Static error curve ε_e(b)** — reconstruction/KL error of quantizing expert e at b bits. Pure weight-space; computable locally with no rented hardware. This is MC-MoE's ε and BitsMoE's E(b) (piecewise: `η·exp(−λb)` for high bits, empirical κ_b for 2–4 bits). Free additional priors requiring no GPU: AlphaQ's HT-SR spectral α, and router-vector L2 norm.
2. **Workload term** — routing **mass** w_e = Σ gate values, plus frequency φ_e, over the rolling window. Use gate mass, not counts. HOBBIT's (arXiv:2411.01433) cumulative gate-norm criticality (terminology per paper body — verify PDF before quoting) (`s(x) = Σ‖G(x)_{e_j}‖` over the token's top-K, thresholds 0.6/0.9) is the refined form, and its **LHU** metric ("how often was this expert needed *at high precision*") is a strictly better promotion signal than raw usage.
3. **Verification term** — measured end-to-end KL against a held-out probe, obtained by actually applying a candidate swap and measuring. **This is the one thing a live server can do that an offline quantizer cannot**, and it is the answer to EvoPress.

**The objective must be KL, never perplexity.** TASA (arXiv:2607.00908) shows PPL-based layer sensitivity has **Kendall τ ≈ 0** with reasoning-task sensitivity, and that PPL-sensitive layers cluster at embeddings/output-head while reasoning-critical layers sit mid-network — so PPL-guided allocation *systematically under-protects reasoning*. llama.cpp discussion #4110 reaches the same conclusion from practice. exllamav3's own `measure_model.py` already uses KL.

### Objective and algorithm

MC-MoE's LP verbatim, ~1 s solve:

> minimize Σ_e Σ_b φ_e^α · w_e^β · (ε_{e,b} · x_{e,b})^γ  s.t. Σ b·x_{e,b} ≤ B, Σ_b x_{e,b} = 1

**Under fixed cardinality this degenerates to a sort, not an LP.** With exactly two tiers and N_L fixed, rank experts by marginal utility

> Δ_e = (ε_{e,3} − ε_{e,4}) · φ_e^α · w_e^β

and take the top N_L. That is `argsort`. The LP only returns if cardinality is ever allowed to float across layers under a global budget — which GEMQ argues is worth doing (global allocation across all 256×77 experts beats per-layer), and which arXiv:2511.05814 supports (middle layers are most skewed) and OLMoE supports (specialization strongest in later layers). Reconcile by choosing the per-layer N_L **once at startup** from a global solve, and letting the online loop only permute membership within each layer.

**Two corrections that the naive design gets wrong:**

- **EvoPress** (arXiv:2410.14649, ICML 2025) shows per-layer error is **not monotone or additive** — "pruning a model further may even significantly recover performance." Every greedy/knapsack allocator assumes exactly the separability this refutes. Practical consequence: small deltas per interval (cap at 1–4 swaps per layer), and validate against the held-out KL probe before committing, with rollback.
- **TASA** shows calibrating *purely* on in-domain data actively degrades quality — pure task data makes the layer-wise Hessian eigenspectrum degenerate. Optimal mixing is model-specific: **50% general for LLaMA-3, 75% for Qwen2.5**. So: derive ε from a **blended** calibration set; use live traffic only for the φ/w weighting. "Specialize" means *reweight*, not *replace*.

### Topology neutrality

Two artifacts, deliberately separated:

- **Policy artifact (neutral).** Map `logical expert id → bits`, keyed on `(checkpoint identity/revision, base quant config, policy schema version, bit menu)`. **Explicitly exclude** rank, world_size, tp_size, device count, device name. The fork's `bits_per_expert` JSON already has exactly this shape.
- **Derived slab cache (topology-keyed, regenerable).** Per-rank materialization.

Both existing precedents violate this and must not be copied verbatim: `startup_plan.compute_plan_fingerprint` includes `rank` and `world_size`; `VllmConfig.compute_hash()` folds in `parallel_config` (`config/vllm.py:370`); and the checkpoint carries `rank{r}` in tensor names with a hard failure on `checkpoint_tp != runtime tp`.

**The rank-slicing is a storage choice, not a format constraint.** The underlying EXL3 trellis slices cleanly at **16-column granularity** because every 16×16 tile is a self-contained tail-biting codeword (`exllamav3/modules/quant/exl3.py:303, 310, 362`). Storing unsharded and slicing at load restores topology neutrality; the rank-pre-sharding buys load-time convenience only. Whether that convenience is load-bearing here was not established and should be checked before proposing a format change.

Convenient accident: because EP is **off**, every TP rank routes the same tokens, so per-expert counts are **logical, not physical**, and therefore already topology-neutral. Under EP they would be physical and would not be. (An all-reduce over the EP group — which exists for any MoE model regardless of `enable_expert_parallel`, `parallel_state.py:1652-1681` — would uniformly multiply counts by tp_size, preserving order; simpler to just read rank 0.)

### Feedback loops

The loop **precision change → expert output change → router logit shift → top-8 membership flip → measured hotness change → different precision decision** is real and **completely unstudied**. Four groups establish that quantization shifts routing (EAQuant arXiv:2506.13329; MoEQuant arXiv:2505.03804; ExpertQuant/Rank-Aware PTQ, OpenReview kPgLp47bJf; EAC-MoE arXiv:2508.01625). GEMQ names "router shifts induced by quantization" as a first-class failure and fixes it by fine-tuning the router afterward. ExpertQuant finds errors are dominated by **near-neighbor rank flips around top-k** — and with top-8 of 256 fine-grained experts, the 8th/9th margin is thin. No published work connects this to a runtime loop; there is zero analysis of oscillation, hysteresis requirements, or convergence.

Mitigations, in cost order:

1. **Keep the router in BF16.** DynaExq keeps all routing components full-precision; this stack already does.
2. **Rollback guard.** Measure top-8 **Jaccard overlap** before/after a swap against the BF16 ground-truth IDs. The collector captures exactly this. Revert if overlap drops below threshold.
3. **Hysteresis.** An expert must clear a wider band to be promoted than to be retained (DynaExq's candidate-set widening; CXL-MoE's sustained-below-threshold rule).
4. **Dwell time.** Minimum residency in a tier, in engine steps, before eligibility to move.
5. **Damping.** Hard cap on swaps per interval.

### Cold start

**Never start from uniform.** Ship a generic allocation from the offline toolchain as the prior; rehydrate the previous session's policy at boot and treat it as the starting point; the online loop only ever perturbs it. Given the fixed-cardinality design, cold start is literally "load the JSON that already exists."

### How this differs from EPLB

| | EPLB | Fungible quant |
|---|---|---|
| Problem | Load balancing | Distortion minimization |
| Objective | Minimize max-rank load → throughput | Minimize aggregate output error → quality |
| Domain | **Physical** expert slots | **Logical** experts |
| Free variable | Placement + **replication** (`num_redundant_experts`) | Bit-width |
| Output | Same-shape permutation | Same-shape permutation *within a tier* |
| Wants | **Uniformity** | To **exploit** non-uniformity |
| Constraint | Ranks × slots | Bytes |

`balanced_packing` and `replicate_experts` are not reusable — there is no replication here, and a perfectly balanced expert distribution is precisely the case where fungible quant has nothing to do. **The only shared machinery is the cadence**: rolling window, `step_interval` trigger, async worker with its own stream and events, and the `is_profile` memory-reservation trick. That is worth reusing verbatim, and factoring it out is itself an upstreamable contribution.

---

## 5. PHASED PLAN

### Phase 0 — This week. No vLLM changes. Own hardware. *This is the go/no-go.*

| # | Task | Effort | Proves / retires |
|---|---|---|---|
| **0a** | Stability analysis on the existing 7.3M-token layer-78 trace. Bucket into ~100 windows; compute per-expert frequency **and gate mass** per window; report Kendall τ of the top-N set across windows, across hours, across days. | Pandas script, hours, **zero GPU** | The load-bearing empirical assumption nobody in the literature has measured. Directly contradicts or confirms MoE-Infinity's "reuse counts even out over time." |
| **0b** | Extend the collector to 3–4 additional MoE layers (early/middle/late) for a short run; rank-correlate against layer 78. | 1 day | That the layer-78 asset generalizes. Layer 78 is the **MTP draft head** — one layer of ~78, and a draft head at that. |
| **0c** | `measure_model.py -l3` with the stock calibration mix → per-tensor dKL-vs-bpw curves. Report the **variance of dKL across experts within a layer**. | 2–5 h GPU | Whether there is anything to allocate at all. If dKL is homogeneous, mixed precision degenerates to uniform and the project ends. |
| **0d** | **THE GO/NO-GO.** Run measure→optimize→compile twice at *identical* target bpw: stock calibration vs a 50/50 blend with live traffic (per TASA). Compare end-task quality and Kendall τ between the two allocations. | +1 day | Whether workload specialization is worth anything. Directly tests VBQ's "freeze the recipe" and AWQ's cross-domain generalization claims. |
| **0e** | Router-shift measurement: run one layer at K3 vs K4, measure top-8 Jaccard against the BF16 ground truth the collector captures. | Hours | The feedback-loop risk. |
| **0f** | Six cheap facts: (i) read b12x `mixed_trellis` — does `prepare_weights` alias or repack, and is a per-expert row write exposed? (ii) benchmark `quantize_exl3()` on one 5120×1536 tensor at K3 and K4 on an idle Blackwell; (iii) `gh issue view --comments` on #50281, #38256, #49198, #48920; (iv) search `vllm-project/llm-compressor` and `compressed-tensors`; (v) search the **vLLM developer forum, discuss.vllm.ai** (never searched — AGENTS.md duplicate check is unsatisfied without it; GitHub Discussions was deprecated Mar 2025) and the **exllamav3 issue tracker**; (vi) **comment on RFC #49702 before 2026-08-18.** | 1 day | Design cost, prior art, and the highest-leverage upstream action currently available. |

**Do not write runtime code until 0d returns.**

### Phase 1 — 2–3 weeks. Plugin only. Startup specialization.

**Deliverable.** A `vllm.general_plugins` entry point that (i) binds `BaseRouter.set_capture_fn` with a single `scatter_add_` into a persistent device buffer, (ii) persists per-expert (count, gate-mass) to `$VLLM_CACHE_ROOT` under a topology-neutral key, and (iii) feeds those statistics into `measure_model.py`/`optimize_model.py` as the φ/w term to emit a specialized `bits_per_expert` JSON and a compiled mixed-K checkpoint.

**Mandatory validation.** Bind the hook, run 100 decode steps **with CUDA graphs enabled**, confirm the counter scales linearly with steps rather than freezing at its capture-time value.

**Proves.** End-to-end specialization value with zero runtime risk. **If VBQ's finding holds, this is the whole product** and Phases 3–5 are never built. **Retires:** every mechanical risk, by not taking any.

### Phase 2 — 1–2 weeks. First upstream PR. No RFC needed.

**Deliverable.** Cache online-quantized weights to disk. Key from a `compute_hash()` with `parallel_config` in `ignored_factors`, imitating `torch_compile_cache` (`compilation/backends.py:962`; `config/compilation.py:697-727`). Store under `VLLM_CACHE_ROOT/quant_cache/{hash}`.

**Why first.** Small, uncontroversial, benefits every FP8/MXFP8 online-quant user today (cf. #46611, #48035 — timeout reports where the error message names quantization as a likely cause), requires no new concepts, and builds the exact infrastructure the policy artifact needs. Best possible way to establish credibility with the quant maintainers before proposing anything ambitious.

### Phase 3 — 2–4 weeks. Second upstream PR.

**Deliverable.** Decouple per-expert load recording from expert parallelism. Today `parallel.py:394-395` raises `"enable_expert_parallel must be True to use EPLB"`, yet the recorder needs nothing from EP and the `_EP` group exists for any MoE model regardless. Ship a standalone per-expert load window + `step_interval` trigger usable at EP=1, plus **the first Prometheus metrics for expert load** (there are currently zero — only a `logger.info` gated on `log_balancedness`), routed through the `ModelRunnerOutput → SchedulerStats → PrometheusStatLogger` pipeline that `cudagraph_stats` already exemplifies.

**Independently useful** to anyone who wants expert-distribution visibility without turning on EP. **Retires** the "EPLB is structurally unreachable in my config" blocker. Coordinate with RFC #32028 (EPLB refactor, ilmarkov) and RFC #49702.

### Phase 4 — 4–8 weeks. Fork + b12x. Only if 0a/0d/0e all pass.

**Deliverable.** Fixed-cardinality membership swap: pre-built K3/K4 artifacts on NVMe → pinned host staging → side CUDA stream → row writes into fixed-shape tier slabs → `build_tiered_maps` into existing device tensors → applied at a quiesced point.

**Start with the brutal version:** `sleep(level=2)` → reload with the new policy → `wake_up`. It drops in-flight requests and costs seconds, but it validates the entire decide-and-apply loop end-to-end before anyone builds the atomic path. Ship that first, always.

### Phase 5 — RFC. Quant-format-agnostic runtime precision reallocation.

Filed only with Phase 0's measurements as the evidence base. An RFC carrying a stability curve and a specialization-value number is a fundamentally different document from one carrying a mechanism sketch.

---

## 6. UPSTREAMING STRATEGY

### Duplicate-work check (AGENTS.md §1)

**No RFC or PR in `vllm-project/vllm` proposes workload-driven runtime bit-width reallocation.** Exhaustive searches across issues and PRs for requantization (106 PR hits), "adaptive quantization"/"adaptive precision" (14), "bit width"/"bpw"/"bits per weight" (20), per-expert + precision (116) returned nothing describing periodic re-decision of per-tensor bit-widths at runtime under a memory budget.

**Caveat that must be closed before filing: the vLLM developer forum (discuss.vllm.ai) was never searched.** (GitHub Discussions was deprecated in Mar 2025 in favor of the forum; pre-deprecation design threads like Discussion #5802 exist but Discussions is no longer the active venue.) Also unsearched: `vllm-project/llm-compressor` and `vllm-project/compressed-tensors`, which is where bit-allocation logic and any per-expert mixed-bit on-disk schema would actually live. If compressed-tensors already has a per-expert bitrate schema, both the novelty claim and the format design change. Run these before filing.

Required commands per AGENTS.md:
```bash
gh pr list --repo vllm-project/vllm --state open --search "online quantization"
gh pr list --repo vllm-project/vllm --state open --search "EPLB rearrange"
gh issue view 50281 --repo vllm-project/vllm --comments
gh issue view 38256 --repo vllm-project/vllm --comments
gh issue view 49198 --repo vllm-project/vllm --comments
gh issue view 48920 --repo vllm-project/vllm --comments
```

### Positioning against the seven neighbours

| Item | Relationship | Action |
|---|---|---|
| **RFC #50281** + PRs #50401, #51285, #51392 (per-layer online quant config; RFC by fxmarty-amd, PR #50401 by Ganeshkusalkar, #51285/#51392 by fxmarty-amd) | **The substrate.** Establishes the layer-pattern → scheme vocabulary. | Frame fungible quant as *"make #50281's static map mutable at runtime and let vLLM author it from observed load."* **Do not add a third PR to that surface** — two already race. |
| **RFC #38256** + PR #37190 (incremental MoE expert offloading, e1n00r, ~980 LOC in review) | **Best potential ally.** Already builds per-expert LFRU hotness + an async H2D pipeline, explicitly *"quant-agnostic,"* cache stores opaque blobs keyed by name. | Propose composition, CC e1n00r. Its existence also independently weakens the "vLLM loads weights once" objection. |
| **#49198** (progressive mixed-precision KV cache) | Same four-step lifecycle (observe → async requantize → atomic remap → free), for KV blocks. | Cite for mechanism acceptability; borrow its logical→physical indirection. No maintainer buy-in yet — precedent for design, not for acceptance. |
| **RFC #48920** + PR #48908 (unify weight-loading lifecycle, aoshen02; PR now closed unmerged — engage on the RFC) | **The boundary.** Explicitly scopes EPLB rebalance and non-checkpoint shape changes OUT; its stated scope excludes ops that "do not replace the base checkpoint" (LoRA add/remove, EPLB rebalance, sleep/wake, KV-cache ops). | A reviewer *will* point here. Engage during the open feedback window; propose fungible quant as a **fourth `WeightLoadSession` source**, not a bypass — it *does* replace base-checkpoint content, so unlike the excluded ops it belongs inside the lifecycle. Engage before the scope hardens into merged code. |
| **RFC #49702** (EPLB platform backend, freyfwt) — **feedback closes 2026-08-18** | **Highest-leverage action available right now.** Only open proposal abstracting *"allocation and movement operations for weight tensors"* behind a replaceable interface, and it already contemplates per-expert tensor sequences. | Comment asking that the interface **describe sizes** rather than assume `empty_like`. Cost: 30 minutes. |
| **#51567** (EPLB fails to transport E8M0 expert state) | Proof the fixed-shape/typed-tensor gap causes **user-visible breakage today**. | Cite to reframe the proposal from speculative feature to general fix for a class of live bugs. Its `tensor.view(torch.uint8)` fix is the right primitive for variable-width payloads. |
| **#41955** (mixed INT4/INT8 GPTQ MoE crashes in `fused_marlin_moe`) | **The likely first objection.** Even a *static* mixed-bit MoE layer breaks current fused kernels. | Answer ready: bucket experts by bit-width, one grouped GEMM per bucket — which is precisely what b12x's two-tier path already does and what MxMoE autogenerates. |

### Does EXL3 not being upstream block this?

**No — if the design is format-agnostic. Yes — if you bundle EXL3.** Upstream has declined EXL3 four times (#19896 closed as not planned; #11416 QTIP closed; #3203, #296, #2645 closed with no implementation), and #39583 / #30136 show active de-scoping of exotic formats to out-of-tree plugins.

Define the interface over a **capability**, not a format. A quant method declares:

```
supports_variable_bitwidth -> bool
available_bitwidths(tensor_name) -> tuple[int, ...]
materialize(tensor_name, bits) -> (opaque_bytes, length)
```

vLLM core owns statistics, policy, budget, scheduling, and the swap transaction. EXL3 ships out-of-tree via `register_quantization_config`; FP8, INT4, MXFP4 and LUT-B can all implement the same interface. This is also the correct engineering boundary regardless of upstreaming.

**The strongest hook into upstream's current interests is PR #50168** — mgoin (core maintainer) prototyping NVIDIA Rubin **LUT-B** at 3.125–3.522 bpw (512 3-bit indices + 8-entry E4M3 codebook per 8×64 tile), with a a **52.08% (baseline variant) → 75.51% (best variant) GSM8K** spread. That gap *is* the "which tensors deserve the bits" problem, in the same bpw band, from a maintainer. He is the natural reviewer, and #50168 is the right venue to make the allocation argument in language upstream already cares about.

### Sequencing

1. Comment on **#49702** now (deadline 2026-08-18).
2. Land **Phase 2** (quant disk cache) as a standalone PR. No RFC. Establishes credibility.
3. Land **Phase 3** (EP-independent expert-load recording + Prometheus metrics) as a second standalone PR. Coordinate with #32028.
4. **Only then** file the RFC, with Phase 0's measurements as its evidence base.

### AGENTS.md compliance (non-negotiable)

- **Pure code-agent PRs are not allowed.** Michel must read every changed line and be able to defend it end-to-end. State this explicitly in the PR body.
- **Disclose AI assistance** in the PR description, and state why the change is not a duplicate (cite #50281, #38256, #48920, #49702 by number).
- **State test commands and results.** Minimum for these phases:
  ```bash
  pytest tests/model_executor/model_loader/test_reload.py -v -s
  pytest tests/v1/cudagraph/ -v -s
  pytest tests/quantization/ -v -s
  pre-commit run --all-files
  pre-commit run mypy-3.10 --all-files --hook-stage manual
  ```
- **No low-value busywork PRs.** Bundle any mechanical cleanup with substantive work.
- Add commit trailers: `Co-authored-by: Claude`, `Signed-off-by: Michel Belleau <michel.belleau@malaiwah.com>`.

---

## 7. RISKS AND KILL CRITERIA

Ordered by when you learn them. Each has a concrete measurement and a threshold.

| # | Risk | Measurement | Kill threshold | Cost | Kills |
|---|---|---|---|---|---|
| **K1** | **Specialization gain is within noise.** VBQ, AWQ and TASA all predict this. | Phase **0d**: two quants at identical bpw, generic vs blended-workload calibration. | End-task delta < 1σ of eval noise, **or** Kendall τ between the two allocations > 0.9. | 1 day | **The whole project.** |
| **K2** | **Sensitivity is homogeneous** — nothing to allocate. | Phase **0c**: variance of dKL across experts within a layer. | Top-N and bottom-N differ by less than measurement error. | Hours | The whole project. |
| **K3** | **Allocation does not drift.** | Phase **0a**: Kendall τ of the top-N set across days/weeks. | τ > 0.9 across weeks. | Hours | Phases 3–5 only. Phase 1 (startup specialization) captures everything. **This is the most likely outcome.** |
| **K4** | **Layer 78 does not generalize** — it is the MTP *draft* head, 1 of ~78 layers. | Phase **0b**: cross-layer rank correlation. | τ ≈ 0 against early/middle layers. | 1 day | Not the project, but forces a full multi-layer collection campaign first, changing every cost estimate. |
| **K5** | **Router instability** — the feedback term dominates the signal. | Phase **0e**: top-8 Jaccard pre/post a K3↔K4 change vs BF16 ground truth. | Overlap < 0.95. | Hours | The closed loop (Phases 3–5), not startup specialization. |
| **K6** | **b12x exposes no per-expert row write.** | Phase **0f(i)**: read the source. | Only whole-tier `prepare_weights`. | Minutes | Not the design — degrades to the ~1 GiB/rank per-layer slab rebuild, one layer per step. Reserved memory rises from 9 MiB to ~2 GiB/rank. |
| **K7** | **Two-tier / K≥3 ceiling is structural.** `len(tiers) != 2` raises (`exl3.py:1591`); metadata restricts `k_values` to 3..6. | Already known. | b12x cannot exceed two tiers. | — | Halves the value: downgrade below K3 is unavailable, so the budget-freeing side of the proposal does not exist. |
| **K8** | **Sleep mode broken on SM120** (#21336, open since 2025-07-21; #48680). | Run with `--enable-sleep-mode` on one card. | Reproduces. | Minutes | The VMM route entirely, and the Phase-4 sleep/wake bootstrap. Fall back to `pause_scheduler(mode="keep")`. |
| **K9** | **Online re-encode is impractical.** Requires a per-linear Hessian (~2.35 GiB/layer, ~210 GiB model-wide) that conversion discards; unmeasured on SM120. | Phase **0f(ii)**: benchmark `quantize_exl3()` at K3/K4 on one tensor. | Any result. | Hours | Already killed by design — the store-both-variants path costs ~607 GB of NVMe and eliminates the entire problem. Measure only to document the tradeoff. |
| **K10** | **`bits_per_expert` is not regenerable.** | Phase **0f**: `git log`/provenance on the existing map. | Hand-tuned outside any toolchain. | Minutes | Phase 1 has no starting point and no allocator to drive. *(Partially retired: the loader schema is fully specified and machine-readable — `exl3.py:507-556` — so the format is regenerable even if this instance was hand-authored.)* |
| **K11** | **Upstream rejects the premise.** | Maintainer comments on #48920 hardening its base-checkpoint-only scope, or on #49702 declining size-aware transfer. | Explicit rejection. | Weeks | The RFC. The work remains valuable as a fork/plugin. |

### What the research could not establish — stated plainly, not guessed

- **b12x / SparkInfer internals.** Not in the authorized repos. I resolved the *decisive* question from the vLLM-side caller — cardinality is compiled, membership is a runtime device tensor (`exl3.py:1885-1901` vs `:1955`) — but whether `prepare_weights` aliases or repacks the stacked slab, whether a per-expert row write is exposed, and whether >2 tiers are possible remain unread.
- **EXL3 encode cost on SM120.** The ~5M weights/s figure derives from issue #112 conversion logs at 5.76 bpw on *unstated* hardware. The K-independence of Viterbi compute is proven from `quantize_tiles_kernel.cuh` (2^(16−K)·2^K = 65536 branch metrics per step, constant), but real K3-vs-K4 timings could differ by tens of percent due to the +17% K≥4 Blackwell occupancy bump and small-tensor underfill.
- **Stale-Hessian quality impact.** Whether re-encoding one expert of 256 against a Hessian captured under a different upstream configuration degrades output is unmeasured and would need an experiment. Moot under the store-both-variants design.
- **GLM-5.2-specific routing behaviour.** Every published expert-activation measurement is on Mixtral, Qwen-MoE, DeepSeek-MoE, OLMoE, or Qwen1.5-MoE. Nothing measures GLM-5.2, its 256-expert top-8 routing, MLA/DSA interaction, or MTP coupling. **The user's own trace is better evidence than any published paper**, which is exactly why Phase 0 must run before anything else.
- **The right objective for an MTP draft head** — acceptance length vs KL vs PPL — is genuinely ahead of published literature. No external validation is available.
- **Whether the current rank-pre-sharding is load-bearing** or merely a load-time convenience. The trellis format itself slices cleanly at 16-column granularity, so this looks like a storage choice, but that was not confirmed.
- **DynaExq has no code release found.** Its dual-pool allocator and stable-handle design would have to be reconstructed from prose, and its EMA α and update period T are undisclosed.
- **Maintainer sentiment is unestablished.** Comment threads were largely inaccessible during the issue sweep. Nobody knows whether this class of idea has already been argued against in a comment. Five minutes of `gh issue view --comments` closes this.

### The bottom line

Two independent published results (VBQ, AWQ) and one measurement (MoE-Infinity's "reuse counts even out over time" across 1000+ sequences) all predict that **K3 fires before K1** — that the allocation converges and stops moving, in which case a one-time warmup specialization captures the value and the continuous rebalancing machinery is wasted. Phase 1 is designed to be exactly that product, and Phase 0a will tell you within a day of pandas work.

Build Phase 0. Then decide.
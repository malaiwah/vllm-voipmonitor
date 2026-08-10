# Top 8 Unresolved Questions (prioritized by go/no-go leverage)

---

### 1. Does workload-specialized bit allocation actually beat generic allocation at equal bpw — and is per-expert sensitivity heterogeneous enough to be worth allocating over?

**Why it matters.** This is the go/no-go and nothing in either prior-art sweep answers it. The counter-evidence is strong and was never weighed: VBQ found learned allocation "stabilizes early enough to freeze into a fixed recipe"; AWQ claims activation salience generalizes across domains *without* calibration overfitting; TASA shows pure in-domain calibration actively degrades quality via Hessian eigenspectrum degeneracy and prescribes 50–75% *general* data. If the workload-specialized allocation is within noise of the generic one, every downstream engineering problem (CUDA graphs, arenas, VMM, tier slabs) is unjustified work. Separately, if `d(KL)/d(bit)` has low variance across the 256 experts, there is nothing to allocate — mixed precision degenerates to uniform.

**Cheapest test.** Entirely offline, no vLLM involved. exllamav3 already ships the tooling: `measure_model.py -l3` produces per-tensor KL-delta-vs-bitrate curves given a BF16 reference dir plus two pre-quantized dirs, and `optimize_model.py` solves the knapsack. Run it twice at identical target bpw — once with the stock bundled calibration mix, once with a calibration set drawn from the user's own live traffic — and compare (a) the *variance* of per-expert dKL across experts within a layer, and (b) the rank correlation between the two allocations, and (c) end-task quality of both compiled quants. Community-reported cost is 2–5 h for a 2-quant comparison. If the Kendall τ between the two allocations is high, or the dKL variance across experts is small, stop here.

---

### 2. Is the hot-expert set stable over hours/days for this deployment — and does layer-78 routing generalize to the other 76 MoE layers?

**Why it matters.** Both sweeps independently name this the single largest evidence gap in the literature: *no paper measures allocation stability over long horizons.* DynaExq's own paper — whose entire premise is long-horizon hotness — publishes no rank-correlation-across-windows, no hotness variance, and does not disclose its EMA α or period T. And MoE-Infinity's measurement cuts against the idea directly: skew is severe *within* a request (<5% of experts repeatedly activated) but "reuse counts even out over time" across 1000+ sequences, i.e. the model-level histogram flattens — which is exactly the statistic an EPLB-style rolling window collects.

**The hole nobody flagged:** the 7.3M-token collector trace is from **layer 78, the MTP draft layer**. It is one layer out of ~78, it is the *draft* head, and the entire per-layer allocation argument (GEMQ: global beats per-layer; arXiv 2511.05814: middle layers are most skewed; OLMoE: specialization strongest in later layers) requires per-layer data the user does not have.

**Cheapest test.** Two parts. (a) On the existing trace: bucket into ~100 windows, compute per-expert frequency and gate-mass per window, then report Kendall τ of the top-N set across windows, across hours, and across days. This is a pandas script, hours of work, zero GPU. (b) Extend the collector to 3–4 additional layers (early / middle / late) for a much shorter run and check whether layer-78 hotness rank-correlates with theirs. If (a) shows τ collapsing toward 0 across days, the continuous-rebalancing premise is dead and a one-time warmup specialization captures the value.

---

### 3. The "allocate at K_max and under-fill" workaround — recommended independently by three feasibility lenses — is arithmetically unaffordable on this hardware. Has the fixed-cardinality alternative been costed?

**Why it matters.** This is a direct internal contradiction in the feasibility work that nobody caught. The weight-lifecycle, cudagraph, and runtime-mutation lenses all converge on the same escape hatch: allocate every mutable expert at K_max, keep byte size constant, and the `copy_`/CUDA-graph/fragmentation problems all vanish (this is precisely the LoRA `max_lora_rank` pattern). But the memory lens's own arithmetic says all-K4 routed experts alone is **86.62 GiB/rank against a ~86.0 GiB budget at util=0.90** — before dense weights, MTP, the ~1 GiB Trellis arena, activations, NCCL, graphs, or any KV cache. The recommended workaround does not fit.

The unexplored alternative dissolves the blocker without paying K_max: **fix the tier cardinalities at startup** (|K4| = N, |K3| = E−N, per layer) and make rebalancing a *membership swap* — one expert out of the K4 slab, one in. Within a tier, every row is the same shape, so the swap becomes exactly the fixed-shape row `copy_` that EPLB already does safely. Memory is constant by construction, `tier_signature` never changes (so no b12x recompile, no new ~1 GiB arena, no leak in the never-evicting `_MIXED_TRELLIS_RUNTIMES` dict), and CUDA-graph pointers stay valid.

**Cheapest test.** Paper first, then code. Compute the fixed-cardinality budget at the current 3.0–3.42 bpw operating point and check whether N is large enough that membership churn buys anything. Then read `exl3.py:1616-1711` and confirm that a same-shape row write into a tier slab plus an in-place mutation of `global_to_combined` / `descriptor_map` is sufficient — which is question 4.

---

### 4. In b12x `mixed_trellis`: does `prepare_weights` alias or copy the slab, is tier membership read from device tensors at launch or baked into host-planned launch geometry, and can it exceed two tiers?

**Why it matters.** This single fact swings implementation cost by an order of magnitude and every feasibility lens flagged it as unreachable (the b12x source is not in the authorized repos). If the kernel reads tier membership from `global_to_combined` / `descriptor_map` on-device, an in-place mutation of those two tensors re-tiers experts with **no re-capture, no reallocation, no recompile** — and question 3's design is nearly free. If tier membership is baked into host-side grid dims in `state["launch"]`, every rebalance is a `compile_mixed_trellis` JIT plus a fresh arena allocation, which the code itself refuses to do under CUDA-graph capture (`exl3.py:1857-1863`). Additionally, `len(tiers) != 2` raises outright (`exl3.py:1583-1595`), so a K2/K3/K4 three-tier policy — which the proposal explicitly describes as upgrade *and* downgrade — is unsupported today.

**Cheapest test.** Read the b12x source. Failing that, empirical: after `process_weights_after_loading`, mutate one entry of `descriptor_map` in place and check whether the layer's output changes for a token routed to that expert. Also check whether `prepared` shares storage with the stacked slab via `data_ptr()` comparison before `param.exl3_tensors.clear()` runs.

---

### 5. Re-encode vs. store-both-variants: what does an EXL3 encode actually cost on SM120, and is the dual-artifact path simply cheaper than the entire online-quantization half of the proposal?

**Why it matters.** **No published system re-encodes online** — DynaExq, HOBBIT/MoE-APEX, MorphServe, DyMoE, FlexQuant, CXL-MoE all store pre-quantized variants and page them. The proposal's step 3 is unprecedented, and the EXL3 lens establishes it needs a per-linear Hessian that conversion computes sequentially and then *throws away* (~2.35 GiB per MoE layer, ~210 GiB model-wide to cache). Meanwhile the store-both path costs roughly K3+K4 ≈ 650 GB of NVMe for the routed experts — trivial — and eliminates Viterbi, Hessian persistence, stale-Hessian quality risk, and the whole "quantize online at startup" subsystem in one move. Nobody costed the two paths against each other, which is the tradeoff the entire design turns on.

Two secondary unknowns folded in here: the multi-GPU encoder would steal SMs from the four *serving* GPUs (nobody costed the interference), and — load-bearing — **it is unknown whether the GG fork's `bits_per_expert` JSON is regenerable by any tool the user controls.** If it was hand-tuned outside the upstream toolchain, there is no allocator to drive.

**Cheapest test.** Benchmark `quantize_exl3()` on one 5120×1536 tensor at K3 and K4 on one idle Blackwell; the derived estimate is ~1.5 s/tensor (~4.7 s/expert) at the ~5M weights/s seen in issue #112, but that number comes from unstated hardware at 5.76 bpw. Separately, `git log`/`grep` the provenance of the existing `bits_per_expert` map.

---

### 6. Which CUDA-graph escape hatch is actually available on *this* hardware — and does the VMM route even work at this granularity?

**Why it matters.** Three candidate escapes were identified and none was validated. (a) **CuMemAllocator stable-VA remap** is the most elegant and vLLM already ships it — but issue #21336 ("vLLM crashes when using --enable-sleep-mode with Blackwell PRO 6000 GPUs") has been **open since 2025-07-21**, plus #48680 reports sleep-mode OOM on NVFP4/SM120. Worse, nobody connected two numbers that are already in the research: the CUDA minimum allocation granularity is a 2 MiB class, and the per-expert-per-layer K3→K4 delta is **1.125 MiB/rank** — *below one granule*, so per-expert VMM remapping reclaims nothing. (b) **Full re-capture** costs 5–20 s plus `torch.compiler.reset()` (elastic-EP path), destroys the prefix cache, and is incompatible with "background, transparent to in-flight requests". (c) **Excluding MoE from graphs** was never investigated at all: `splitting_ops` is a user-settable `CompilationConfig` field currently initialized to `_attention_ops`; adding `vllm::moe_forward` would put the MoE outside the captured region. Nobody checked whether that is legal or what it costs.

**Cheapest test.** Three cheap probes: run with `--enable-sleep-mode` on one SM120 card and see if #21336 reproduces; set `compilation_config.splitting_ops` to include `vllm::moe_forward` and measure decode tok/s delta vs. baseline and vs. `enforce_eager`; and compute the granule-aligned expert bundle size needed for VMM to reclaim anything.

---

### 7. Does changing an expert's bit-width shift the router enough to destabilize the statistics that drove the decision?

**Why it matters.** Four independent groups (EAQuant, MoEQuant, ExpertQuant/Rank-Aware PTQ, EAC-MoE) establish that quantization shifts routing, and GEMQ explicitly names "router shifts induced by quantization" as a first-class failure mode requiring router fine-tuning. ExpertQuant finds "most errors arise as near-neighbor rank flips around the top-k". Every runtime system (DynaExq, HOBBIT) changes precision at runtime. **No paper connects the two** — there is zero published analysis of oscillation, required hysteresis, or convergence of an online precision loop. With top-8 of 256 fine-grained experts, the 8th/9th margin is thin. A loop that promotes expert A, thereby changing which tokens reach A, thereby changing A's measured hotness, can oscillate or lock in a bad allocation. EvoPress independently warns that per-layer error is non-monotone and non-additive, which invalidates the greedy/ILP allocators everyone else uses.

**Cheapest test.** The user already has the exact instrumentation. Take a held-out sample, run the model with one layer's experts at K3 and again at K4, and measure **top-8 Jaccard overlap** and the distribution of rank flips against the BF16 ground-truth IDs the collector captures. If overlap is >0.98, the loop is stable and hysteresis is cosmetic; if it is 0.85, the feedback term dominates and the controller needs a rollback guard.

---

### 8. Are the two "easy"-rated statistics hooks actually functional, and is the persistence artifact topology-neutral?

**Why it matters.** Several low-risk ratings do not survive scrutiny, and they sit under the whole stats layer:

- **`BaseRouter.set_capture_fn` was rated "easy"** while the same lens separately established that MoE ops are *not* CUDA-graph splitting ops, so the router executes **inside** captured graphs. A Python callback invoked from inside a captured region runs once at capture and never on replay unless its body is exclusively pure tensor ops on persistent buffers (which is precisely why EPLB's recorder is a single `scatter_add_`). Any `.item()`, host branching, list append, or per-step Python constant silently freezes. This is the difference between "the stats hook works" and "the stats hook reports capture-time values forever."
- **`enable_return_routed_experts` is hard-asserted off under DCP>1** (`scheduler.py:249-253`), which their DCP4 config trips — the in-tree capturer is unavailable, though the router hook itself is not gated.
- **The persistence precedent is topology-baked.** `startup_plan.py`'s fingerprint includes `rank` and `world_size`; `VllmConfig.compute_hash()` folds in `parallel_config`; the GG checkpoint carries `rank{r}` in tensor names and hard-fails if `checkpoint_tp != runtime tp`. The stated "topology neutral" requirement is currently violated by the storage format itself, and nobody asked *why* it is pre-sharded (load-time performance, or something structural) — the underlying EXL3 trellis slices cleanly at 16-column granularity, so this looks like a storage choice, not a format constraint.

**Cheapest test.** Bind a `capture_fn` that does one `scatter_add_` into a persistent buffer, run 100 decode steps with cudagraphs enabled, and check the counter scales linearly with steps rather than freezing at the capture-time value. Separately, diff `startup_plan.compute_plan_fingerprint` against what a topology-neutral policy key would need (checkpoint identity + base quant config + schema version, explicitly excluding rank/world_size).

---

## Suspicious absences (searched thinly or not at all)

- **`vllm-project/llm-compressor` and `vllm-project/compressed-tensors` were never searched** — explicitly admitted. This is where upstream bit-allocation logic and any on-disk representation of per-expert mixed bit-widths would live. If compressed-tensors already has a per-expert bitrate schema, both the novelty claim and the format design change.
- **GitHub Discussions were never searched** (no MCP endpoint). vLLM does design work there. AGENTS.md's duplicate-work check is not satisfied without this pass.
- **The exllamav3 issue tracker was never searched** for hot-swap, multi-bitrate residency, or runtime bit changes — the one community most likely to have already tried this.
- **No code release found for DynaExq**, the closest prior art. Its dual-pool allocator and stable-handle design would have to be reconstructed from prose.
- **Comment threads were largely inaccessible** across the vLLM issue sweep — maintainer sentiment, including any prior rejection of this class of idea, is unestablished. `gh issue view --comments` on #50281, #38256, #49198, #48920 is a five-minute fix.
- **Nothing measures GLM-5.2 specifically**, nor the interaction with MLA/DSA sparse attention, nor bit allocation for an MTP draft head where the right objective is plausibly *acceptance length* rather than KL or PPL. The user's collector work is ahead of the literature here, which means no external validation is available.
- **Time-sensitive:** RFC #49702 (EPLB platform backend — the only open proposal abstracting "allocation and movement operations for weight tensors" behind a replaceable interface, and which already contemplates per-expert tensor sequences) has a feedback window closing **2026-08-18**. Commenting there is the highest-leverage, lowest-cost action currently available and is independent of everything above.
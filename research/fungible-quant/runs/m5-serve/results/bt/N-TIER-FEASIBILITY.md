# N-tier mixed-trellis feasibility

**Date:** 2026-08-11 · **Scope:** read-only investigation, no source modified.
**Sources:** `/home/mbelleau/src/gg-vllm` (branch `fq/m1-stats-collector`, HEAD `505ffaa7b`),
b12x 1.1.0 as installed in the r33 rootfs
(`/home/mbelleau/rootfs/gg-v20-r33/opt/venv/lib/python3.12/site-packages/b12x/`),
b12x source repo `/home/mbelleau/src/b12x` (1.2.1).
**Empirical probe:** `scratchpad/probe_tiers.py`, RTX PRO 6000 Blackwell (188 SM,
101376 B opt-in shared), fruit geometry H=1024 I=512 topk=8.

---

## Answer first

1. **The `2` is not load-bearing in hardware. It is a code/ABI constant.** A 3-tier
   (K2/K3/K4) megakernel **compiles today with zero spill**: 151 registers/thread vs
   143 for the shipped K3/K4 pair, identical 51328 B shared memory, identical
   `blocks_per_sm=1`, and **identical kernel-launch count** (the mixed path is one
   cooperative grid, not one launch per tier). Measured, not inferred — see §1.4.
2. **Why 2: it is what the shipped product needed, nothing more.** The commit is
   literally titled *"add one-grid mixed K3/K4 Trellis path"*, the class docstring says
   *"One cooperative grid over two native Trellis bitrates"*, and the underlying
   `_moe_body` is already documented as *"the hybrid **multi-tier** entry"* with
   generic emit hooks. Hypotheses (a) launch scaling and (b) descriptor bit budget are
   **disproven** by evidence. (c) allocator simplicity is a contributing but weak
   factor. The honest verdict is closest to **(d) arbitrary — a 2-way `if/elif` and a
   fixed positional parameter list, shipped for one use case.**
3. **Lifting to N is ~150 lines of straightforward b12x change plus an ABI bump — but
   it is in b12x, which we do not own,** and it requires a CuTe recompile.
4. **It would not give us the K2→K3→K4 live-upgrade ladder anyway.** A mixed layer's
   *total bit budget is conserved at prepare time* regardless of tier count, because
   the combined capacity is pinned to the global expert count and each tier's slab has
   exactly its population's rows. With N tiers you get a richer *permutation* space at
   constant memory — not the ability to grow. **Recommendation: do not ask for N tiers.
   Use per-layer tier PAIRS, which are already supported end-to-end.** See §4.

---

## 1. Where the 2 is actually load-bearing

### 1.1 Our call site (trivially generalizable)

`/home/mbelleau/src/gg-vllm/vllm/model_executor/layers/quantization/exl3.py`

| line | content | generality |
|---|---|---|
| 1583-1590 | `tiers = {bits: (expert_ids…) for bits in sorted(set(bitrates))}` | **already N-tier** |
| **1591** | `if len(tiers) != 2: raise` | **the guard** |
| 1613-1687 | `for bits, expert_ids in tiers.items(): prepared_tiers.append(api.prepare_weights(...))` | **already N-tier** — the prepare loop needs no change |
| 1689-1691 | `api.build_tiered_maps(tier_ids[0], tier_ids[1], device=device)` | 2-indexed |
| 1698 | `api.combine_trellis_rotations(*prepared_tiers)` | **already splatted** — only the callee is 2-ary |
| 1886-1889 | `tier0_num_experts=…[0][1], tier1_num_experts=…[1][1], tier0_bits=…, tier1_bits=…` | 2-indexed |
| 1955-1966 | `run_mixed_trellis(x, mixed["tiers"][0], mixed["tiers"][1], …)` | 2-indexed |

The GG-side prepare path is **90% already N-tier**. Only three sites index `[0]/[1]`.

### 1.2 Descriptor / `global_to_combined` encoding — **not the constraint**

Two distinct maps, both `int32`, both length `total_experts`:

- **`global_to_combined`** — indexed by *global* expert id → combined slot, `-1` unmapped
  (built at `mixed_trellis.py:1184-1191`). Passed to route packing as `expert_map`
  (`mixed_trellis.py:1410`). **Tier-agnostic.**
- **`descriptor_map`** — indexed by *combined* slot → `(tier << 8) | local`
  (encode: `mixed_trellis.py:1192-1196`, `[*range(len(tier0_ids)), *((1 << 8) | i …)]`).

**Single decode site**, in the CuTe kernel:

```
mixed_trellis.py:353   tier = descriptor >> Int32(8)
mixed_trellis.py:354   local_expert = descriptor & Int32(0xFF)
```

Grep across all of b12x confirms exactly one decode site and one encode site
(`grep -rn "descriptor >> \|>> Int32(8)\|& Int32(0xFF)\|(1 << 8)" b12x/` returns only
`mixed_trellis.py:353,354,1193` plus an unrelated `nsa_indexer` hit).

The ABI is documented in `kernel.py:12775-12812` (`build_w4a16_tier_local_map`):
> *"Entry g = (tier << 8) | local_expert_id for a mapped global expert id g, -1 for
> unmapped."* … *"the descriptor local-id field is 8 bits"*

**Bit budget:** local = bits 0-7 (max 256 experts/tier, enforced at
`mixed_trellis.py:1177-1178` and `mixed_trellis.py:189-190`). The tier field occupies
bits 8+ of a *signed int32* whose only reserved value is negative-for-unmapped. So the
tier field has **23 usable bits ≈ 8.4 M tiers**. **Hypothesis (b) is disproven** — the
descriptor bit budget is not remotely the constraint. `swap.py:78`'s
`DESCRIPTOR_TIER_SHIFT = 8` matches the b12x ABI exactly and needs no change for N.

Note the kernel *already* tolerates holes: `mixed_trellis.py:350` checks
`combined_expert >= 0 && < total_experts` and `:352` checks `descriptor >= 0`.
`build_tiered_maps` is what forbids them (`:1180-1183`, must be a disjoint partition of
`[0, total)`); `swap.py:786-793` mirrors that.

### 1.3 The b12x kernel — where 2 *is* hardcoded

`b12x/moe/_shared/kernels/w4a16/mixed_trellis.py`:

| line | what | nature |
|---|---|---|
| 119-122 | `class W4A16MixedTrellisKernel: """One cooperative grid over two native Trellis bitrates."""` `ABI_VERSION = 6` | docstring + ABI stamp |
| 124-130 | `__init__(*, driver, tier0, tier1)` | fixed arity |
| 131-188 | geometry-agreement checks looped over `(driver, tier0, tier1)` | generic loop, 3-tuple literal |
| 189-192 | per-tier ≤256 experts; `driver.num_experts == tier0 + tier1` | 2-way sum |
| 202-207 | `blocks_per_sm = min(driver, t0, t1)`, `shared_words = max(driver, t0, t1)` | already reductions |
| 210-221 | `__cache_key__` lists `tier0`/`tier1` subkeys | 2 entries |
| **307-412** | **`_emit_tier_tile`: `if tier == Int32(0) … elif tier == Int32(1) …`** | **the CuTe dispatch — the real 2** |
| 414-451 | `__call__` ABI: 12 `t0_*`/`t1_*` `cute.Pointer` params + `tier0_num_experts`, `tier1_num_experts` scalars | **fixed positional ABI** |
| 456-563 | per-tier `cute.make_tensor` layouts using `self.tierN.trellis_bits` | 4 blocks, 2 tiers |
| 645-786 | `kernel()`; `fc1_emit`/`fc2_emit` built with `partial(self._emit_tier_tile, …)` at 698-745 | 2-tier arg packs |
| 853-872 | `compile_mixed_trellis(… tier0_num_experts, tier1_num_experts, tier0_bits=3, tier1_bits=4 …)` | fixed signature, **K3/K4 defaults** |
| 919-923 | `W4A16MixedTrellisKernel(driver=make_kernel(total, tier0_bits), tier0=…, tier1=…)` | 2 |
| 1167-1197 | `build_tiered_maps(tier0_global_ids, tier1_global_ids, …)` | 2-ary |
| 1200-1211 | `combine_trellis_rotations(tier0, tier1)` → 4× `torch.cat((t0.x, t1.x), dim=0)` | 2-ary |
| 1277-1288 | `run_mixed_trellis(x, tier0, tier1, …)` | 2-ary |
| 1310-1337 | two hand-written `_validate_mixed_trellis_tier_storage` calls | 2 |
| 1418-1543 | one `launch.compiled(...)` with 12 tier pointers spelled out | fixed ABI |

**Is `trellis_bits` a compile-time `const_expr`?** Yes, effectively. Each tier is a
separate `W4A16FusedMoeKernel` whose `trellis_bits` is a Python attribute read inside
`@cute.jit` code — `kernel.py:4270 tile_u32 = 8 * self.trellis_bits`,
`kernel.py:4276 if cutlass.const_expr(self.trellis_bits <= 4)`,
`kernel.py:4286 elif cutlass.const_expr(self.trellis_bits == 5)`,
`kernel.py:1088-1092 b_unit_bytes = 4 * self.trellis_bits`. So **each tier's decoder is
a separately specialized, fully inlined body** in the one megakernel. Adding a tier adds
two more inlined GEMM bodies (its `fc1` and `fc2`), which is exactly the cost §1.4
measures. Supported bitrates: `kernel.py:166 _TRELLIS256_BITS = (2, 3, 4, 5, 6)` —
**K2 is a first-class kernel bitrate.** (GG's *uniform* path separately restricts to
3/4/5/6 at `exl3.py:1732`; the *mixed* path imposes no such restriction.)

**Launches per layer per token — one grid, not one per tier.** `run_mixed_trellis`
issues exactly three device calls, none of which scale with tier count:
`pack_topk_routes_by_expert` (`mixed_trellis.py:1406`, keyed on `total_experts`, tier-blind),
`launch.compiled(...)` (`:1418`, a single `cooperative=True` grid — see the `.launch()`
at `:637-643`), and `launch.topk_sum.compiled(...)` (`:1544`).
**Hypothesis (a) — "kernel launches scale linearly with tiers" — is disproven.**

**Rotation combination is trivially generalizable.** `combine_trellis_rotations`
(`:1200-1211`) is four `torch.cat` calls over `dim=0`; the combined-slot layout
(tier0 `[0,t0)`, tier1 `[t0,t0+t1)`) is just concatenation order and extends to N by
construction. Our `swap.py` depends on that layout (module docstring consequence 2,
lines 16-19) but only via "slot = offset of my tier + local", which is N-safe.

### 1.4 Empirical: what a third tier actually costs

Probe method: subclass `W4A16MixedTrellisKernel` in scratchpad (no source touched), add
a third `elif tier == Int32(2)` arm dispatching to a K2-specialized `W4A16FusedMoeKernel`,
compile through `b12x_compile`, read `_query_w4a16_kernel_resources`. The probe reuses
tier1's *pointers* for the third arm — it measures the codegen cost of a third inlined
decoder, which is the thing in question, not a correct 3-tier ABI.

Production geometry (`moe_block_size=8`, `exl3.py:76 _MIXED_TRELLIS_ROUTE_BLOCK_SIZE = 8`;
tile `(128,128,32,512)` from `exl3.py:1566-1567`, which is what any `hidden % 512 == 0`
model gets — fruit H=1024 and GLM-5.2 H=6144 both land here):

| tiers | shared B | regs/thread | local (spill) B | blocks/SM | compiles |
|---|---|---|---|---|---|
| K3/K4 (**shipped**) | 51328 | **143** | 0 | 1 | yes |
| K2/K4 | 51328 | 139 | 0 | 1 | yes |
| K2/K3 | 43136 | 145 | 0 | 1 | yes |
| **K2/K3/K4** | **51328** | **151** | **0** | **1** | **yes** |

Ceilings: 255 regs/thread (architectural), 101376 B opt-in shared (device).
**A third tier costs +8 registers/thread and nothing else.** Shared memory is
`max` over tiers (`mixed_trellis.py:205-207`), so adding a *lower* bitrate is free;
per-bitrate single-tier footprint at this geometry is K2 34944 / K3 43136 / K4 51328 /
K5 59520 / K6 67712 B — all comfortably under 101376. Occupancy cannot degrade: this is
a persistent cooperative grid at `blocks_per_sm = 1` in every configuration.

Register pressure is **not monotone in tier count** (K2/K3 = 145 > K3/K4 = 143; the
3-tier 151 < a 255 ceiling with 104 to spare), so "registers forced the choice" does not
survive contact with the data.

Caveat: a real 3-tier ABI adds 6 more pointer parameters and 1 more `Int32` count.
Those live in the kernel parameter space (32 KB on Blackwell), not registers; expect the
true figure slightly above 151 but nowhere near 255.

Contrast — at `moe_block_size=64` the same geometry puts K4 at *exactly* the 101376 B
ceiling and K5/K6 fail to construct at all
(`ValueError: W4A16 shared-memory footprint exceeds device opt-in limit: 109568 > 101376`).
That is a real wall, but **it is not the production configuration** and it is a
*bitrate* wall, not a *tier-count* wall.

---

## 2. Why 2 — the verdict

**Evidence gathered:**

- `gg-vllm 680fe3b0a` (Martin Vit, 2026-07-30) — **`feat(exl3): execute mixed K3/K4
  experts in one grid`**. One-line message, **no body, no rationale**.
- `b12x 5d640ee` (2026-07-30, 2 minutes earlier) — **`moe: add one-grid mixed K3/K4
  Trellis path`**. One-line message, no body.
- `git log -S "len(tiers) != 2"` returns exactly one commit: `680fe3b0a`. **The guard
  was born with the feature and has never been revisited.**
- Later touches (`b0976b7 moe: harden mixed Trellis launch contracts`,
  `2d69f95f4 exl3: harden mixed Trellis integration contracts`,
  `cc8c270 fix(trellis): make mixed expert counts runtime-dynamic (#117)`) hardened
  *around* two tiers; none discusses the count.
- The guard's own message names the cause: `exl3.py:1592-1594`
  *"the one-grid mixed Trellis path **currently** requires exactly two bitrates"*.
  "currently" is the author's own signal that this is a status, not a limit.
- The shared driver is documented as generic: `kernel.py:6878-6882` —
  *"Phase assembly shared by the single-tier fused kernel and the hybrid **multi-tier**
  entry … The emit hooks delegate per-tile expert resolution and dispatch"*, and
  `kernel.py:1599-1600` refers to *"the hybrid multi-tier route map"*. The `_moe_body`
  contract (`kernel.py:6874-6877`, `fc1_emit_tile`/`fc2_emit_tile` as
  `cutlass.Constexpr` hooks) is **already tier-count-agnostic.** The 2 lives only in
  the hook implementation and the outer ABI.
- No design doc in `b12x/docs/` (16 files) or `b12x/skills/` mentions mixed-trellis tier
  counts. `docs/moe-execution-model.md` (203 lines) has zero hits for "tier"/"mixed".

**Against each hypothesis:**

| hypothesis | verdict | evidence |
|---|---|---|
| (a) kernel launches scale linearly with tiers | **disproven** | one cooperative grid regardless of tiers (`mixed_trellis.py:1418`, `:637-643`); route-pack and topk_sum are tier-blind |
| (b) descriptor bit budget | **disproven** | 8 bits local + 23 free tier bits in int32 (`:353-354`, `kernel.py:12784-12790`) |
| (c) per-tier slabs must be separately allocated; 2 keeps the allocator simple | **contributing, weak** | each tier is an independent `prepare_weights` result with its own `w13`/`w2`/scales, exact-size-validated at `mixed_trellis.py:1228-1274`, and the ABI spells all 12 pointers out at `:1418-1498`. Real plumbing, but mechanical — no allocator *design* problem |
| (d) genuinely arbitrary | **closest to correct** | K3/K4 was the GLM-5.2 product need; the guard says "currently"; the driver is already documented as multi-tier; a 3-tier build compiles clean today |

**Stated plainly: the constraint is a 2-way `if/elif` in `_emit_tier_tile` plus a
positional CuTe ABI that spells `tier0`/`tier1` out by hand. There is no evidence anyone
evaluated N > 2 and rejected it.** This is inference from absence-of-rationale plus a
positive compile result; b12x's authors could still have an unwritten reason. **What
would settle it:** ask the b12x maintainers directly, or check for a `#117`-style issue
thread we do not have locally.

---

## 3. Cost to lift to N tiers

### Trivial (mechanical loop-over-tiers)

| file:line | change |
|---|---|
| `exl3.py:1591` | delete the guard (or relax to `< 2` / `> N_MAX`) |
| `exl3.py:1689-1691` | `build_tiered_maps(*tier_ids, device=device)` |
| `exl3.py:1886-1889` | `tier_num_experts=[…], tier_bits=[…]` |
| `exl3.py:1955-1966` | `run_mixed_trellis(x, mixed["tiers"], …)` |
| `mixed_trellis.py:131-188` | replace the `(driver, tier0, tier1)` literals with `(driver, *tiers)` — already loops |
| `mixed_trellis.py:189-192` | `sum(t.num_experts for t in tiers)` |
| `mixed_trellis.py:202-207` | already `min`/`max` reductions — take over `tiers` |
| `mixed_trellis.py:210-221` | cache key over `tuple(t.__cache_key__ for t in tiers)` |
| `mixed_trellis.py:1200-1211` | `combine_trellis_rotations(*tiers)` → `torch.cat([t.x for t in tiers])` |
| `mixed_trellis.py:1310-1337` | loop the storage validation |
| `swap.py:78` | no change — `DESCRIPTOR_TIER_SHIFT = 8` is already the right ABI |

### Real work

1. **`build_tiered_maps` (`mixed_trellis.py:1167-1197`).** Variadic `*tier_global_ids`;
   descriptor becomes `[(t << 8) | i for t, ids in enumerate(tiers) for i in range(len(ids))]`.
   ~15 lines. The partition invariant at `:1180-1183` generalizes unchanged.
2. **`compile_mixed_trellis` (`:853-1076`).** Signature from
   `tier0_num_experts/tier1_num_experts/tier0_bits/tier1_bits` to sequences; `tier_args()`
   emitted `len(tiers)` times in `compile_args` (`:1000-1001`); `MixedTrellisCompileResult`
   fields (`:53-56`) become tuples. Note `:919-920` `driver=make_kernel(total, tier0_bits)`
   — the driver's own `trellis_bits` is vestigial (its LUT slots are explicitly unused,
   `:773-776`) but it still sizes the driver's shared-memory region, so keep it at the
   **lowest** tier bitrate to avoid inflating `shared_words`.
3. **Per-tier staging in our swap engine.** `swap.py:730-742` builds `self._stages` from
   a single engine-wide `self.tier_bits` pair; `swap.py:707` defaults it to `(K3, K4)`;
   `swap.py:561-563` hard-rejects anything else
   (`if tuple(mixed["tier_bits"]) != (K3, K4): raise`); `MixedLayerState` (`:534-573`)
   has exactly `tier0`/`tier1` + `tier0_globals`/`tier1_globals`. This becomes
   `tiers: list`, `tier_globals: list[list[int]]`, and an `ExpertStage` per distinct
   bitrate. `ExpertStage` already takes `bits` as its first argument, so the staging
   primitive itself needs no change. `_validate_host_maps` (`:833-847`) generalizes to
   a per-tier offset walk.
4. **Rotation combination consumers.** Our swap engine writes combined-slot rows
   (docstring consequence 2); with N tiers the slot base becomes a prefix sum instead of
   `0 / t0`. Small.

### Hard / blocked (b12x-owned, CuTe recompile, new ABI)

1. **`_emit_tier_tile` (`mixed_trellis.py:307-412`).** The `if tier == 0 / elif tier == 1`
   chain must become an unrolled `for tier_idx in cutlass.range_constexpr(len(tiers))`.
   Each arm is a *distinct specialization* of `_run_tile`, so the megakernel grows two
   inlined GEMM bodies per tier. **Measured cost: +8 registers/thread for the third
   tier, 0 spill (§1.4).** Not blocked by hardware — blocked by ownership.
2. **The `__call__` / `kernel` positional ABI (`:414-451`, `:645-679`).** 6 pointers +
   1 count per tier, plus the per-tier `cute.make_tensor` layout blocks (`:456-563`).
   Because CuTe entries are Python-variadic-hostile, the practical fix is either
   (i) recompile per tier count with a generated signature, or (ii) switch to a single
   packed **pointer-table tensor** (`int64[num_tiers][6]`) — cleaner and permanently
   N-safe, but a strictly larger ABI change.
   **`ABI_VERSION = 6` (`:122`) must bump**, invalidating every cached compile.
3. **Disk/memory compile cache.** `KernelCompileSpec.from_key("moe.w4a16.mixed_trellis",
   ABI_VERSION, cache_key)` (`:1028-1030`) — a bump means a cold CuTe compile on first
   boot for every geometry. My probe compiles took **minutes each** cold.

### Quantified

- **Extra GEMM launches per layer per token: 0.** One cooperative grid regardless of N.
- **Extra registers: +8/thread for tier 3** (143→151, ceiling 255). Extrapolating
  ~8/tier, register pressure does not bite before ~N=14 — long past any practical need.
- **Extra shared memory: 0** when the new tier's bitrate ≤ the current max
  (`shared_words = max`, `:205-207`). Adding a *higher* bitrate costs
  ~8192 B per bit at the production tile (34944/43136/51328/59520/67712 for K2..K6).
- **Extra occupancy loss: 0.** `blocks_per_sm = 1` in all measured configurations.
- **Extra slab memory: 0.** Experts are *partitioned*; total slab bytes =
  Σ over experts of that expert's own bitrate, independent of how many tiers the
  partition has. Per-expert payload = `3·H·I·b/8` bytes
  (derivable from `swap.py:801-806`; validated against `policy.py:465-473`'s measured
  GLM-5.2 TP4 figures: K2 2,399,244 / K3 3,578,892 / K4 4,758,540 / K5 5,938,188 B per
  rank per expert, slope 1,179,648 B per K).
- **Per-tier fixed overhead: negligible.** 4-byte dummy `w13_scale`/`w2_scale`
  (`:1247-1248`) + `E` fp32 global scales × 2 ≈ under 2 KiB per layer per tier.
- **Buffers** (`make_mixed_trellis_buffers`, `:1079-1149`) are sized by
  `size_m·top_k·hidden/intermediate` and `total_experts` — **tier-count-independent.**

---

## 4. The practical question: can we run a K2→K3→K4 ladder?

### 4.0 The finding that reframes everything: bit budget is conserved

**A mixed-trellis layer cannot grow its total bit budget while serving, with any number
of tiers.** Three independent constraints pin it:

1. `run_mixed_trellis` requires **both maps to have exactly
   `tier0_num_experts + tier1_num_experts` elements** (`mixed_trellis.py:1338-1352`),
   and `global_to_combined` is simultaneously used as the router's `expert_map` indexed
   by *global* expert id (`:1410`, validated `exact_num_experts` inside
   `pack_topk_routes_by_expert`). Therefore **Σ tier capacities ≡ number of global
   experts.** There is no room to over-allocate a tier.
2. Each tier's slab has **exactly its population's rows**, exact-size-validated
   (`:1228-1274` — `int(tensor.numel()) != expected_elements` fails closed), and is
   allocated once from `num_experts=len(expert_ids)` at `exl3.py:1671`.
3. `swap.py:786-793` encodes the same: *"v1 requires occupancy == capacity"*.

So the only legal runtime operation is a **cardinality-preserving permutation** of
experts across tiers. With 2 tiers that is a K3↔K4 exchange (what the swap engine does).
With 3 tiers it is a 3-cycle. **In neither case does the layer's average bitrate move.**

**Consequence for the stated goal:** "boot at K2 for fastest time-to-first-serve, then
live-upgrade toward K3/K4" is a *memory-growth* operation. It is not a tier-count
problem. N tiers would not deliver it.

### 4.1 Is a 3-tier (2,3,4) layer feasible, and at what cost?

**Technically yes** — §1.4 proves the codegen. Cost: +8 regs/thread, 0 extra shared
memory, 0 extra launches, 0 extra slab bytes, plus a b12x ABI bump we do not own and a
cold recompile of every mixed layer.

**But its value to us is much smaller than it looks.** What it buys is a *finer rate
allocation at constant memory*: instead of ranking experts into two sensitivity buckets
per layer, three. Whether that improves KLD per byte enough to justify an upstream ABI
change is unmeasured. **Not determined** — settling it needs an offline rate-allocation
sweep on real sensitivity data (2-bucket vs 3-bucket optimal allocation at matched
bytes), which needs no kernel change at all.

### 4.2 Cheaper alternatives at 2 tiers

**(i) Different tier PAIRS on different layers — ALREADY FULLY SUPPORTED. This is the
recommendation.**

`tier_bits` is **per-layer, not global**, on every axis:

- Config: `exl3.py:559-576 rank_sliced_layer_bitrates(layer_name)` parses the layer index
  out of the name and returns `self.rank_sliced_bits_by_layer[layer_index]` — a per-layer
  bitrate tuple. Stamped per layer at `exl3.py:1357-1360`.
- Prepare: `exl3.py:1583-1590` derives `tiers` from *that layer's* bitrates. A layer whose
  bitrates are all in `{2,3}` gets a (2,3) pair; another gets (3,4).
- Runtime cache: `exl3.py:1840-1856` keys on `tier_signature` (bits **and** counts), so
  different pairs get different runtimes automatically.
- b12x compile cache: `mixed_trellis.py:935-942` keys on `kernel.__cache_key__`, which
  includes each tier's `trellis_bits`. **Different bit pairs = separate compiles;
  different expert splits at the same bit pair = a cache HIT**
  (`:955-968` + the explicit comment at `:216-218`, which is what `cc8c270 make mixed
  expert counts runtime-dynamic` bought). So D distinct pairs cost D cold CuTe compiles
  at first boot (disk-cached thereafter) and **zero** steady-state cost.
- **Already proven in production with a non-K3/K4 pair**: `policy.py:474-476` records that
  `glm52-mixed-k3k5` *"carries real K5 experts and they measure 5,938,188 B/rank exactly"*.
  A **(3,5)** layer has shipped. The pair is genuinely free choice.

So a model-wide **K2/K3/K4/K5 ladder already exists today** — it is just distributed
*across* layers rather than *within* one. Shallow layers (2,3), mid (3,4), deep (4,5),
each layer swapping inside its own pair. `policy.py:32-33 K_UNIVERSE = (K2,K3,K4,K5)`
shows our policy layer already reasons in four rungs.

**What we would have to change (small, all ours):** `swap.py:707` `tier_bits` is
*engine-wide*, `swap.py:561-563` hard-rejects any pair but `(K3,K4)`, and
`swap.py:736-742` builds one `(K3,K4)` stage pair set. These must become per-layer.
`ExpertStage` already takes `bits` as a parameter, so this is bookkeeping — make
`self._stages` a dict keyed by the layer's pair, and read the pair from
`mixed["tier_bits"]` instead of asserting it. `admin.py:1900-1902` passes
`tier_bits=(P.K3, P.K4)` and would stop passing it at all.

**(ii) A (2,4) pair skipping K3.** Works and is the *cheapest* configuration measured
(139 regs/thread, lowest of all four). But it makes every swap a 2-bit jump: the
promotion granularity doubles (`1,179,648 B/K` × 2 = 2.36 MB/rank per promotion on
GLM-5.2 TP4), which coarsens the memory controller's step size and makes the KLD/byte
frontier lumpier. Useful as a *per-layer* choice where sensitivity is bimodal; a poor
global default.

**(iii) Re-preparing one layer at a quiesce point — expensive, and it breaks CUDA
graphs.** Concretely what a re-prepare costs:

1. **Source tensors are gone.** `exl3.py:1706-1710` clears `param.exl3_tensors` and nulls
   `exl3_backing` right after prepare, deliberately ("rather than retaining both
   representations for every layer"). `swap.py` docstring line 30 says the same: *"the
   artifact pair is the only source of truth"*. So a re-prepare must **re-read every
   expert's fragments from disk/HF** for the whole layer — 256 experts × ~3.6 MB/rank at
   K3 ≈ 0.9 GB/rank of IO for one GLM-5.2 layer.
2. **Recompile, if the bit pair changed.** New `tier_signature` ⇒ miss at
   `exl3.py:1857` ⇒ `compile_mixed_trellis` for **both** the decode and prefill states
   (`exl3.py:1910-1915`) ⇒ two cold CuTe compiles (minutes) unless disk-cached. If only
   the *split* changed at the same bit pair, this is a cache hit and nearly free.
3. **Compilation is illegal mid-serve.** `exl3.py:1860-1864` raises if a stream is
   capturing, and b12x raises `raise_if_kernel_resolution_frozen` (`:1022-1024`) once the
   engine has started.
4. **CUDA graph invalidation — the actual killer.** Slab and buffer addresses are baked
   into the graph as literal pointer arguments at `mixed_trellis.py:1427-1498`, and the
   tier counts as literal `Int32` at `:1538-1539`. Reallocating slabs or changing a tier
   population **invalidates every captured graph touching that layer.** Full re-capture
   is required. This is precisely the property our swap engine was designed to avoid —
   its whole thesis (`swap.py:1-12`) is "row rewrites, no reallocation, no recompile,
   CUDA-graph-safe".
5. **Peak memory doubles for that layer** during prepare (old + new slabs live together).

Verdict on (iii): viable only as a rare, deliberate, **advertised** operation — a
maintenance-window "re-plan the model", not something the swap loop can reach for. It is
the *only* mechanism that can grow a layer's bit budget, so if growth is genuinely
required, this is the path — but it should be framed as a controlled restart of one
layer, with graph re-capture, not as a swap.

### 4.3 Recommendation

**Do not pursue N tiers. Do (i): per-layer tier pairs.** Reasoning, in order:

1. **N tiers does not solve the stated problem.** §4.0 — bit budget is conserved under
   permutation regardless of tier count. A 3-tier layer still cannot grow. Asking b12x
   for an ABI change that does not deliver the goal is the wrong ask.
2. **Per-layer pairs already give the full K2..K5 ladder**, model-wide, today, with zero
   kernel change, and one non-K3/K4 pair (3,5) has already shipped (`policy.py:474-476`).
   The cost is D cold compiles at first boot, disk-cached thereafter, and zero
   steady-state overhead — the compile cache is explicitly split-agnostic
   (`mixed_trellis.py:216-218`).
3. **The work is entirely on our side of the seam** and is small: lift `tier_bits` from
   engine-wide to per-layer in `swap.py` (:561-563, :707, :736-742) and stop hardcoding
   `(P.K3, P.K4)` at `admin.py:1902`. `ExpertStage` is already bits-parameterized.
4. **Boot-at-K2-then-grow should be reframed as a boot-time decision, not a runtime one.**
   Pick each layer's *pair* and *split* from the memory budget at load; then let the swap
   engine do what it is good at — moving the *right* experts into the high rung within
   that fixed budget as sensitivity data arrives. That is a real, measurable quality win
   at constant memory and constant time-to-first-serve.
5. **Keep (iii) in the back pocket, named honestly.** If we later need the model's total
   bit budget to grow while up, that is a per-layer re-prepare with graph re-capture. Cost
   it as such (~0.9 GB/rank IO per GLM-5.2 layer + re-capture), gate it behind a quiesce +
   explicit operator action, and never let the swap loop trigger it.
6. **If 3 tiers still looks attractive later, prove the value offline first** — a
   2-bucket vs 3-bucket rate-allocation sweep at matched bytes on real sensitivity data.
   That needs no kernel work. Only if it shows a material KLD/byte gain is the b12x ABI
   conversation worth having — and then the ask should be the packed pointer-table ABI
   (§3 hard item 2), which is N-safe once and forever, not a hardcoded three.

---

## Appendix: reproducing the measurements

```bash
cd /tmp/claude-1000/.../scratchpad
PROBE_BLOCK=8 CUDA_VISIBLE_DEVICES=1 \
  /home/mbelleau/protensors-work/vllm-voipmonitor/research/fungible-quant/runs/gg-env/gg-run.sh \
  python probe_tiers.py k234     # also: k34, k24, k23, sizing
```

`probe_tiers.py` subclasses `W4A16MixedTrellisKernel` in scratchpad and compiles through
the stock `b12x_compile`. **No file under `/home/mbelleau/src/` or in the r33 rootfs was
modified.**

### Open items / not determined

- Whether b12x's authors ever evaluated N > 2. No rationale exists in any commit message
  or doc we can read. **Settled by:** asking the b12x maintainers, or a `#117`-style
  issue thread we do not have locally.
- Whether 3-bucket rate allocation beats 2-bucket at matched bytes. **Settled by:** an
  offline sweep on real per-expert sensitivity — no kernel change needed.
- Exact register cost of a *correct* 3-tier ABI (6 extra pointer params + 1 count).
  Measured lower bound 151/255; the delta from parameter space should be small.
  **Settled by:** implementing the real ABI, which requires b12x write access.

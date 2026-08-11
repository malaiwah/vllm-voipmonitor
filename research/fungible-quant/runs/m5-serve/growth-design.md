# Implementing `growth_supported` — unpaired promotions

**Goal:** let the loop promote an expert K3→K4 *without* demoting another, when
the memory envelope has room. Today `/fq/state` reports
`growth_supported: false` and `record["promotions_applied"]` is hard-wired
`False`.

## Growth is blocked at two independent layers

**1. No promotion is ever proposed.** Every live decide line reads
`headroom=None`:

```
FQ decide rank=0 step=100 interval=1 swaps=64 promotions=0
    budget_rejections=0 used=71850230784 B headroom=None
```

`headroom=None` means `self.budget.limit_bytes is None`, so `budget_filter`
never runs and the `promotions` list is always empty. `VLLM_FQ_MEMORY_BUDGET`
is simply not set on this serve. Fixing only this yields proposals that still
cannot be applied — which is the state BT-6 already exposed once, and is worse
than nothing because the metrics would move.

**2. There is no free slot to promote into.** `exl3.py` prepares each tier at
exactly its occupancy:

```python
api.prepare_weights(..., num_experts=len(expert_ids), ...)
```

A swap works because it is a *substitution* — one expert's rows overwrite
another's inside a fixed slab. A promotion needs a slab with **one more
expert** than it has.

## Two ways to get that slot

### A. Reallocate the tier on demand

Allocate a bigger K4 slab, copy the existing experts, add the new one, rebuild
`global_to_combined` / `descriptor_map` / `rotations`, flip under quiesce, free
the old.

Measured cost on this box: the K4 slab for one layer is
`56 × 4.76 MB ≈ 266 MB/rank`, and during the copy **both** slabs are resident,
so the transient peak is ~532 MB for a single layer. Device free after KV
sizing is **3.90 GiB/rank**, of which the swap engine already reserves
**1.00 GiB** for staging — leaving 2.90 GiB. One layer at a time fits, but:

- the transient cost scales with the layer's *current* K4 count, so it grows
  as the model converges;
- every promotion pays a full slab copy, making the common case expensive;
- and the peak competes with staging, so growth and swaps contend.

### B. Pre-size the tier with a declared reserve *(recommended)*

Prepare each mixed layer with `num_experts = n_k4 + reserve`. A promotion then
writes into an already-allocated free slot: no realloc, no transient doubling,
no contention with staging. Demotion frees a slot back.

Cost, across the 48 mixed layers, per rank:

| reserve/layer | cost | promotions available |
|---|---|---|
| 8 | **1.70 GiB** | 384 |
| 16 | 3.40 GiB | 768 |
| 32 | 6.81 GiB | 1,536 |

The reserve is **permanently allocated and initially unused** — which sounds
wasteful and is exactly right. It is declared at boot, so the memory preflight
projects it like any other weight bytes, and the operator sees the true
footprint *before* the engine starts rather than discovering it when a
promotion doubles a slab mid-serve.

This is ordinary capacity planning: reserve the growth you intend to use,
account for it up front, and refuse growth beyond it.

## Recommendation

**Take B.** It converts an unbounded runtime allocation into a bounded,
declared, preflight-visible one — which is the same principle that made the
memory preflight worth building.

With the current 3.67 GiB KV and 2.90 GiB spare, a reserve of **8 slots/layer
(1.70 GiB)** fits without touching KV, and 384 promotions is far more than the
loop's 64-per-interval cap can spend in any reasonable convergence.

## Work items

1. **Reserve plumbing.** `VLLM_FQ_GROWTH_RESERVE` (default 0 — off, so nothing
   changes for existing deployments). Thread it into the tier preparation so
   the K4 tier is built at `n_k4 + reserve`, and teach `MixedLayerState` that
   occupancy < capacity is legal.
2. **Preflight.** Add `reserve × expert_bytes[K4] × mixed_layers` to the
   projection so the boot-time budget check sees it. Without this, growth
   silently eats the KV cache — the exact failure the preflight exists for.
3. **Promotion path.** `apply_fn` currently receives only a swap list. Give it
   the promotion list too, and have the engine write into a free slot and
   extend `tier1_globals`.
4. **D1 relaxation, carefully.** `store.validate_policy` enforces
   `occupancy == capacity`. Growth means occupancy may *exceed* the boot
   `n_k4_per_layer`, so the declared cardinality must be updated in the same
   committed document — `loop._doc_for` already recomputes it, which is why
   that code exists.
5. **Refusal.** Growth beyond the reserve must be refused with the reserve
   size and the remedy named, never by silently falling back to a swap.
6. **Instrument.** `fq_growth_slots_free{layer}` and a `promotions_applied`
   counter distinct from `swaps_applied` — the proposed/applied split already
   proved necessary once.

## What must not happen

Growth must never come out of the KV cache. The envelope is the envelope: if
the reserve does not fit alongside a usable KV cache, the answer is a smaller
reserve or a smaller policy, not a smaller cache. That is the same lesson as
the −3.1 GiB boot, and it is easier to relearn than it looks.

# Rebase onto fresh GG + upstream overlap analysis — 2026-08-11

Prepared ahead of the upstream PR, to satisfy the duplicate-work check that
`AGENTS.md` makes mandatory. Everything below was verified by reading and
running upstream code, not by reading PR descriptions.

## 1. Target branch — `dev/gilded-gnosis`, and `main` is a trap

| repo | branch | state |
|---|---|---|
| gg-vllm | `fq/m1-stats-collector` @ `2158a69f3` | 11 commits atop base, **0 behind** |
| base | `origin/dev/gilded-gnosis` @ `e2666d9a6` (2026-08-07) | unchanged after fetch |
| backup | `fq/m1-pre-rebase-backup` @ `2158a69f3` | created |
| remote | `work/fq/m1-stats-collector` = `2158a69f3` | already in sync |

Our branch was cut from `dev/gilded-gnosis` and that is the correct target.
GitHub reports `default_branch: main`, but:

```
git rev-list --left-right --count origin/main...origin/dev/gilded-gnosis  ->  0   416
git merge-base origin/main origin/dev/gilded-gnosis -> e12b91b03 (== origin/main)
```

`main` (`e12b91b03`, 2026-07-09) is a strict **ancestor**, 416 commits behind
— the stale vanilla-vLLM mirror. Rebasing onto it would delete the EXL3
backend, the b12x fused-MoE path, and the `MoERunner`/`BaseRouter` classes we
bind to. **Any automation that trusts `default_branch` on this repo will
destroy the branch.** 59 of those 416 commits touch `fused_moe/`,
`model_loader/` or `quantization/` — notably `08b5c1821` (EXL3 backend),
`8c4069a25` (b12x fused trellis primary path), `c7c7ea416` (Sparkinfer unified
fused MoE API), `680fe3b0a` (mixed K3/K4 one grid).

**Rebase result: no-op, 0 conflicts, no history rewritten, nothing
force-pushed.** We are already current.

## 2. Upstream overlap analysis

Four open PRs land in our area: #280 (GG EXL3 R7 native mixed K3/K4/K5
runtime), #279 (load R7 per-(expert, projection) checkpoints), #277 (load
mixed Trellis directly into tier slabs), #281 (InstantTensor borrowed-buffer).

### (a) `r7_routed_experts` vs our `hybrid_tr3_tail` — parallel, interoperable

PR280's `Exl3Config.override_quantization_method` keeps **both** paths
(`exl3.py:1055-1065`): `r7_routed_experts` with
`schema == "r7-complete-v2-checkpoint-v1"`, *and* `hybrid_tr3_tail` with
`format == _RANK_SLICED_FORMAT`, where `_RANK_SLICED_FORMAT = "exl3-trellis"`
(`exl3.py:342`) — exactly our string. Our external-bitmap contract survives:
the `bits_per_expert` `"file.json:field"` validation moves from line 498 to
1219 with identical logic and error text.

The contracts are genuinely different and a checkpoint picks one:

| | ours (`hybrid_tr3_tail`) | theirs (`r7_routed_experts`) |
|---|---|---|
| granularity | per **expert** | per **(expert, projection)** — gate/up/down independent |
| K values | K3/K4 | K3/K4/K5 |
| where bitrate lives | **external JSON bitmap**, emitted at boot from the policy | **baked into checkpoint tensors** at quantization time |
| mutable at runtime | yes — that is the point | no |
| `bits` value | `"mixed"` | `"mixed_tensor"` |

Their schema is deliberately immutable (validated, sorted, frozen to
`{3,4,5}`). Ours is a boot-time indirection precisely so the policy can
rewrite it. **No conflict; they do not supersede us.**

### (b) Tier slabs — upstream owns the representation, and we are already an adapter

`MixedLayerState.from_exl3_mixed_trellis` (`swap.py:512`) consumes GG's
existing `layer.exl3_mixed_trellis` dict. All six keys we read (`tiers`,
`tier_ids`, `tier_bits`, `global_to_combined`, `descriptor_map`, `rotations`)
are present and identically named on `dev/gilded-gnosis` (l.1692), pr277
(l.2714) **and** pr280 (l.3117). #277 changes how slabs are *populated*
(direct-to-slab load, avoiding a copy), not the runtime dict shape — so **#277
obsoletes nothing of ours and needs no adaptation.**

The real issue is pr280's **second** assembly site (`exl3.py:4039`, the R7
native path), which reuses the same key names with **different semantics**:

- maps built by `build_projection_tiered_maps(gate, up, down, tier_slots=…)`
  instead of `build_tiered_maps(tier0_ids, tier1_ids)`;
- `tier_ids` becomes **FC1 slot counts** (integers), not global expert-id lists;
- new `tier_gate_experts` / `tier_up_experts`.

Our adapter reads `tier_ids` as id lists and mutates them on swap. On an
R7-native layer it **fails loudly, not silently** (`tier_bits == (3,4,5)`
trips our `!= (K3,K4)` guard; a hypothetical 2-tier R7 layer would `TypeError`
unpacking ints) — but it does not work there. Adaptation cost is bounded:
`swap.py` is 1117 lines with 43 `tier0/tier1` references, 15 `K3/K4`
constants, 2 `build_tiered_maps` call sites; supporting projection-granular
3-tier means generalizing tier arity and the slot-identity model in the commit
protocol. Substantial refactor of `MixedLayerState` + `_stage_maps_for_layer`
+ plan/apply — not a rewrite.

**Recommendation: scope v1 to the rank-sliced expert-granular path (which
#280 preserves) and state that R7-native swap is future work.**

### (c) Duplication — the dynamic differentiator holds, with two honest caveats

Verified by reading their code, not their PR text:

- **No runtime re-tiering anywhere in #280/#277.** Every `prepare_tier` call
  (pr280 l.3076, 3109, 3888, 3981, 4027) is load-time. Grepping
  `swap|retier|hotness|expert_stats|promote|demote` in their `exl3.py` returns
  nothing. Their contribution is *static* mixed-K load + dispatch.
- Nothing in #280/#277/#279/#281 collects routing statistics or makes
  allocation decisions.

**Caveat 1 — vLLM's EPLB is the strongest duplicate-work challenge and a
reviewer will raise it.** `vllm/distributed/eplb/eplb_state.py` already
maintains `expert_load_pass` and a windowed `expert_load_window` of shape
`(window_size, num_moe_layers, num_physical_experts)` — structurally similar
to our `stats.py` ring. Defensible differences, which the PR must state up
front rather than bury:

- EPLB rebalances **placement** (replicating hot logical experts as redundant
  physical experts, rearranging across devices) and never changes precision;
  we change **bitrate in place** under a fixed memory budget.
- EPLB counts **physical** experts (post-replication, rank-sliced); the swap
  policy needs **global/logical** identity.
- We track **gate mass** (routing-weight sum), not just hit counts, and the
  policy scores on mass.
- EPLB's counters require `enable_eplb=True`, which switches on the whole
  replication/rearrangement machinery we do not want.

**Caveat 2 — partial conceptual overlap between their new
`exl3_online_cache.py` and our `lazy_encode.py`.** Theirs is a synchronous,
blocking, whole-projection encode at a single uniform bitrate
(`VLLM_EXL3_ONLINE_TRELLIS_BITS`) with a disk cache keyed by model+encoder
identity and a 600s lock. Ours is asynchronous and explicitly non-blocking: on
a resolver miss it substitutes a fallback K and enqueues `(layer, expert, k)`
to JSONL for an out-of-band worker. Different purpose, different granularity,
opposite blocking semantics — not duplicate. The honest framing is that our
drain worker *could* call their `load_or_quantize` instead of shelling out to
`fruit_encode_driver.py`; offering that consolidation strengthens the
duplicate check rather than weakening it.

**Bottom line: the dynamic differentiator is real and verified.**

## 3. Conflict forecast if #280 lands first — zero change

Tested, not estimated: a trial `git rebase --onto pr280 origin/dev/gilded-gnosis`
in a throwaway worktree replayed all 11 commits **cleanly, 0 conflicts**, both
hooks intact (`"progressive"` at lines 43/77; gpu_worker hook present), and
the committed CPU suite ran **100 passed, 1 skipped, 10 deselected**. File
sets are disjoint: #280 touches `config/quantization.py`, `envs.py`,
`exl3.py`, `exl3_online_cache.py`, `model_loader/utils.py`,
`models/deepseek_v2.py`, `warmup/kernel_warmup.py`; we touch
`model_loader/__init__.py` and `v1/worker/gpu_worker.py`. **No dependency on
#280 need be declared.**

Relevant to whether #280 can land soon: it calls
`mixed_api.build_projection_tiered_maps`, which exists **only** on b12x branch
`codex/r7-mixed-trellis-k345-v2-20260810` and is **not merged into b12x
master**. #280 is blocked on unmerged b12x work. `build_tiered_maps` (our
2-tier API) survives unchanged on that branch, so "b12x r7 + #280 both land"
still does not break `swap.py`.

## 4. Tests (verbatim)

Working tree (includes uncommitted WIP), `CUDA_VISIBLE_DEVICES=""`,
`--noconftest` (the repo conftest needs a compiled `vllm._C`):

```
1 failed, 121 passed, 1 skipped, 10 deselected in 6.19s
FAILED tests/exl3_fungible/test_stats_cpu.py::test_out_of_range_ids_are_dropped_not_scattered
```

Committed branch (replayed on #280): `100 passed, 1 skipped, 10 deselected in
5.77s`. Imports: 11/11 OK against the built runtime.

**The single failure is a genuine pre-existing bug in uncommitted WIP, not a
rebase artifact** — see §5.

## 5. Bug found: `torch.histc` folds the padding sentinel into a real expert

`torch.histc`'s last bin is **closed at `max`**, so `id == num_experts` is not
dropped — it lands in the final expert's bin:

```
torch.histc(tensor([0., 8., -1., 300.]), bins=8, min=0, max=8) -> [1,0,0,0,0,0,0,1]
```

`-1` and `300` drop correctly; `8` does not. Since `num_experts` is the usual
padding sentinel for `topk_ids`, this fires on real traffic and biases the
routing histogram toward exactly one expert — **statistical corruption feeding
the swap policy.** Memory-safe (the illegal-access class the fast path was
added to fix is genuinely fixed), but it silently poisons the decision input.

Fix: bin with `bins=num_experts+1, max=num_experts+1` and slice off the
overflow bin, or mask before binning.

## 6. Other state

- **b12x 3 behind** (`7cecbb2` → `80715be`): rewrites w4a16 kernels heavily,
  but `mixed_trellis.py` is byte-identical and `build_tiered_maps` keeps its
  signature — safe to advance.
- **exllamav3 current** (`704aefd`, 0 behind).
- **Drift risk low.** All three integration points survive even on
  `dev/b12x-moe-attn-lin` (907 ahead / 162 behind):
  `BaseRouter.{global_num_experts,capture_fn,set_capture_fn}` intact and still
  invoked in the routing path, `MoERunner.router`/`layer_id` intact,
  gpu_worker anchor block intact, `model_loader/__init__.py` only drops
  `bitsandbytes`.
- **One fragility**: `integration.py` imports `MoERunner` from
  `fused_moe.layer`, which is now only a **re-export** of
  `fused_moe.runner.moe_runner`. Import from the real module with a fallback.
- The GG rootfs r33 image is **already patched in place with our hooks** (its
  `model_loader/__init__.py`, `gpu_worker.py`, and a copy of `exl3_fungible/`
  in site-packages) — it is the M2 dryrun deploy, not pristine GG.

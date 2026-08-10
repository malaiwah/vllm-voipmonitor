# Gilded Gnosis (GG) fork — EP / EPLB / MoE / DCP investigation

**Setup verified:** `origin/main` = merge-base = `99267c23`. `gg/dev/gilded-gnosis` HEAD = `e2666d9a`. `origin/main..gg/dev/gilded-gnosis` contains 4052 commits total, of which upstream vLLM PRs (subject ending in `(#NNNNN)`) account for the bulk. Filtering those out leaves **163 GG-authored commits** (no trailing PR number), primarily by `Martin Vit <martin@voipmonitor.org>` (71 commits) plus collaborators (`Brandon Music`, `FujitsuPolycom` merge, others), spanning 2026-04-05 → 2026-07-30. That 163-commit list is what I treat as "GG-authored" below; everything else is upstream drift.

---

## 1. EPLB (`vllm/distributed/eplb/`) — GG did NOT touch it

```
git log --oneline --author='martin@voipmonitor.org' origin/main..gg/dev/gilded-gnosis -- vllm/distributed/eplb/   →  (empty)
git log --oneline origin/main..gg/dev/gilded-gnosis -- vllm/distributed/eplb/ | grep -v '(#[0-9]\+)$'            →  (empty)
```
Every one of the 22 commits touching `vllm/distributed/eplb/` (e.g. `24270941 [EPLB] Support EPLB for DeepSeek v4 Mega Moe`, `ac614587 [EPLB] Enable nixl eplb communicator for elastic ep`, `4f423bd5`, `d9b49907 [MoE Refactor] EPLB refactoring for FusedMoE`, `a2abce64 [EPLB] Mask padding in EPLB load recording`) carries an upstream `(#NNNNN)` suffix and none is authored by the GG account. `vllm/distributed/eplb/{async_worker,eplb_communicator,eplb_state,eplb_utils,policy/*,rebalance_execute}.py` (1529 insertions / 654 deletions total) is 100% upstream drift.

Likewise `vllm/model_executor/layers/fused_moe/expert_map_manager.py` (the `ExpertMapManager` class, `determine_expert_placement_strategy`, `enable_eplb`/`num_redundant_experts` plumbing) is untouched by GG.

GG's only incidental brushes with the `enable_eplb`/`num_redundant_experts`/`expert_load` tokens:
- `07c083c6 fix(exl3): validate runtime plans and backend selection` — adds a unit-test helper that stubs `SimpleNamespace(enable_eplb=False)`; no functional EPLB change.
- `767e64a6 models: DeepSeek V2/V4 and MiniMax M3 enablement` — in `deepseek_v4.py`, adds `self.n_local_experts = self.experts.expert_map_manager.local_num_experts` right after the existing `FusedMoE(..., enable_eplb=parallel_config.enable_eplb, num_redundant_experts=eplb_config.num_redundant_experts)` call — just reads a value off the (upstream) manager, doesn't alter EPLB logic.
- `b0d9820e b12x: integrate SM120 kernel stack` — this is where GG's b12x/exl3 code *reacts to* `enable_eplb` by refusing it (see §2/§4).

**Conclusion: GG built its custom MoE kernels to sit alongside upstream's unmodified EPLB machinery, and then explicitly disabled EPLB for those kernels rather than extending it.**

---

## 2. `vllm/model_executor/layers/fused_moe/` — GG additions (exl3 / b12x / sparkinfer)

The directory's 129-file / +28673/-6906 diff vs `origin/main` is overwhelmingly the **upstream** MoE refactor (`fused_moe/oracle/*`, `runner/*`, `router/*`, `prepare_finalize/*`, `experts/*` reorg, `dc68bd8c [MoE Refactor] FusedMoE/MoERunner inversion refactor`, `d9b49907`, etc.) — GG touches upstream `layer.py`/`fused_moe.py` in only one non-PR commit (`228023b3`, itself a cherry-pick of upstream PR #38990).

GG's own additions live in **new files** introduced by the giant foundational squash commit `b0d9820e "b12x: integrate SM120 kernel stack (attention, MLA, indexer, MoE, linear)"` (67 files, +17021/-170) and refined by the `exl3:`/`feat(exl3)`/`fix(exl3)` commit series:

- **`vllm/model_executor/layers/fused_moe/b12x_moe.py`** (1647 lines) — `B12xExperts`, the fused SM120/Blackwell W4A16/NVFP4/MXFP4 MoE kernel path (via `b12x.moe.fused_moe`). `_supports_parallel_config` (line 1044):
  ```python
  return (
      not moe_parallel_config.use_ep
      and moe_parallel_config.ep_size <= 1
      and not moe_parallel_config.use_all2all_kernels
      and not moe_parallel_config.enable_eplb
  )
  ```
  → **TP-only**, EP and EPLB both hard-rejected.

- **`vllm/model_executor/layers/fused_moe/b12x_ep_moe.py`** (441 lines) — `B12xEPExperts`, the EP-specific variant (`b12x.moe.ep_moe`), replicated-input adapter. `_supports_parallel_config` (line 150):
  ```python
  return (
      moe_parallel_config.use_ep
      and moe_parallel_config.ep_size > 1
      and moe_parallel_config.tp_size == 1
      and moe_parallel_config.dp_size == 1
      and moe_parallel_config.pcp_size == 1
      and moe_parallel_config.sp_size == 1
      and not moe_parallel_config.use_all2all_kernels
      and not moe_parallel_config.enable_eplb
  )
  ```
  → EP **is** supported for b12x W4A16/NVFP4, but only "pure EP" (no TP/DP/PCP/SP mixing, no all2all kernels), and **EPLB is still explicitly disallowed**. Weight/`expert_map` are prepared once and pinned before CUDA-graph capture (`process_weights_after_loading`), and `_prepare_ep_expert_map` raises `RuntimeError("B12X EP expert_map changed before/during CUDA graph capture")` if the map ever changes — architecturally incompatible with EPLB's live weight/mapping rebalancing.
  Confirmed by dedicated tests in `tests/model_executor/layers/test_b12x_ep_moe.py`: `test_b12x_tp_and_ep_parallel_contracts_are_exclusive` and `test_b12x_ep_rejects_disruptive_parallel_variants` (parametrized over `{"dp_size":2}, {"pcp_size":2}, {"sp_size":2}, {"enable_eplb": True}` — all rejected).

- **`vllm/model_executor/layers/quantization/exl3.py`** (2447 lines, added by `08b5c182 [Quant] EXL3 (exllamav3 trellis) quantization backend`, then heavily extended by ~20 later `exl3:` commits) — the trellis-quantized (ExLlamaV3/EXL3) linear + MoE backend. Contains two MoE code paths per its module docstring:
  - "correctness" dense path via `exllamav3_ext.exl3_gemm` (`Exl3MoEMethod`), and
  - the "rank-sliced" fast path that dispatches through **B12X's unified `fused_moe`** API (commit `c7c7ea41 exl3: use Sparkinfer unified fused MoE API`, `8c4069a2 exl3: select the b12x fused trellis kernel as the primary compute path`).

  Both go through `Exl3MoEMethod.create_weights`, which contains the explicit gate (line ~1297):
  ```python
  if self.moe.moe_parallel_config.use_ep:
      raise NotImplementedError(
          "EXL3 correctness MoE currently supports TP but not expert parallelism"
      )
  ```
  → **EP is unconditionally unsupported for EXL3 MoE** (dense or rank-sliced/b12x-fused). Only TP.
  `Exl3MoEMethod` also does not override `supports_eplb` (defaults to `False` in `fused_moe_method_base.py:156`), so the generic upstream gate in `routed_experts.py:150-160` (`if enable_eplb and not quant_method.supports_eplb: raise NotImplementedError(f"EPLB is not supported {quant_method.__class__.__name__}.")`) also fires for EXL3. Same for `B12xExperts`/`B12xEPExperts` — neither overrides `supports_eplb`, so it's `False` there too.

- **`vllm/model_executor/layers/fused_moe/routed_experts.py`** (GG-touched by `08b5c182`, `b0d9820e`, `417c7e0d`) — GG changed `RoutedExperts.load_weights` to qualify `is_fused` via the matched weight-mapping entry rather than tensor rank, specifically to keep EXL3's rank-3 per-expert trellis tensors from being mangled by the fused transpose/chunk loader path. The upstream `enable_eplb`/`supports_eplb` gate this file contains (§ above) is untouched upstream logic that GG's new quant methods now flow through.

**Env-var-gated selection**: `VLLM_USE_B12X_MOE` (registered `envs.py:69/1123`) selects the b12x MoE backend when `--moe-backend auto`; referenced in `vllm/compilation/b12x_capture.py:16`, `vllm/config/virtual_tp.py:629,725`, `fused_moe/oracle/{mxfp4.py:589, nvfp4.py:296-303}`.

**No sparkinfer-specific MoE gating on EP was found beyond the above** — "sparkinfer" in this tree is the vendor/API namespace GG migrated its integration onto (`73e4a8cd sparkinfer: migrate vLLM integration to namespaced APIs`, `5898b419 refactor(mla): use SparkInfer query projection API`, `c7c7ea41 exl3: use Sparkinfer unified fused MoE API`) rather than a separate MoE kernel family; its constraints are the ones documented above under b12x/exl3.

---

## 3. DCP (Decode Context Parallel)

**DCP itself is an upstream vLLM feature**, already present at the merge-base (`vllm/config/parallel.py:297 decode_context_parallel_size: int = 1`, validated `tp_size % dcp_size == 0` since DCP reuses TP-group GPUs; upstream commits `95ed0fea DCP supports hybrid attention`, `7fc97042 Add DCP + Eagle support`, `f05603fa [Bugfix][DCP] Cast LSE to fp32`, etc.). GG did **not** invent DCP; it heavily hardened/extended DCP for its custom sparse-attention indexer (GLM-5.2 / DeepSeek-V2-style DSA) and MLA stack. This is the single largest cluster of GG commits (~50 of the 163), all under `dcp:`/`fix(dcp)`/`perf(dcp)`/`test(dcp)`/`fix(mla)`/`fix(mrv2)` prefixes.

Key GG additions, in `vllm/distributed/parallel_state.py` (function `init_distributed_environment`/DCP group init, ~lines 2080-2160) and `vllm/model_executor/layers/sparse_attn_indexer.py`:

- **Partial/replicated indexer topology** (`d45653cb [GG] feat(dcp): support partial replicated indexer topology`, `f4a5b62d`, `1e5ed5c5`, `508fe525`, `e322acd8`, `78475629`, `4d344bdd`): `_build_indexer_replica_group_ranks()` and `_validate_indexer_shard_count()` split the TP group into "indexer shards" that can be a divisor of `dcp_size` (not necessarily equal to it), building `_INDEXER_DCP` and `_INDEXER_QUERY_SPLIT` group coordinators separate from the main `_DCP` group. Controlled by `VLLM_DCP_INDEXER_SHARDS` and `VLLM_DCP_REPLICATE_INDEXER_CACHE` (also validated/consumed in `vllm/model_executor/models/deepseek_v2.py:637-665`, which raises if both are set together).
- **Owner merge** (`bedfd110 perf(dcp): merge sparse top-k by row owner`, `541bf688`, `332035a8`): `_merge_b12x_dcp_topk_by_owner` / `_merge_b12x_prefill_dcp_topk` in `vllm/model_executor/layers/sparse_attn_indexer.py` (~line 1290-1400) — instead of all-gathering full top-k candidate sets across DCP ranks, each row's designated "owner" rank receives just that row's candidates via all-to-all and does a local `run_row_topk` (from `b12x.attention.nsa_indexer.tiled_topk`), cutting collective volume. Gated by `VLLM_DCP_TOPK_OWNER_MERGE`, with `VLLM_DCP_QUERY_SPLIT`/`VLLM_DCP_QUERY_SPLIT_MIN_CONTEXT_TOKENS` controlling when queries are split across the query-split group vs. replicated (`4e6960ff perf(indexer): gate query split by context crossover`, `443e1d67 perf(indexer): restore query-split indices without score traffic`, `a87f739a Add DCP prefill query split (Fix A) and CKV gather with layer prefetch (Fix B)`).
- **CKV gather + layer prefetch** ("Fix B" in commit history: `a87f739a`, `e408f0d1 Fix B12X CKV-gather prefetch: enable it and fix correctness`, `9a017dab Fix CKV-gather small-context truncation`, `adf65d42`, `0fd51789`, `40f63da4`, `92d75ada perf(dcp): preallocate budgeted CKV layer prefetch`, `f0d49be0 fix(dcp): harden CKV prefetch workspace lifecycle`): in `vllm/v1/attention/backends/mla/b12x_mla_sparse.py`, classes `_CKVPrefetchWorkspacePool`, `_CKVPrefetchState`, `_CKVPrefetchStateRegistry`, functions `_global_causal_lens_for_ckv_gather`, `_map_global_topk_to_gathered_ckv[_kernel]` — prefetches compressed-KV (CKV) for future layers on a **dedicated side-stream/communicator** (`get_dcp_ckv_prefetch_group()`, a separate `_DCP_CKV_PREFETCH` GroupCoordinator built in `parallel_state.py` specifically because "concurrent collectives on one NCCL communicator from two streams is unsupported"). Gated by `VLLM_B12X_MLA_CKV_GATHER[_MIN_TOKENS/_MAX_TOKENS]`, `VLLM_B12X_MLA_CKV_PREFETCH_DEPTH`, `VLLM_B12X_MLA_CKV_PREFETCH_WORKSPACE_MIB`.
- Broader DCP/MLA plumbing: `2d082b4a dcp: LSE contract, global top-k, project-before-merge, hybrid a2a dispatch`, `c0b4904e kv-cache: DCP-replicated draft groups and DFlash SWA support`, `4535bee9 feat(dcp): replicate target sparse-indexer cache`, `639aeff8 fix dcp a2a buffers across full cuda graphs`, plus a whole "safe query BMM"/"fused MLA query" sub-series (`5898b419`, `86485f16`, `760363e0`, `14bbb570`, `c253e53b`, `a9ddd792`, `0e821ff7`, `dd9a91dc`, `9c8306bf`) that keep DCP's FP8 KV compatible with fused BF16 query projections (`a218cf15 fix(mla): allow fused BF16 query with DCP FP8 KV`, `6a2edcf1 fix(mla): keep DCP outputs head-major for safe BMM`).

**DCP × EP interaction:** I found **no explicit coupling or conflict check between `decode_context_parallel_size` and `use_ep`/`ep_size`/`enable_expert_parallel`** anywhere in `vllm/config/parallel.py`, `vllm/config/vllm.py`, or `vllm/distributed/parallel_state.py` (`grep` for `ep_size|use_ep|enable_expert_parallel` near `dcp` in those files returns nothing). Architecturally they're orthogonal: DCP shards attention/KV within the (global) tensor-parallel group, while EP is a per-`FusedMoE`-layer choice (`moe_parallel_config.tp_size`/`ep_size`, independent fields) about how *expert weights* are distributed. `B12xEPExperts._supports_parallel_config` does not test `dcp_size` at all, so nothing in the code stops DCP>1 with the b12x EP MoE path — but note that `B12xEPExperts` requires `moe_parallel_config.tp_size == 1`, and since DCP's global TP group must be `≥` `dcp_size` (`tensor_parallel_size % decode_context_parallel_size == 0`), running b12x-EP MoE together with DCP still requires the model's *global* `--tensor-parallel-size` (attention side) to be a multiple of your DCP degree — that's an upstream constraint, not something GG added or restricted further.

`tools/spark/README.md` (GG's own DGX-Spark 2-node launch doc, added by non-PR commit `fc5fd2f9 local: serve scripts, DGX Spark tooling, build pins`) documents operational DCP guidance but says nothing about EP: "Decode context parallelism defaults to DCP=2... DCP uses NCCL A2A through 64 scheduled tokens and NCCL AG/RS for larger batches... DSpark [speculative decode] requires DCP to be off" — that DCP-off requirement is for the DSpark draft model, not related to EP.

---

## 4. Explicit EP-unsupported asserts/warnings found

| Location | Condition | Effect |
|---|---|---|
| `vllm/model_executor/layers/quantization/exl3.py` (`Exl3MoEMethod.create_weights`, ~line 1297) | `self.moe.moe_parallel_config.use_ep` | `raise NotImplementedError("EXL3 correctness MoE currently supports TP but not expert parallelism")` — EP entirely unsupported for EXL3 MoE (both the exl3_gemm correctness path and the b12x-fused rank-sliced path share this gate). |
| `vllm/model_executor/layers/fused_moe/b12x_moe.py:1038-1044` (`B12xExperts._supports_parallel_config`) | `use_ep` or `ep_size>1` or `use_all2all_kernels` or `enable_eplb` | Kernel silently declines to be selected (not an error, but excludes b12x's TP-only fused W4A16/NVFP4/MXFP4 MoE path from EP/EPLB configs). |
| `vllm/model_executor/layers/fused_moe/b12x_ep_moe.py:140-151` (`B12xEPExperts._supports_parallel_config`) | requires `use_ep and ep_size>1 and tp_size==1 and dp_size==1 and pcp_size==1 and sp_size==1 and not use_all2all_kernels and not enable_eplb` | The EP-capable b12x kernel only activates for "pure EP"; **EPLB is explicitly excluded** even here. |
| `vllm/model_executor/layers/fused_moe/b12x_ep_moe.py` `_prepare_ep_expert_map` | expert_map tensor identity/shape changes while `_is_current_stream_capturing()` | `raise RuntimeError("B12X EP expert_map changed before/during CUDA graph capture")` — structurally incompatible with EPLB's live rebalancing. |
| `vllm/model_executor/layers/fused_moe/routed_experts.py:150-160` (upstream mechanism, but now triggered by GG's new quant methods since neither overrides `supports_eplb`) | `enable_eplb and not quant_method.supports_eplb` | `raise NotImplementedError(f"EPLB is not supported {quant_method.__class__.__name__}.")` fires for `Exl3MoEMethod` and (implicitly, since it also doesn't override the property) for the b12x experts classes. |

No markdown/docs in the GG tree explicitly say "EP is unsupported/not recommended" in prose — the constraint is expressed entirely in code (asserts/`NotImplementedError`) and unit tests, not documentation. I could not find any GG-authored `docs/` page discussing EP at all (all `docs/**/expert_parallel*.md` hits are unmodified upstream docs).

---

## 5. New env vars related to MoE parallelism / rebalancing (`vllm/envs.py`, diff +921/-261 lines vs origin/main)

Directly MoE/EP/rebalancing-relevant, all newly added by GG (`84df2e33 exl3: register the EXL3 runtime env knobs in envs.py`, and the `b0d9820e` squash):

- `VLLM_USE_B12X_MOE` (bool, default 0) — selects b12x fused MoE backend (line 69/1123).
- `VLLM_B12X_MOE_FORCE_MODELOPT_PREP` (bool) — line 92/1241.
- `VLLM_MAX_TOKENS_PER_EXPERT_FP4_MOE` — **pre-existing upstream var**, unchanged by GG (checked via `git log -p`; false-positive from initial grep, not GG's).
- `VLLM_EXL3_TRELLIS_MIN_M` / `VLLM_EXL3_TRELLIS_MAX_M` / `VLLM_EXL3_TRELLIS_BLOCK_M` / `VLLM_EXL3_PREFILL_CHUNK` / `VLLM_EXL3_PREFILL_TRELLIS` / `VLLM_EXL3_PREFILL_BLOCK_M` / `VLLM_EXL3_EXT_PATH` / `VLLM_EXL3_ABI_SHIM` (lines 2286-2294) — EXL3 trellis GEMM tuning/runtime knobs (not EP-specific; TP/M-bucketing tuning).

DCP-related (not strictly EP, but adjacent, listed since they govern MoE-adjacent sparse-indexer sharding under the same parallel topology as EP would use):
`VLLM_DCP_PROJECT_BEFORE_MERGE[_MIN_PREFILL_TOKENS]`, `VLLM_DCP_A2A_MAX_TOKENS`, `VLLM_DCP_A2A_LARGE_BACKEND`, `VLLM_DCP_SHARD_DRAFT`, `VLLM_DCP_REPLICATE_INDEXER_CACHE`, `VLLM_DCP_INDEXER_SHARDS`, `VLLM_DCP_GLOBAL_TOPK`, `VLLM_DCP_QUERY_SPLIT[_MIN_CONTEXT_TOKENS]`, `VLLM_DCP_TOPK_OWNER_MERGE`, `VLLM_USE_B12X_DCP_A2A`, `VLLM_B12X_MLA_DCP_GATHER_IN_WORKSPACE`, `VLLM_B12X_MLA_CKV_GATHER[_MIN_TOKENS/_MAX_TOKENS]`, `VLLM_B12X_MLA_CKV_PREFETCH_DEPTH`, `VLLM_B12X_MLA_CKV_PREFETCH_WORKSPACE_MIB` (all in the same `envs.py:60-92, 1092-1241` block).

I found **no `B12X_*`, `SPARKINFER_*`, or `EXL3_*` (non-`VLLM_`-prefixed) env vars** used for MoE parallelism itself — the one bare non-`VLLM_` var found, `B12X_MLA_DCP_GATHER_IN_WORKSPACE`, is only a legacy fallback read inside the `VLLM_B12X_MLA_DCP_GATHER_IN_WORKSPACE` lambda (`os.getenv("VLLM_B12X_MLA_DCP_GATHER_IN_WORKSPACE", os.getenv("B12X_MLA_DCP_GATHER_IN_WORKSPACE", "0"))`), and it's about DCP/MLA workspace gather, not EP/rebalancing.

---

## 6. `rust/src/`

This directory is **not GG-authored** — it was added to vLLM by upstream commit `39910f2b "[Rust Frontend] Move code from vllm-frontend-rs (#43283)"` (post-dating your baseline `origin/main`, hence it "looks GG-only" only because the merge-base predates it). Per `rust/README.md`: it's **`vllm-frontend-rs`**, "a Rust drop-in alternative frontend for vLLM" — a Cargo workspace of crates (`vllm-cmd`/`vllm-rs` CLI, `vllm-server` axum OpenAI-compatible HTTP API, `vllm-chat` templating/tool-parsing, `vllm-text` tokenizer/detokenizer, `vllm-llm` token-in/out facade, `vllm-engine-core-client` ZMQ+MessagePack transport to the headless Python engine, plus `bench`/`metrics`/`mock-engine`/`managed-engine`/`parser` crates). It runs as a Python-supervised subprocess (`VLLM_USE_RUST_FRONTEND=1`) and talks to the Python engine core over ZMQ — it is a **northbound serving/API layer rewrite**, not GPU code. `git grep` for `nccl|all_reduce|allreduce|expert|moe` inside `rust/src` turns up only test fixtures and a `glm45_moe.rs` **tool-call parser benchmark** (named after the GLM-4.5-MoE model, i.e. reasoning/tool-call text parsing, not fused-MoE math). **It has no relationship to collectives, EP, EPLB, or MoE execution** — GG did not modify or extend it beyond whatever ordinary rebase drift it picked up from upstream (no non-`(#NNNNN)` commits touch `rust/`).

---

## What I could NOT find

- Any GG-authored change inside `vllm/distributed/eplb/` itself (confirmed absence, not just "not found by search").
- Any markdown/docs prose (as opposed to code asserts/tests) in the GG tree stating EP is unsupported/not recommended.
- Any explicit code-level coupling/exclusion between DCP (`decode_context_parallel_size`) and EP (`use_ep`/`ep_size`/`enable_expert_parallel`) — they appear to be treated as independent, freely-combinable axes; the only real interaction is that DCP's `tp_size % dcp_size == 0` requirement bites when a b12x-EP MoE layer's own `moe_parallel_config.tp_size` is forced to 1 (that's about the MoE layer's internal TP, not the model's global attention TP, so it does not by itself forbid DCP+EP together, but I could not run the code to confirm no other subtlety exists at scheduler/model-runner level).
- Sparkinfer-specific (as opposed to b12x-specific) EP restrictions — in this codebase "sparkinfer" denotes an API namespace GG's MoE/MLA code migrated onto, not a separate MoE kernel implementation with its own EP policy.
- `VLLM_MAX_TOKENS_PER_EXPERT_FP4_MOE` turned out to be a pre-existing upstream var (initially flagged by a keyword grep, then ruled out via `git log -p`) — noting this so it isn't mistaken for a GG addition.
# Is the fungible-quant design topology neutral?

Date: 2026-08-11. Method: code reading + cheap CPU probes against the live tree
(`/home/mbelleau/src/gg-vllm`, branch `fq/m1-stats-collector`) and the local
segment artifacts. No GPU work. Every claim below carries a `file:line`.

## The one-line answer

**The *policy* layer is topology neutral and provably so. The *artifact* layer is
frozen to TP4 and cannot be made neutral without re-quantizing. The *runtime
loop* is neutral across TP only, degrades to observe-only under PP, and is
refused outright under EP and DP.**

The design splits cleanly in two, and the two halves have opposite answers:

| Layer | Neutral? | Why |
|---|---|---|
| `fq-policy/2` documents, `policy.decide`, decision digests | **Yes, enforced** | logical expert ids only; `store.validate_policy` bans `rank`/`world_size`/`tp`/`device` (`store.py:52-54`) |
| Segments / manifest / tensor names | **No — TP4-frozen** | each of the 4 rank slices is an *independent* EXL3 quantization with its own H-side rotation (measured, below) |
| Stats collector + interval loop | **TP only** | correct under TP; fails closed under PP; unreachable under EP/DP |
| Swap engine (`swap.py`) | **TP only, and not yet wired** | takes a `rank` argument it never derives; no cross-rank agreement check on the *effective* plan |

---

## 1. Segments: does `rank_sliced_tp4` encode a TP4-specific slicing?

**Yes, fundamentally.** This is the single most important finding and it is
stronger than the naming suggests.

### 1.1 What the name says

`fq_repack.py:58` hardcodes `DEFAULT_LAYOUT = "rank_sliced_tp4"`, stamped into
every segment's `__metadata__` (`fq_repack.py:432`) and into the family manifest
(`fq_repack.py:322`). The docstring is explicit that this is verbatim
passthrough: *"v1 layout keeps the source's rank-sliced granularity verbatim
(layout=rank_sliced_tp4); unsharding is a later, T4-verified upgrade"*
(`fq_repack.py:10-11`). `repack_layer` (`fq_repack.py:389-467`) copies source
bytes with `mm[body_off + a : body_off + b]` — it never reshapes, merges or
splits anything.

The `.rank{r}.` segment is load-bearing in four separate regexes that must all
agree:

- `fq_repack.py:50-52` (producer)
- `fragments.py:89-91` (resolver)
- `progressive.py:70` (`_RANK_SEG_RE`, loader filter)
- `swap.py:90-92` (`tensor_name`, swap staging)
- `exl3.py:157-160` (`_RANK_SLICED_WEIGHT_RE`, the model side)

### 1.2 What the checkpoint says

The assembled artifact's own config is unambiguous
(`/home/mbelleau/glm52-k3-assembled/config.json`, `hybrid_tr3_tail`):

```json
"tp": 4,
"slicing": {
  "gate_proj": "TP4 N-slice: rank r owns output rows [512r,512r+512)",
  "up_proj":   "TP4 N-slice: rank r owns output rows [512r,512r+512)",
  "down_proj": "TP4 K-slice: rank r owns input columns [512r,512r+512)"
}
```

`_configure_rank_sliced` requires the `tp` key to be present
(`exl3.py:452-462`), and `create_weights` hard-refuses a mismatch:

```python
checkpoint_tp = int(self.quant_config.rank_sliced_metadata["tp"])
if checkpoint_tp != layer.exl3_tp_size:
    raise ValueError("rank-sliced EXL3 checkpoint TP does not match runtime: ...")
```
(`exl3.py:1311-1317`)

So a TP2 or TP8 serve of a TP4 segment family does not *silently* misbehave —
it refuses at model-construction time. Good. But *can* it be made to work?

### 1.3 Why it cannot merge: the rotations are per-rank, and they differ

EXL3 stores, per (expert, projection, rank): a `trellis` block-coded weight, an
input rotation `suh`, an output rotation `svh`, and an `mcg` codebook scalar.
For GLM-5.2 (hidden 6144, intermediate 2048, TP4 slice 512), one expert's tensor
set is:

```
gate_proj.rank{r}.suh     F16 [6144]      <- H-side (input),  NOT split by TP
gate_proj.rank{r}.svh     F16 [512]       <- I-side (output), split by TP
gate_proj.rank{r}.trellis I16 [384,32,48] <- [H/16, I_part/16, 16*K]
down_proj.rank{r}.suh     F16 [512]       <- I-side (input),  split by TP
down_proj.rank{r}.svh     F16 [6144]      <- H-side (output), NOT split by TP
down_proj.rank{r}.trellis I16 [32,384,48]
```
(measured from `/home/mbelleau/fq-primed/segments-342/expanded/layer-003.k3.safetensors`)

If the four ranks were slices of *one* quantization, the H-side vectors
(`gate.suh`, `up.suh`, `down.svh` — all 6144-long, on the un-split axis) would be
**identical** across ranks. They are not. sha256 of each, layer 3, from three
independent artifact families:

| tensor | rank0 | rank1 | rank2 | rank3 |
|---|---|---|---|---|
| `shared_h.gate_proj.suh` (willfalco 3.42) | `a1bb919b…` | `65de53db…` | `a894dc9a…` | `e9a0c0b4…` |
| `shared_h.down_proj.svh` (willfalco 3.42) | `63bffbc2…` | `ee4ea9df…` | `2e6bbe51…` | `3a7b119e…` |
| `e0.gate_proj.suh` (our GLM-5.2 K3 encode) | `daca11034c` | `b3c3428053` | `d508bd352a` | `cc7ca93c0d` |
| `e0.down_proj.svh` (our GLM-5.2 K3 encode) | `81b22d1706` | `c2d14d6649` | `3b6938dd4c` | `e58df206dc` |
| `e0.gate_proj.suh` (Fruit proxy K2) | `d42ad3c970` | `f5d7c83c55` | `261638cff5` | `9041a884bd` |

(sources: `/home/mbelleau/fq-primed/segments-342/shared-h/layer-003.shared.safetensors`,
`/home/mbelleau/fq-segments/GLM-5.2-EXL3-FQ/layer-003.k3.safetensors`,
`/home/mbelleau/fq-0c/fruit-segments/layer-003.k2.safetensors`)

Our own verification tool confirms the semantics: `decode_proj`
(`fq_verify.py:836-852`) dequantizes **each rank slice separately** through its
own `LinearEXL3(suh=…, svh=…, trellis=…, mcg=…)` and only then concatenates in
the dense fp16 domain:

```python
for slot in slots:
    lin = LinearEXL3(None, suh.numel(), svh.numel(), suh=suh, svh=svh, ...)
    ws.append(lin.get_weight_tensor())
cat_dim = 0 if proj == "down_proj" else 1
return torch.cat(ws, dim=cat_dim).T.contiguous()
```

Concatenation happens in fp16, never in the bit domain. That is the whole
answer: **the four slices are four independent quantizations that happen to
tile one weight matrix.**

### 1.4 The contrast that proves it is not intrinsic to EXL3

EXL3's Hadamard is block-diagonal with block 128 (`exl3.py:69`), so a
*monolithic* EXL3 MoE checkpoint **can** be re-sliced to any TP at load time by
plain `narrow` on 128-aligned boundaries — `_shard_tensors_for_tensor_parallel`
does exactly that (`exl3.py:1450-1477`), slicing `w13_svh`/`w13_trellis` (dim 1)
and `w2_suh`/`w2_trellis` (dim 0) while **leaving `w13_suh` and `w2_svh`
untouched** — precisely because in a monolithic quant the H-side rotations are
shared across all ranks. `_slice_exl3_tensor` enforces the 128 alignment
(`exl3.py:998-1012`).

And that path is explicitly switched off for us:

```python
if self.quant_config.rank_sliced_metadata is None:
    self._shard_tensors_for_tensor_parallel(layer)
```
(`exl3.py:1396-1397`)

So: EXL3 is TP-flexible; *our source quant* is not, because it was produced as
four separate quantizations (`producer: encode_b300.py`, per the same
`hybrid_tr3_tail`).

### 1.5 What is actually possible, precisely

| Transform | Possible in the bit domain? | Why |
|---|---|---|
| TP4 → TP4 | yes (today) | identity |
| TP4 → TP8, TP16 | **arithmetically yes, not implemented** | splitting rank *r*'s 512-wide slice into 2×256 / 4×128 keeps the H-side rotation intact and splits the I-side one at 128-aligned boundaries (`exl3.py:998-1012`); trellis tiles are 16-wide (`[H/16, I/16, 16K]`) so the cut is whole-tile. Blocked at TP32 (64 < 128). |
| TP4 → TP2, TP1 | **no** | merging needs two slices to share the un-split-axis rotation; they measurably do not (§1.3). Requires dequantize → concat → **re-quantize**. |
| TP4 → any, via re-encode | yes | that is just "make a new artifact family" |

**Verdict on question 1: fundamental, not incidental — for merging.** The
`rank_sliced_tp4` label is honest. Downward reslicing (TP8/TP16) is incidental
and fixable; upward merging is not fixable without re-quantization, and the fq
tooling has no dequant/requant path (`fq_repack.py` is byte-verbatim by
construction, `predicate: repack-of`).

Two incidental TP4-isms worth naming separately, both cheap to fix:

- `fq_verify.py:908` hardcodes `range(4)` when reassembling slices for the
  similarity check. Should read `tp` from the manifest/config.
- The five regexes in §1.1 all *require* a `.rank{r}.` segment. A monolithic
  (tp=1) family would not match any of them. Making the segment optional is a
  small, mechanical change — but pointless until a monolithic mixed-K prepare
  path exists (`_prepare_mixed_rank_sliced_weights` is gated on
  `rank_sliced_metadata`, `exl3.py:1408-1409`).

---

## 2. Expert parallelism (EP)

**Known-broken by explicit refusal, in two places, and it never reaches our
code.**

```python
if self.moe.moe_parallel_config.use_ep:
    raise NotImplementedError(
        "EXL3 correctness MoE currently supports TP but not expert parallelism")
```
(`exl3.py:1297-1300`, at `create_weights`)

```python
if layer.expert_map is not None:
    raise NotImplementedError("EXL3 MoE expert maps/EPLB are not supported")
```
(`exl3.py:2356-2357`, at `apply`)

Even with those removed it would still fail: under EP vLLM sets the MoE's
`tp_size=1` (`config.py:1242-1246`), which then trips the checkpoint-TP gate at
`exl3.py:1311-1317` (`checkpoint=4, runtime=1`).

### Does the stats collector double-count or miss experts under EP?

**Neither — the collector is EP-correct in the counting domain; it is the
*policy → weights* mapping that would break.**

- The capture fn histograms `topk_ids` over `[0, global_num_experts)`
  (`stats.py:130-156`), and it is bound to the **router**
  (`integration.py:153-154`), which emits *global logical* ids before any
  dispatch. So every rank sees every expert id it routed to — no misses, no
  double counting within a rank. `stats.py:33-35` states the contract:
  *"num_experts: logical expert count per MoE layer (EP=1 → logical == physical;
  D4 keeps the policy domain logical)"*, and `integration.py:149` reads
  `global_num_experts`, not the local count.
- Under EP *without* DP all TP/EP ranks process the same token batch, so all
  ranks compute the same histogram — agreement holds for the same reason it
  holds under TP.
- What *would* break is downstream: `_prepare_mixed_rank_sliced_weights` builds
  its tiers from `layer.local_num_experts` (`exl3.py:1574`), so `tier_ids` would
  be **local** ids, while `fq-policy/2` and `SwapPlan` speak **global** ids. And
  `SwapEngine._validate_layer` requires the tier orderings to be a partition of
  `[0, num_global_experts)` (`swap.py:713-718`) — under EP each rank holds only
  a slice, so that assertion fails closed. Latent, unreachable today.

**Verdict: fundamental for the current mixed-trellis kernel (its slabs are
built over the full global expert set); the collector half is already neutral.**
Making EP work needs (a) a per-rank local/global expert-id map threaded through
`policy` → `swap`, and (b) an EP-aware mixed-trellis prepare. Neither is a
small change.

---

## 3. Data parallelism (DP)

**Known-broken, and for a non-obvious reason: DP changes the MoE's TP size even
when EP is off.**

`FusedMoEParallelConfig.make` flattens TP across DP *unconditionally*, before
the `use_ep` branch:

```python
flatten_tp_size = dp_size * pcp_size * tp_size
...
tp_size, tp_rank = FusedMoEParallelConfig.flatten_tp_across_dp_and_pcp(...)
if not use_ep:
    return FusedMoEParallelConfig(tp_size=tp_size, tp_rank=tp_rank, ...)
```
(`config.py:1119-1127`, `1219-1230`)

So `--tensor-parallel-size 4 --data-parallel-size 2` (no EP) gives the MoE
`tp_size=8`, which `exl3.py:1311-1317` refuses against `checkpoint=4`. There is
a second, quieter inconsistency underneath: weight-name filtering uses
`get_tensor_model_parallel_rank()` (`exl3.py:771`), which is 0..3, while
`exl3_tp_size` would be 8 — the two disagree about what "rank" means.

Beyond the refusal, DP breaks the loop's core assumption. The T6 property is
*"every rank computes the same swap list because it sees the same inputs"*; DP
replicas each schedule their **own** token batch, so each router sees a
different sample. Our own test names this exact hazard as the saboteur case:

> *"Saboteur rank: it observed its OWN shard's routing sample instead of the
> logical global one — the exact bug T6 exists to catch."*
> (`tests/exl3_fungible/test_cross_rank_t6_cpu.py:159-163`)

and proves ±5% per-expert noise diverges the decision digest in ≥25% of
intervals (`test_cross_rank_t6_cpu.py:296-307`).

The lead-rank machinery is also DP-blind: `rank = dist.get_rank()`
(`integration.py:103`) is the **global** rank and `is_lead = rank == 0`
(`loop.py:722`), so with DP only replica 0 would ever persist — while both
replicas share one `PolicyStore` path (`loop.py:723`, keyed by manifest only).

**Verdict: incidental but deep.** DP-neutrality needs a routing-stats
all-reduce across the DP group before `decide()` (making the observation
logically global again), plus a DP-aware lead election, plus a DP-aware store
key — *and* it still needs the artifact question of §1 solved, because the MoE
TP size changes.

---

## 4. Pipeline parallelism (PP)

**Untested and silently degrading: the weights load fine, the fq loop
self-disables.**

The weight side is genuinely PP-neutral. PP does not touch
`moe_parallel_config.tp_size`, and `normalize_rank_sliced_weight_name`
(`exl3.py:764-773`) filters on `get_tensor_model_parallel_rank()` — the rank
*within* the TP group — which is 0..3 on every PP stage. So a TP4×PP2 serve
loads the right slices on all 8 GPUs.

The loop side does not survive. `maybe_init_fq_collector` walks
`runner.model.modules()` (`integration.py:141-145`), which on a PP stage
contains only that stage's layers; `MoERunner.layer_id` is
`extract_layer_index(self.layer_name)` (`moe_runner.py:933-937`) — the **global**
model layer index. So the collector's layer set is a strict subset of the
policy's, and `_map_collector_layers` (`loop.py:348-369`) raises.

Measured directly (probe script, CPU, throwaway):

```
== case A: collector has ALL layers (TP / single-stage) ==
   OK  layers= [0..7]  map= {0:0, ..., 7:7}
== case B: collector has ONLY stage-0 layers, policy has all (PP=2) ==
   RAISED ValueError : FQ loop: cannot map policy layers [0..7] onto
                       collector layers [0, 1, 2, 3]
== case C: equal counts, different ids (positional fallback) ==
   OK  policy layers= [0,1,2,3] -> collector ids {0:10, 1:11, 2:12, 3:13}
       (SILENT MISMAP, warning only)
```

Case B's `ValueError` propagates out of `build_from_env` into
`maybe_init_fq_state`, which catches everything and returns a bare collector
(`integration.py:111-117`):

> *"FQ loop init failed — falling back to collector-only"*

So under PP you get M1 observability and **no policy engine at all**, on every
rank, with one exception line in the log. That is fail-safe, not fail-loud.

**Does cross-rank agreement hold under PP?** The question is malformed as posed:
under PP different ranks hold different layers, so there is nothing for them to
agree *about* per-layer. The correct shape is agreement *within* each PP stage's
TP group, over that stage's layer rows. Nothing in the code expresses that: the
policy document is model-global (`loop.py:299`), `is_lead` is global rank 0
(`loop.py:722`), i.e. stage 0 only — a lead that cannot observe half the layers
it would be persisting decisions for.

Case C is the one genuine hazard here: when the collector's layer count happens
to match the policy's but the ids differ, `loop.py:363-366` maps **positionally
with a `logger.warning` and continues**. Under a PP layout with equal-sized
stages plus a per-stage policy that would silently attribute stage-1 statistics
to stage-0 layer rows. It should be a hard error, or the policy should carry the
stage's layer ids and be matched on them.

**Verdict: incidental and fixable.** Concretely: (1) intersect the policy rows
with the collector's layer ids instead of demanding a superset; (2) elect a lead
per PP stage (`is_lead = get_tensor_model_parallel_rank() == 0`) and shard
`decisions/` by stage; (3) delete the positional fallback.

---

## 5. What *is* neutral — and it is the important half

### 5.1 The policy domain is topology-free by construction, and it is enforced

`store.py:11-13`: *"Policies are keyed by logical expert id only — no
rank/world_size/tp/device fields (D4)."* Not a comment — a check:

```python
for banned in ("rank", "world_size", "tp", "device", "devices"):
    if banned in doc:
        raise ValueError(f"policy must be topology-neutral; found {banned!r}")
```
(`store.py:52-54`, tested at `test_cross_rank_t6_cpu.py:325-337`)

### 5.2 Cross-rank agreement is a determinism argument, not a collective

There is **no** collective anywhere in the package — `grep -n "all_reduce|dist\.|barrier|torch.distributed"` over
`exl3_fungible/*.py` returns exactly one hit, `integration.py:101-103`, and that
is only reading the rank number. `decision_sha` (`loop.py:454-457`) is *logged*
(`loop.py:511-514`) and never compared at runtime.

That is a deliberate design choice and it is backed by real evidence:

- **T6, offline:** 4 spawned interpreters with distinct `PYTHONHASHSEED`,
  distinct global RNG seeds and distinct `RANK`/`LOCAL_RANK`/`CUDA_VISIBLE_DEVICES`
  produce bit-identical 50-interval trajectories, including policy hashes and
  decision-record hashes (`test_cross_rank_t6_cpu.py:278-293`). Non-vacuity is
  itself tested (`:265-275`), and the digest is proven to *catch* divergence
  (`:296-307`).
- **T6, live TP4:** *"every interval line is emitted identically by all four TP
  ranks (`Worker_TP0..TP3`)"* — `runs/m2-dryrun/report.md:65-67`. Caveat stated
  in that report: the run produced 0 swaps, so it exercises the decision path,
  not the apply path.
- **Membership agreement across a live apply, TP4:** the M3 reload evidence has
  all four ranks reporting the same `policy_sha` at boot, after swap 1 and after
  swap 2:

  | file | rank 0 | rank 1 | rank 2 | rank 3 |
  |---|---|---|---|---|
  | `state-boot-042b.json` | `29bb0a958019cd1f` | `29bb0a958019cd1f` | `29bb0a958019cd1f` | `29bb0a958019cd1f` |
  | `state-after-swap1.json` | `44248b1d317ce819` | `44248b1d317ce819` | `44248b1d317ce819` | `44248b1d317ce819` |
  | `state-after-swap2.json` | `29bb0a958019cd1f` | `29bb0a958019cd1f` | `29bb0a958019cd1f` | `29bb0a958019cd1f` |

  (`runs/m3-reload/*.json`; `runs/m3-reload/report.md:107,125`)

### 5.3 The M3 apply path is the topology-correct pattern

`fq_reload` is the one place that gets multi-rank apply right, and it is worth
naming as the template:

- one externally-decided policy, broadcast identically to every worker via
  `collective_rpc fq_reload_experts` (`fq_reload.py:173`, driver at `:445-450`);
- each worker reads **its own** slice using the authoritative source,
  `rank = int(layer.exl3_tp_rank)` (`fq_reload.py:206`) — not `dist.get_rank()`;
- an explicit cross-rank verification RPC, `fq_expert_state`
  (`fq_reload.py:147-171`), whose driver **enforces** agreement:
  `shas = {r["policy_sha"] for r in res["results"]}; return 0 if len(shas)==1`
  (`fq_reload.py:470-476`).

### 5.4 Multi-rank filesystem hygiene

`FragmentResolver._atomic_write` is pid-qualified and therefore safe with N
worker processes sharing one cache root (`fragments.py:818-822`).

---

## 6. Rank assumptions in `swap.py`

The swap engine is correct *given* a correct `rank`, but it never derives one
and nothing checks its output for cross-rank consistency. It is also **not wired
to live layers yet** — `apply_fn` is never bound anywhere in the tree
(`grep -rn apply_fn vllm/` hits only `loop.py`), and `apply_mode=atomic` logs
*"recording proposals only (M4 live wiring pending)"* (`loop.py:547-551`). So
everything here is a pre-wiring finding, cheap to fix now.

1. **`rank` is an un-derived constructor argument defaulting to 0**
   (`swap.py:630, 641`), threaded into every fragment read
   (`swap.py:869-870`). If it is wired the way `integration.py:103` wires the
   loop — `dist.get_rank()` — then TP4×PP2 gives ranks 4..7 on stage 1, which
   ask for `.rank4.`…`.rank7.` tensors that do not exist. That surfaces as a
   bare `KeyError` (`swap.py:290-292` local source, `swap.py:386-391` resolver
   source), *not* as the droppable `FragmentUnavailable` — so it kills the
   worker rather than pending the promotion. **Fix: derive from
   `layer.exl3_tp_rank`, as `fq_reload.py:206` already does.**

2. **`on_unavailable="drop"` makes the effective plan rank-local, with no
   agreement check.** A pair whose fragments cannot be supplied is dropped and
   the *surviving* pairs become `StagedBatch.plan` (`swap.py:866-883, 959-971`).
   Fragment availability is per-rank state (cache contents, a flaked HTTP
   range), so two TP ranks can legitimately end an interval with **different
   memberships** — and `apply()` contains no barrier, no digest exchange, no
   check of any kind (`swap.py:975-1090`). Today this is the most dangerous
   *incidental* topology gap in the codebase: it converts a transient IO failure
   into silent cross-rank membership divergence. **Fix: after staging, all-reduce
   a digest of `staged.plan` across the TP group and drop the pair everywhere or
   nowhere.** (The out-of-band M3 path already has the equivalent check —
   `fq_reload.py:470-476`.)

3. **`policy_store.commit` inside `apply()` is not lead-gated**
   (`swap.py:1076-1078`). The loop path is safe because only the lead gets a
   store at all (`loop.py:568, 736`), but the engine API invites every rank to
   pass one. Two aggravating factors if that happens:
   `PolicyStore._atomic_write` uses a **non**-pid-qualified temp name
   (`store.py:69`, contrast `fragments.py:820`), and history rotation numbers
   generations with `gen = len(glob("*.json"))` (`store.py:85`) — both race
   across processes. **Fix: assert lead-only in `apply()`, and pid-qualify
   `store._atomic_write`.**

4. **`ExpertStage` geometry is per-partition, undocumented as such.**
   `ExpertStage(bits, hidden_size, intermediate_size)` builds
   `trellis[H/16, I/16, 16K]` (`swap.py:119-145`) where `intermediate_size` must
   be `intermediate_size_per_partition` (512 at TP4), not the model's 2048.
   Passing the wrong one fails closed at the shape check
   (`swap.py:305-309`, `swap.py:397-401`) — but the parameter name invites it.
   **Fix: rename to `intermediate_size_per_partition`.**

5. **`MixedLayerState` assumes the full global expert set** —
   `num_global_experts = global_to_combined.numel()` (`swap.py:571-573`) and
   `_validate_layer` requires `tier0_globals + tier1_globals` to be exactly
   `range(num_global_experts)` (`swap.py:713-718`). Correct under TP, wrong
   under EP (see §2). Fails closed.

Minor, benign: `EncodeQueue.enqueue` dedups only against its own in-memory view
(`lazy_encode.py:136-157`), so N ranks missing the same fragment append N
identical lines. Parse-time dedup absorbs it (`lazy_encode.py:93-95`); the only
cost is a fatter JSONL.

---

## 7. Bottom line

### Supported today (evidence-backed)

- **TP4, single node, no PP/DP/EP.** This is the only tested configuration and
  it is tested well: TP4 boot (`runs/m0-boot-gate.md:13`), TP4 dryrun loop with
  four-rank identical interval lines (`runs/m2-dryrun/report.md:65-67`), TP4
  live membership reload with four-rank digest agreement
  (`runs/m3-reload/report.md:107,125` and the `state-*.json` table in §5.2), TP4
  serve script (`runs/m5-serve/serve-glm52.sh:111`).
- **Any TP that equals the checkpoint's `tp`.** Nothing in fq is TP4-specific
  *as code* except `fq_verify.py:908`; a `tp=8` source family would work
  end-to-end unchanged. The TP4-ness lives in the artifacts, not the algorithm.

### Untested

- **TP4 × PP.** Weights should load (§4); the fq loop provably will not arm
  (measured, §4) and degrades to collector-only. No PP run has ever been
  attempted.
- **Downward reslicing TP4 → TP8/TP16.** Arithmetically sound (§1.5), zero
  implementation, zero evidence.

### Known-broken

- **EP, any size.** Refused twice, `exl3.py:1297-1300` and `exl3.py:2356-2357`;
  would also fail the TP gate. Fundamental for the current kernel.
- **DP > 1, with or without EP.** Refused via the TP-flattening at
  `config.py:1125` → `exl3.py:1311-1317`; and even if allowed, per-replica
  routing samples break the T6 determinism argument (`test_cross_rank_t6_cpu.py:159-163`).
- **TP ≠ checkpoint `tp` (e.g. TP2 or TP8 on TP4 segments).** Refused at
  `exl3.py:1311-1317`. Merging (TP2/TP1) is **fundamentally** impossible without
  re-quantizing — the per-rank H-side rotations differ (measured, §1.3).
- **K5 as a mixed tier on SM120** — unrelated to topology but part of the same
  "what actually runs" picture; see `k5-shared-memory-limit.md`.

### Ranked recommendations

1. **Cross-rank agreement on the *effective* swap plan** before M4 goes live
   (§6.2). This is a correctness hole, not a feature gap, and it is cheapest to
   close now while nothing is wired.
2. **Derive `SwapEngine.rank` from `layer.exl3_tp_rank`** (§6.1) so a future
   PP deployment cannot ask for `.rank7.`.
3. **Delete the positional layer-mapping fallback** (`loop.py:363-366`) — a
   silent mismap is strictly worse than the `ValueError` two lines below it.
4. **Make PP degrade loudly and then correctly**: intersect policy rows with the
   collector's layers, elect the lead per TP group rather than globally (§4).
5. **Rename `intermediate_size` → `intermediate_size_per_partition`** in
   `ExpertStage`/`SwapEngine`, and lead-gate `policy_store` in `apply()` (§6.3-4).
6. **State the artifact contract in the manifest**, not only in the layout
   string: `fq-manifest/1` carries `layout: rank_sliced_tp4` but no machine-
   readable `tp`, so a consumer must string-match. Add `"tp": 4` to
   `build_manifest` (`fq_repack.py:314-333`) and have `fq_verify` read it
   instead of `range(4)`.

### Suggested tests (none written here — this document changes no behaviour)

- `test_loop_cpu.py`: assert `_map_collector_layers` raises on a strict-subset
  collector (the PP case) and — after fix 3 — on an equal-length/different-id
  collector.
- `test_swap_cpu.py`: assert two `SwapEngine`s given the same plan but
  different per-rank fragment availability produce detectably different
  `staged.plan` — the test that fix 1 must make pass.
- A manifest test asserting `tp` is present and that `fq_verify` honours it.

---

*Terminology note for the operator: "topology neutral" is being used for two
different things in this project, and conflating them is the source of the
confusion. **Decision neutrality** — every rank reaches the same answer without
talking — is real, enforced, and proven (§5). **Artifact neutrality** — the same
bytes serve at any TP — is false and, for TP reduction, unfixable without
re-quantizing (§1). The design deliberately bought the first at the price of the
second: `store.validate_policy` bans topology from the policy precisely so the
topology can live entirely in the artifact.*

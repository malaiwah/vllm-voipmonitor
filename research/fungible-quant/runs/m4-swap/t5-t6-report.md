# M4 — T5 (torn-update) + T6 (cross-rank agreement) verdicts — 2026-08-10

Closes the two remaining M4 test gates from `implementation/03-testing-validation.md`
and wires the swap engine's fragment source to loader v2's `FragmentResolver`.
Code: gg-vllm `fq/m1-stats-collector`, commits `aaf333e99` + `163c9ee7d`
(`exl3_fungible/swap.py`, `tests/exl3_fungible/test_swap_t5_{cpu,gpu}.py`,
`test_cross_rank_t6_cpu.py`, `test_swap_resolver_cpu.py`, shared harness in
`toy_segments.py`). CPU suite green package-wide: **121 passed / 11 skipped**
(77 at the M4 report, +20 from the concurrent M2 loop work, +24 here).

GPU work on **GPU 4** (RTX PRO 6000 Blackwell, SM120) through
`runs/gg-env/gg-run.sh` (r33 rootfs, torch 2.12.0+cu132), taken during an
inter-layer gap of the 0c encode ring after two consecutive polls at 0% util
/ <1 GiB (GPUs 0–3 were lock-held by the M2 dry-run agent throughout). Two
suites, back to back, both green:

* `test_swap_gpu.py` (**T3/T4 regression** — the commit path was refactored
  here, so the M4 verdicts were re-earned, not assumed): **5 passed**, apply
  window best **0.051 ms** (1 pair) / **0.302 ms** (8 pairs), stage 0.114 /
  0.640 ms, 0.046 / 0.369 MB H2D — same numbers as the M4 report within
  noise, so flip-time bookkeeping costs nothing measurable.
* `test_swap_t5_gpu.py` (**T5**): **5 passed**.

## Verdicts

| Test | Verdict | What it proves |
|---|---|---|
| **T5 — torn-update fault injection** | **PASS (bitwise, 6/6 abort points)** — with the engine change below | The commit protocol is aborted after every step k ∈ {quiesce, slabs, rotations, maps, memo, persist}. `quiesce`/`slabs`/`rotations` → forward `torch.equal` to the PRE-swap output; `maps`/`memo`/`persist` → `torch.equal` to the POST-swap output (the fresh-built layer T4 pinned). Over all six abort points exactly **two** distinct outputs exist — no third state is observable. Mirrored on CPU over every byte the kernel reads. |
| **T5 — abort + rollback** | **PASS (bitwise)** | Abort after the flip (`memo`) → the swap is committed and `apply(plan.inverse())` (fragments re-read from the artifact pair) restores the pre-swap output bitwise. Abort before the flip → the engine restores in-window and the *next* apply of the same plan still lands the full swap. |
| **T5 — non-vacuity control** | **PASS** | The same aborts with fail-atomic staging OFF produce a genuine **third** output (and a fourth: tearing grows between step 1 and step 2). The harness can see tearing, so the equalities above are not vacuous — and the restore is load-bearing, not decoration. |
| **T6 — cross-rank agreement** | **PASS (bit-identical, 4 ranks × 50 intervals)** | 4 simulated ranks in independent spawned interpreters produce byte-identical swap lists, policy hashes, decision records and decision digests across a 50-interval chained trajectory. Divergence control passes: a rank fed its own shard-local routing sample is caught by the digest. |
| **FragmentResolver-backed staging** | **LANDED (11 CPU tests)** | `ResolverFragmentSource` stages swaps through boot's supply chain (local dirs → content-addressed cache → HF/mirror sources) with the same trust filtering and sha verification, byte-identical to `LocalSegmentSource`; supply failures become pending promotions instead of failed intervals. |

## T5 — what the gate actually found

`02-swap-engine.md` §Commit protocol says intermediate states are unreachable
because the engine is quiesced. That is true *while the window holds*. T5 asks
the question the doc does not: **what is observable when control leaves the
window early?** In the v1 row-write design the two destination rows are both
live (`02 §Why row writes happen inside the quiesce window`), so steps 1–2 ARE
the tear — the displaced expert's slot already holds its successor's bytes.
Two engine changes make the T5 property hold rather than merely be asserted:

1. **The flip is the only visibility point.** Live tier orderings and the
   generation counter now commit *with* the map copy, in a `finally`.
   Previously an abort at step 4/5 (memo hook raising, `policy_store.commit`
   hitting a full disk) propagated out before the host bookkeeping ran,
   leaving flipped device maps described by stale in-memory orderings — the
   next `stage()` would then resolve slots against a membership that no longer
   existed. This was a real bug, found by the gate, not by review.
2. **Fail-atomic staging** — `stage(fail_atomic=True)` also stages both
   experts' PRE-swap encodings and the pre-swap maps; `apply()` replays that
   restore batch inside the same quiesce window if it aborts before the flip.
   Cost: one extra host-side read per expert and a second pinned staging set
   (max_pairs × 7.875 MiB); **zero** change to what the commit writes — same
   op lists, same H2D bytes (asserted). Opt-in, so the default keeps the
   02 §Why-row-writes pinned budget.

Recovery matrix, as tested:

| Abort point | Layer state after the window | Correct repair |
|---|---|---|
| quiesce (pause fails) | fully-old, nothing written | none |
| slabs / rotations, `fail_atomic=True` | fully-old (restored in-window) | none; engine reusable |
| slabs / rotations, `fail_atomic=False` | **torn** | re-apply the plan (roll forward) — tested; or restart, slabs are a cache |
| maps / memo / persist | fully-new, host view agrees | `apply(plan.inverse())` — tested bitwise |
| persist (store raises) | fully-new, `current.json` untouched | boot rehydrates the previous committed policy (T8) |

Extra leg beyond the gate: a CUDA graph captured **before** an aborted swap
replays the PRE-swap output after the restore, and the POST-swap output once
the plan is re-applied. The restore is written through the same live tensors
T3 proved are read as data — a repair the graph could not see would be worse
than no repair.

### Design consequence to carry (binds on 02)

> Quiesce makes intermediate states unobservable only for the duration of the
> window. **Every exit from the window must be consistent**: before the flip,
> restore (fail-atomic staging) or roll forward; at or after the flip, the
> swap is committed and rollback is the inverse plan. An abort is not a
> no-op. The M4 apply wiring should stage with `fail_atomic=True` in serving
> — the pinned cost is bounded and known, an unrepaired torn layer is not.

### T5 CPU half

`test_swap_t5_cpu.py` runs the same six abort points against a hand-assembled
CPU layer and compares a sha256 over *every byte the kernel would read* (both
slabs, all four combined tables, both maps) — strictly stronger than one
forward, which only reads the rows its routes touch, and it keeps the property
defended in the always-green CPU suite. 7 tests, including the multi-pair
restore, the persist-failure case and the "restore costs only reads" check.

## T6 — why four processes and not four GPUs

The claim is that every rank computes the same swap list **without a
collective**. The mechanism is not the hash — the hash was only ever evidence
(03 §T6 says so itself). The mechanism is that the policy domain is *logical*:
`fq-policy/2` documents are keyed by logical expert id and
`store.validate_policy` refuses any rank/world_size/tp/device field (D4), and
`policy.decide` is a pure function with total-order sort keys. A TP4 serve run
would evaluate that pure function four times on the same inputs; the GPUs
contribute nothing to the property, and they would *weaken* the evidence — one
parent process, shared module state, one shared RNG make accidental agreement
easy.

So the harness runs four ranks as independent spawned interpreters, each made
hostile to agreement in ways a real TP4 run is not:

* a different `PYTHONHASHSEED` per rank (verified distinct in-process) — any
  dict/set iteration-order dependence in `decide()`/`explain()` diverges;
* a different global RNG seeding (`random` + legacy `numpy.random`) — any
  accidental use of global randomness diverges;
* a different `VLLM_FQ_RANK` / `LOCAL_RANK` / `RANK` / `CUDA_VISIBLE_DEVICES`
  — any topology leak into the decision diverges;
* nothing is shared but one integer seed: each rank *reconstructs* the 50
  intervals' stats, eps, pins, dwell and guard knobs itself, so input
  construction is proven deterministic too.

The 50 intervals are a **trajectory** — each interval's swaps are applied
before the next is decided — so a single divergence at interval i cascades
through every later interval and hash. Measured on the committed scenario
(6 layers × 24 experts, N_K4 = 8): **283 swaps over 50 intervals, all 50
intervals moving, 50 distinct decision digests and 50 distinct policy
hashes**, budget invariant `n_k4 == 8` holding throughout; trajectory digest
`ec0bf749e57bc805…`. The sabotaged rank (±5% per-expert routing noise, i.e.
a rank that read its own shard's sample instead of the logical one) diverges
in **45 of 50** intervals — a uniform *scale* would not diverge at all, since
the score is linear in count and the hysteresis test is a ratio, which is
itself a useful invariance to have on record.

Compared per interval and in aggregate: the ordered swap list, the resulting
membership, `store.policy_hash` of the persisted `fq-policy/2` document, the
`decision_log.explain` record digest, and the decision sha. All four ranks
agree bit-for-bit; the trajectory digests are equal.

Two closing legs:

* **"Then kill the debug collective and rerun"** (03 §T6): the trajectory is
  recomputed in-process with no hashing, no subprocesses and no env games; the
  swap lists still match every rank. Determinism is the mechanism.
* **The digest is the shipped one**: the test cross-checks its definition
  against `loop.FungibleQuantState.decision_sha`, the digest each rank logs
  per interval in the M2 serve loop, so the serve log alone is auditable
  evidence at TP4.

### Real TP4 spot-check — not run, and why that is sufficient

GPUs 0–3 were held by the M2 dry-run agent for the whole window
(`/home/mbelleau/fq-0c/.serve-quad.lock`, taken 20:17) and 4–7 ran the encode
ring at 100% util. A TP4 spot-check would have re-evaluated the same pure
function on four ranks with the same inputs and printed four equal shas — a
strictly weaker instance of what the simulation already covers (same process
tree, shared parent env, no hash-seed or RNG hostility). The honest statement
is that T6 is a property of the policy domain's determinism and topology
neutrality, both of which are tested directly. What a TP4 run *would* add is
not agreement but the plumbing around it — that the loop reads the same stats
on every rank and that the shas actually reach the log — and that belongs to
T7's soak, where a real 4-rank serve runs anyway.

## FragmentResolver-backed swap staging

`swap.FragmentSource` had one implementation (`LocalSegmentSource`, plain file
IO over an fq-segment dir). `ResolverFragmentSource` adapts loader v2's
`FragmentResolver` to the same protocol — duck-typed (`resolve` +
`materialize`), so `swap.py` keeps its no-imports stance and still loads
standalone in the GPU harness. Swaps therefore stage through the same ladder
as boot: local segment dirs → content-addressed cache → the manifest /
`VLLM_FQ_SOURCES` chain of HF repos and trusted mirrors, with attestation
trust filtering (10 §4) and per-fragment sha256 verification. Tested:

* **byte fidelity** — resolver-staged and locally-staged applies produce
  byte-identical layer state (and the fresh-build reference);
* **remote staging** — with no local segments at all, both fragments are
  fetched through ranged reads and sha-verified before staging;
* **cache reuse** — a second resolver over the same cache dir (the other TP
  rank, or this one after a restart) stages the same swap with zero remote
  range reads;
* **trust rejection** — a signer-configured resolver refuses an unattested
  mirror, and the engine's default is to fail closed.

### Pending promotions (07 §1) — the graceful-degradation semantics

`stage(plan, on_unavailable="drop")` turns a supply failure into a **pending
promotion** instead of a failed interval: the whole pair is dropped (both
experts keep their tier, so per-layer K4 cardinality is preserved — D1), the
surviving pairs are `StagedBatch.plan`, and the reasons land in
`StagedBatch.dropped` / `ApplyReport.dropped` plus a warning line. Callers must
build the policy document to persist from `staged.plan`, never from the
requested plan — otherwise `current.json` would claim swaps that never
happened.

Droppable (supply): fragment missing from every source, untrusted attestation,
sha mismatch, and — the 07 case — a fragment the `VLLM_FQ_K_FALLBACK` ladder
substituted at a lower K, which both fails to fit a K4 slab row and *means*
the K4 encode has not landed. The resolver has already queued that encode; the
promotion simply pends until it does. Not droppable (structural, fail-closed):
missing/misshaped tensors, foreign mcg, an unregistered layer, bad residency,
or a resolver bug. Classification is by exception class name across the MRO,
so a third-party source can signal "pend this" without importing swap.py.

## Files

* `vllm/model_executor/layers/quantization/exl3_fungible/swap.py` —
  `FragmentUnavailable`, `is_unavailable_error`, `ResolverFragmentSource`,
  `DroppedPair`, `stage(fail_atomic=, on_unavailable=)`, flip-time bookkeeping,
  in-window restore.
* `tests/exl3_fungible/test_swap_t5_gpu.py` — T5 proper (5 tests, SM120).
* `tests/exl3_fungible/test_swap_t5_cpu.py` — T5 state-level matrix (7 tests).
* `tests/exl3_fungible/test_cross_rank_t6_cpu.py` — T6 (6 tests).
* `tests/exl3_fungible/test_swap_resolver_cpu.py` — resolver source (11 tests).
* `tests/exl3_fungible/toy_segments.py` — shared CPU layer-state harness,
  state fingerprint, and a tree-module loader (the gg rootfs carries a synced
  copy of the package in site-packages; tests whose verdict is about tree code
  must not silently validate that copy).

## Carried forward

* **Wire `fail_atomic=True`** into the M4 apply path when `APPLY_MODE=atomic`
  goes live, and surface `ApplyReport.dropped` as a `fq_pending_promotions`
  counter beside `fq_swaps_total` — a promotion that pends every interval is
  the supply-side analogue of the rollback-thrash signal T7 watches for.
* **`swap.py` is not in the rootfs site-packages copy** of the package that
  the serve runs import; whatever syncs the others needs to include it (and
  `decision_log.py`, `loop.py`) before `APPLY_MODE=atomic` can be exercised
  in a real serve.
* T5 leaves the **spare-slot ring** (02 §Why row writes…, "Later (optional)")
  untouched: with a spare row per tier the pre-flip window would write only
  unreferenced rows, and fail-atomic staging would become unnecessary rather
  than merely cheap. Buy it only if T7 shows the pause budget matters.

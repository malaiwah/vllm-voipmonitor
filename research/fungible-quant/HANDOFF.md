# Fungible Quant / Progressive Tensors — start-here handoff

**Written 2026-08-11 ~20:05Z**, on the instance that replaced the one pre-empted
at 19:31Z.

This assumes **you know nothing** about this project. Read §1–§4 before running
anything. §7 (Sharp edges) is the highest-value section — most of the time lost
here was lost to failures that looked exactly like success.

Every path below is absolute.

---

## 1. What this project is, from zero

### The vocabulary you need

- **MoE / expert.** GLM-5.2 is a Mixture-of-Experts model: each transformer
  layer holds 256 "experts" (small FFNs), and a **router** picks a handful per
  token. So most experts are idle for any given token, and some experts are far
  more used than others.
- **Quantization / EXL3 / trellis.** Weights are compressed to a few bits each.
  EXL3 is a trellis-coded scheme; the compressed blob for one expert at one
  bitrate is a **trellis slab** plus small rotation vectors (`suh`, `svh`).
- **K.** Shorthand for bits-per-weight. We use **K2, K3, K4, K5** = 2, 3, 4, 5
  bits. Higher K = better quality, more memory. Measured, per TP rank:
  `K2 = 2,399,244 B`, `K3 = 3,578,892 B`, `K4 = 4,758,540 B`,
  `K5 = 5,938,188 B` per expert. K3→K4 costs **1,179,648 B** per expert.
- **TP4 / rank.** Tensor parallelism across 4 GPUs. Every expert is sliced into
  4 rank-slices. All byte figures above are **per rank**.
- **Tier.** Within one layer, experts are partitioned into (at most) two
  bit-width groups: `tier0` and `tier1`, e.g. `tiers=((3, 224), (4, 32))` means
  224 experts at K3 and 32 at K4. **A layer can hold exactly two bit-widths —
  no more** (enforced at `/home/mbelleau/src/gg-vllm/vllm/model_executor/layers/quantization/exl3.py:1591`).
- **Segment / fragment.** A **segment** is a whole-layer safetensors file at one
  K (`layer-034.k4.safetensors`). A **fragment** is one expert's slice pulled out
  of it. Segments are the warm-restart asset; fragments are derived and cheap to
  rebuild.
- **Policy / posture.** A JSON document (`fq-policy/2`) mapping every
  `(layer, expert)` to its K. The live arrangement is the "posture".
- **Swap.** Moving one expert K3→K4 while another goes K4→K3, so per-layer
  counts never change. **Paired** by design.
- **Quiesce.** A brief pause of the engine during which weights may be rewritten.

### The two ideas

**Progressive Tensors.** Ship a quantized MoE as *per-expert fragments at several
bitrates* rather than one monolithic file at one bitrate. A serve then downloads
only what its policy asks for. Schemas: `fq-segment/1`, `fq-attestation/1`,
`fq-manifest/1`, `fq-policy/2`, `fq-heatmap/1`.

**Fungible quant.** While the model is serving, watch which experts actually get
routed to, then *move experts between bit-widths at runtime* — promote hot ones,
demote cold ones — without restarting, **without reallocating any tensor**, and
without invalidating CUDA graphs.

Target: **GLM-5.2**, TP4, 8× RTX PRO 6000 Blackwell (SM120).
End goal: an evidence-rich PR to vllm-project.

### Why "without reallocating" is the whole trick

CUDA graphs record kernel launches with **pointers baked in**. If you reallocate
a weight tensor, every captured graph now points at freed memory. So the swap
engine only ever rewrites *the contents of rows* and *the contents of the routing
maps* — never their addresses. That single constraint explains most of the
design, including why growth and re-preparation are hard (§8).

---

## 2. The machine

8 GPUs, ~90 GiB usable each. **This box is pre-emptible** — it can vanish at any
moment. Only `/home` survives. See §9 for recovery.

No docker/podman: namespaces are disabled. The runtime is an **extracted
container rootfs** at `/home/mbelleau/rootfs/gg-v20-r33`, entered through a shim.

---

## 3. Every path

### Code

| What | Absolute path |
|---|---|
| vLLM fork (all the code) | `/home/mbelleau/src/gg-vllm` |
| — branch / remote | `fq/m1-stats-collector` → `github.com/local-inference-lab/vllm` |
| Our module | `/home/mbelleau/src/gg-vllm/vllm/model_executor/layers/quantization/exl3_fungible/` |
| Tests (~691 CPU) | `/home/mbelleau/src/gg-vllm/tests/exl3_fungible/` |
| GG's mixed-trellis prepare | `/home/mbelleau/src/gg-vllm/vllm/model_executor/layers/quantization/exl3.py` |
| Research repo (docs, runs, results) | `/home/mbelleau/protensors-work/vllm-voipmonitor` |
| — branch / remote | `claude/gg-overview-exploration-jchgd3` → `github.com/malaiwah/vllm-voipmonitor` |
| Research root for this work | `/home/mbelleau/protensors-work/vllm-voipmonitor/research/fungible-quant` |

Files inside `exl3_fungible/` and what each is for:

| File | Role |
|---|---|
| `progressive_loader.py` | `--load-format progressive`: boot from segments + policy, no assembled checkpoint |
| `progressive.py` | segment spec/resolver, per-K byte measurement from real headers |
| `fragments.py` | multi-source resolution, trust filtering, sha verification, K-fallback ladder |
| `swap.py` | the atomic swap engine (stage → apply, CUDA-graph safe) |
| `loop.py` | the interval loop: collect stats → decide → apply |
| `policy.py` | tier constants (`K2..K5`), decide/budget-filter, policy docs |
| `integration.py` | wires the loop into the vLLM worker; async staging; cross-rank readiness |
| `admin.py` | the `/fq/*` HTTP surface and worker RPCs |
| `memory_preflight.py` | projects the mixed-K footprint **before** a load; refuses impossible policies |
| `stats.py`, `heatmap.py` | routing counters and the activation matrix endpoint |
| `store.py`, `decision_log.py` | policy persistence and per-interval decision records |
| `occupancy_table.py`, `convergence.py`, `lazy_encode.py` | composition table, convergence, on-demand encode queue |

### Scripts (all under `/home/mbelleau/protensors-work/vllm-voipmonitor/research/fungible-quant/runs/`)

| Script | Absolute path |
|---|---|
| Rootfs runner shim | `/home/…/runs/gg-env/gg-run.sh` |
| **Deploy source → rootfs** | `/home/…/runs/gg-env/deploy-fq.sh` |
| Serve launcher (both instances) | `/home/…/runs/m5-serve/serve-demo1.sh` |
| **Reap GPUs by device** | `/home/…/runs/m5-serve/reap-devices.sh` |
| Health sweep | `/home/…/runs/health/sweep.sh` |
| Load driver | `/home/…/runs/m5-serve/swap_evidence.py` |
| Heatmap capture / render | `/home/…/runs/m5-serve/capture_heatmap.sh`, `render_heatmap.py` |
| EN/ZH contrast | `/home/…/runs/m5-serve/lang_contrast.py` |
| Re-tier verifier | `/home/…/runs/m5-serve/verify_retier.py` |
| Byte-based BT metrics | `/home/…/runs/m5-serve/bt_metrics.py` |
| M3 reload-under-quiesce | `/home/…/research/fungible-quant/tools/fq_reload.py` |

(`/home/…/runs/` = `/home/mbelleau/protensors-work/vllm-voipmonitor/research/fungible-quant/runs/`.)

### Data

| What | Absolute path | Size |
|---|---|---|
| Local segments (manifest dir, resolves first) | `/home/mbelleau/glm52-segments` | 268G |
| Segment + fragment cache (shared by both serves) | `/home/mbelleau/cache/fq-demo1` | 303G (299G segments, 3.4G fragments) |
| Assembled K3 checkpoint (the `model=` path) | `/home/mbelleau/glm52-k3-assembled` | 295G |
| M0 seed tree (symlink source for the mixed tree) | `/home/mbelleau/fq-segments` | 260G |
| Mixed K3/K5 inputs (176 symlinks into the above) | `/home/mbelleau/fq-segments-mixed-k3k5` | small |
| Release staging | `/home/mbelleau/fq-release-stage` | 40K + K2 files |
| Calibration capture (1,050,468 tokens) | `/home/mbelleau/glm52-capture` | 13G |
| Encode work dirs | `/home/mbelleau/glm52-work-k2`, `-k4`, `-k5` | ~50M total |
| Container images | `/home/mbelleau/images/{gg-v20-r33, gilded-gnosis-v20-r12-field-review, glm52-exl3-vast-latest}` | 12G each |
| Extracted rootfs (the live runtime) | `/home/mbelleau/rootfs/gg-v20-r33` | 22G |
| Python venv for tooling | `/home/mbelleau/venvs/fq` | — |
| **Secrets** | `/home/mbelleau/.fq_env` | chmod 600 |

Published: HF **`malaiwah/GLM-5.2-EXL3-FQ-segments`** — K2 **75**, K3 **92**,
K4 **99**, K5 **24** layer segments; manifest `k_variants=[2,3,4,5]`.

**`/home/mbelleau/.fq_env` holds HF_TOKEN, HUGGING_FACE_HUB_TOKEN, GH_TOKEN,
GITHUB_TOKEN. Never echo it, never commit it.** Load with:
```bash
set -a; . /home/mbelleau/.fq_env; set +a
```

### Logs and results

- Serve logs: `/home/…/runs/m5-serve/results/demo1/` and `.../results/demo2/`
- Battle-test reports: `/home/…/runs/m5-serve/results/bt/*.md`
- PR materials: `/home/…/runs/pr/`
- Design docs: `/home/…/runs/m5-serve/*.md`

---

## 4. THE trap: the runtime is not the source tree

**The serve imports `exl3_fungible` from the ROOTFS, not from
`/home/mbelleau/src/gg-vllm`.** Editing and committing has zero effect on the
next boot — silently. This has cost at least three full boots.

`serve-demo1.sh` runs `deploy-fq.sh` itself. If you launch any other way, deploy
first. To verify:

```bash
diff -q /home/mbelleau/src/gg-vllm/vllm/model_executor/layers/quantization/exl3_fungible/loop.py \
        /home/mbelleau/rootfs/gg-v20-r33/opt/venv/lib/python3.12/site-packages/vllm/model_executor/layers/quantization/exl3_fungible/loop.py
```

**Running the tests** needs a neutral cwd (the source tree shadows the installed
`vllm`) and `--noconftest` (the repo conftest imports `tblib`, absent here):

```bash
cd /tmp && \
/home/mbelleau/protensors-work/vllm-voipmonitor/research/fungible-quant/runs/gg-env/gg-run.sh \
  python -m pytest /home/mbelleau/src/gg-vllm/tests/exl3_fungible/ \
  -q -p no:cacheprovider --noconftest -k "cpu or preflight"
```
Expect **691 passed, 10 deselected**. Note tests run against the **deployed**
rootfs copy, so *deploy before testing* or you will test stale code.

---

## 5. How we use the two sets of cards — the concurrency discipline

We run **two independent vLLM instances at once**, splitting the 8 GPUs 4+4.
This is deliberate: one instance holds a stable configuration while the other
takes the risky change, so a failed experiment never costs all the GPU capacity,
and two independent confirmations are available for any claim.

| | **demo1** | **demo2** |
|---|---|---|
| GPUs | 0,1,2,3 | 4,5,6,7 |
| Port | 8100 | 8200 |
| `FQ_TAG` | `demo1` | `demo2` |
| Typical role | the *slow, realistic* one: CUDA graphs + JIT (`FQ_FAST=0`) | the *fast-iteration* one: eager, no compile (`FQ_FAST=1`) |
| tmux window | `fq:demo1` | `fq:serve2` |

Launch form (both use the same script; the tag and device list do the splitting):

```bash
M5=/home/mbelleau/protensors-work/vllm-voipmonitor/research/fungible-quant/runs/m5-serve
cd $M5 && FQ_TAG=demo2 FQ_DEVICES_ENV=4,5,6,7 FQ_FAST=1 FQ_GPUMEM=0.95 \
  FQ_MAXLEN=8192 VLLM_FQ_LIVE_APPLY=1 \
  bash serve-demo1.sh policy-demo1-fitted.json 8200 2>&1 | tee $M5/results/demo2/serve-<name>.log
```
Positional args are `<policy.json> [port] [extra vllm args...]`
(`serve-demo1.sh:15-16`).

### What is shared and what is isolated — this matters

- **SHARED (read-mostly, safe):** `VLLM_FQ_CACHE=/home/mbelleau/cache/fq-demo1`.
  Both instances read segments and fragments from the same content-addressed
  cache. This is intentional — it makes the second boot warm.
- **ISOLATED (per `FQ_TAG`):** everything *written* —
  `VLLM_FQ_ARTIFACT_DIR`, `VLLM_FQ_CACHE_ROOT` (the policy store),
  `VLLM_FQ_DUMP_STATS`, `VLLM_FQ_ENCODE_QUEUE`, `PROMETHEUS_MULTIPROC_DIR`,
  `CUDA_CACHE_PATH=/home/mbelleau/cache/jit-$FQ_TAG/cuda`.

So the two instances **cannot** corrupt each other's policy state. This was
checked explicitly while chasing a divergence bug, and it eliminated
"the two serves are fighting" as a hypothesis.

### Rules for two-instance work

1. **NEVER start a second boot while the other instance is still LOADING.**
   Serve concurrently, yes. **Load serially.** This is not a style preference —
   see the hard evidence below.
2. **Change one variable at a time, on one instance.** If demo2 is verifying a
   fix, do not simultaneously change demo1's policy *and* its execution mode.
3. **Never launch onto cards you have not just verified free.** Orphaned TP
   workers survive process-group kills and hold 15–22 GiB each.
4. **Keep both busy** once both are up. Idle cards are wasted; there is always a
   pending battle test.

### The concurrent-boot OOM — measured, 2026-08-11

Two worker deaths in one hour, both silent (process gone, `exit code: None`, **no
Python traceback** — so not an exception; the loop catches those):

- 19:50Z — demo2 lost `VllmWorker-0` while **demo1 was mid-load**.
- 19:57Z — demo1's engine failed init while **demo2 had just restarted loading**.

Cause, from the cgroup itself:

```bash
cat /sys/fs/cgroup/memory.max      # 1374389534720  = 1280 GiB
cat /sys/fs/cgroup/memory.peak     # 1374389534720  = 1280 GiB  (pegged)
cat /sys/fs/cgroup/memory.events   # oom 16 / oom_kill 2  <-- exactly the two deaths
```

**A single loading instance sits at ~507 GiB of cgroup-accounted memory** (mostly
page cache from streaming segments, which counts against the cgroup). Two at once
reach the 1280 GiB ceiling and the kernel kills a worker. `free -g` looks
*innocent* — it showed ~1 TB "available" — because the pressure is cgroup-scoped,
not host-scoped. **Always check `/sys/fs/cgroup/memory.events`, not `free`.**

So: stagger boots. Wait for `Application startup complete` on one instance before
launching the other. If you must overlap, lower `VLLM_FQ_PREFETCH_DEPTH` and/or
drop caches between loads — untested.

### Env vars the launcher honours

`FQ_TAG FQ_DEVICES_ENV FQ_FAST FQ_GPUMEM FQ_MAXLEN` and
`VLLM_FQ_ADMIN_API VLLM_FQ_APPLY_MODE VLLM_FQ_ARTIFACT_DIR VLLM_FQ_BUDGET_MIN_KV
VLLM_FQ_BUDGET_UTIL VLLM_FQ_CACHE VLLM_FQ_CACHE_ROOT VLLM_FQ_DENSE_SOURCE
VLLM_FQ_DUMP_STATS VLLM_FQ_DWELL_STEPS VLLM_FQ_ENABLE VLLM_FQ_ENCODE_QUEUE
VLLM_FQ_GATE_MASS VLLM_FQ_HEATMAP VLLM_FQ_INTERVAL_STEPS VLLM_FQ_JACCARD_FLOOR
VLLM_FQ_KEEP_LAYERS VLLM_FQ_K_FALLBACK VLLM_FQ_MANIFEST_DIR
VLLM_FQ_ON_UNAVAILABLE VLLM_FQ_POLICY VLLM_FQ_PREFETCH_DEPTH VLLM_FQ_SOURCES
VLLM_FQ_SOURCES_MODE VLLM_FQ_TABLE_EVERY_INTERVALS VLLM_FQ_TRUST_PREDICATES
VLLM_FQ_VERIFY` plus `VLLM_FQ_LIVE_APPLY` (default **off** — live swapping is
opt-in) and `VLLM_FQ_BUDGET_ENFORCE=0` to override a preflight refusal.

---

## 6. Monitoring — what we watch and how

Three layers, deliberately different in kind:

### (a) The 10-minute health sweep — periodic, whole-system

```bash
bash /home/mbelleau/protensors-work/vllm-voipmonitor/research/fungible-quant/runs/health/sweep.sh
```

Prints one line per job (campaign supervisor, K2/K4/K5 encode, 0c capture,
publish, m5-serve), plus per-K encoded-layer counts, per-instance load progress
and install counts, all 8 GPUs, and free disk. It tracks log growth between runs
and prints `+NNNN B` or `idle`.

Currently also wired as a **recurring cron job (`a5e1d0be`, `*/10 * * * *`)**
that re-runs the full health-check prompt. Session-only — it dies with the
Claude session and auto-expires after 7 days. **Re-create it after a
preemption.**

### (b) Log monitors — event-driven, per serve

One persistent `Monitor` per live instance, tailing its log through a grep that
covers **both success and failure**:

```
INSTALLED|invalid swap|NameError|interval at step .* failed|
differ from this interval|Application startup complete|BudgetExceeded|
No available memory|died unexpectedly|tier changes across
```

The rule that earned this filter: **a monitor that only greps the happy path is
silent through a crashloop, and silence looks exactly like "still running."**
Always include the failure signatures.

### (c) Endpoint probes — for assertions, not for reassurance

`GET /fq/state`, `/fq/layer/{n}`, `/fq/heatmap`, `POST /fq/heatmap/reset`,
`POST /fq/retier`, `POST /fq/tune`. Use `curl --compressed` — the heatmap body
is ~2.4× larger without gzip, and the endpoint says so in its own `warnings`.

**Caveat that has already burned us:** `/fq/layer` and `/fq/state` report the
**loop's** view (`state.tier_of`), not the device's. See §7.2.

---

## 7. Sharp edges — read twice

### 7.1 About ten distinct failures have looked exactly like success

Staged batches silently discarded; a cache pruner that was a permanent no-op
because `bfs` rejects relative `-newermt` and `|| true` ate the error; a sweep
watching a log from a dead run; an experiment that re-ran its own control
because an unconditional `export CUDA_MODULE_LOADING=EAGER` overrode an explicit
`LAZY`.

**Rule: assert on artifacts, never on liveness.** Byte counters, install counts,
tier maps read back, done-JSON counts, HF segment counts, GiB delivered.
A healthy-looking log is not evidence.

### 7.2 `/fq/layer` and `/fq/state` show the LOOP's view, not the device

Both resolve through `_loop_state(worker) → state.tier_of` — the policy loop's
array, **not** the engine's `MixedLayerState.tier1_globals` that the kernel
actually indexes. When they diverge the endpoint actively misleads: it once
confirmed `e51 = K4` while `engine.stage()` was rejecting that very swap as
"e_out must be resident K4".

Consequence: BT-6's "14 experts verified moved" was **not** independent evidence.
Task **#70** adds a device-truth surface. Until then, never cite `/fq/layer` as
proof that weights moved.

### 7.3 Reap GPUs BY DEVICE, never by process pattern

Workers exec through the rootfs `ld-linux` shim, so their argv contains neither
"vllm" nor "VLLM::" — **`pkill -f vllm` matches nothing** and leaves orphans
holding 15–22 GiB. Conversely `pgrep -f 8200` matches **too much**:
`--hf-overrides` carries 150+ sha256 digests and one contained `...8200...`,
which killed a healthy serve on a different port.

Always:
```bash
bash /home/…/runs/m5-serve/reap-devices.sh 4,5,6,7
```
It kills by `nvidia-smi --query-compute-apps` and waits for the memory to
actually come back (driver release lags the kill by 10–20 s; a boot started too
early reaps itself).

### 7.4 tmux: never send C-c and the next command in one burst

If the foreground process is still running, the typed command is swallowed as
**its stdin** and never executes. This left demo2 down ~10 minutes with an
orphan API server while the relaunch silently never happened. Send C-c, wait,
confirm a prompt, send the command — then **verify the log file exists**.

### 7.5 Layer 78 asymmetry

Loader domain = **76 layers**; decision domain = **75**. GLM-5.2's MTP layer 78
has a router the collector never binds. This caused four separate failures
(documents dropping layer 78; phantom `n_k4=0` rows averaged into metrics).
See `/home/…/runs/m5-serve/layer-78-constraint.md`.

### 7.6 Exactly two tiers per layer — and why more tiers would NOT help

`exl3.py:1591` — `if len(tiers) != 2`. Our own extra restriction is a single
guard at `swap.py:560` refusing anything but `(K3, K4)` — but `swap.py:706`
already takes `tier_bits` as a constructor parameter and uses it throughout, so
the engine is **already generic over a pair**.

**Measured, not assumed** (`/home/…/results/bt/N-TIER-FEASIBILITY.md`): a
3-tier subclass of `W4A16MixedTrellisKernel` was compiled through stock
`b12x_compile` on this hardware at production geometry. A third tier costs
**+8 registers/thread and nothing else** (K3/K4 = 143 regs, K2/K3/K4 = 151;
identical 51328 B shared, 0 spill, 1 block/SM). Tier count adds **zero** kernel
launches (one cooperative grid), and the descriptor `(tier << 8) | local` has
**23 free tier bits** in its int32. So the limit is a code constant, not
hardware — the guard's own text says "*currently* requires exactly two" and
`_moe_body` is already documented as the "hybrid **multi-tier** entry".

**But lifting it would not buy the K2→K3→K4 ladder.** A mixed layer's total bit
budget is *conserved* regardless of tier count: `run_mixed_trellis` pins the sum
of tier capacities to the global expert count (`global_to_combined` doubles as
the router's `expert_map`) and each slab is exact-size-validated against its
population. **The only legal runtime operation is a cardinality-preserving
permutation.** Boot-at-K2-then-grow is a memory-*growth* problem, not a
tier-count problem. Do not spend effort on N tiers expecting a ladder.

**What does work today: per-layer tier PAIRS.** `tier_bits` is per-layer on
every axis — config (`exl3.py:559-576`), prepare, runtime cache key
(`exl3.py:1851`), and the b12x compile cache. And `policy.py:474-476` records
that **`glm52-mixed-k3k5` already shipped with real K5 experts** — a `(3,5)`
pair in production. A model-wide K2..K5 ladder therefore exists *today*,
distributed across layers. The remaining work is ours and small: lift
`tier_bits` from engine-wide to per-layer in `swap.py` (`:561-563`, `:707`,
`:736-742`) and drop the hardcode at `admin.py:1902`. `ExpertStage` is already
bits-parameterized.

### 7.7 The engine never reallocates, on purpose

`fq_reload.py`'s docstring states the invariant: *"Same per-layer tier
cardinality is REQUIRED (fixed tier_signature keeps the compiled launches and
CUDA graphs valid); membership is the only degree of freedom."*
`capture_model` (`/home/mbelleau/src/gg-vllm/vllm/v1/worker/gpu_model_runner.py:6859`)
captures whole-model over batch sizes, so graph re-capture is **not** per-layer.

### 7.8 Two engines was a real hazard — understand why it happened

`MixedLayerState.from_exl3_mixed_trellis` **copies** the tier orderings out of
the module. A committed `apply()` updated only that copy, so the admin API's
engine and the loop's engine each tracked a private host view over one shared
set of device maps, and any engine built later started from *boot* orderings.
Fixed in `40b6f5e8a`: one engine per worker via `state.swap_engine`, plus
`MixedLayerState.publish()` writing committed orderings back to the module at
the same visibility point as the maps. M3's `fq_reload` had this write-back
right all along (its step 4); M4 dropped it.

### 7.9 A test that reimplements the code under test is worthless

`test_apply_fn_may_report_what_it_actually_installed` recomputed the adoption
arithmetic inline and asserted on its own copy. It stayed green while the real
branch raised `NameError` on its first-ever execution — that branch only runs
*after* a successful install, so it had never run in any test. Now rewritten to
drive the loop through `drive_hot_interval()`. **Any branch that only executes
on a success path needs a test that actually reaches it.**

### 7.10 Retract loudly and keep the wrong reasoning visible

Three claims have been retracted so far:
1. *"~3.92 GiB stranded by the allocator"* — measured **0.00 GiB** freed, 16
   times; the comparison confounded different `max_model_len`.
2. *"Routing churns ~39% per interval"* — artefact: `_jaccard` scored
   empty-vs-empty as 0.0 and 27 of 75 layers have `n_k4=0` and cannot churn.
   Real ≈ **0.879**.
3. *"~2,300 promotion ceiling"* — used nvidia-smi total instead of
   `torch.cuda.mem_get_info`, plus an invented 4 GiB KV floor. Real ≈ **4,200**.

`/home/…/results/bt/ROUTING-FLATNESS.md` carries a retraction banner at the top
with the original text preserved below it. **Do this** — the wrong reasoning is
the useful part.

---

## 8. Where we are right now

### Live

- **demo1** — GPUs 0–3, port 8100, **CUDA graphs + JIT**, policy
  `/home/…/runs/m5-serve/policy-demo1-graphs32.json`, live apply on. Loading.
  First graphs boot to clear the memory preflight.
- **demo2** — GPUs 4–7, port 8200, eager, policy `policy-demo1-fitted.json`,
  live apply on. Restarting after a worker death. Log
  `/home/…/results/demo2/serve-bt6c3.log`.
- **Encode campaign** — complete, not stalled: K2 75/75, K4 75/75, K5 24/75
  (deprioritised, SM120 shared-memory limit, see
  `/home/…/runs/m5-serve/k5-shared-memory-limit.md`). Publish done.
- **Disk** — 1.2T free of 4.0T. Michel enlarged the volume during the
  preemption; the old 180G floor pressure is gone.
- **tmux** — session `fq`; windows `claude campaign fragprune demo1 pause
  serve13 bt6 heatmap serve2`. **Never kill session `fq` or window 0.**

### Proven, with artifact evidence

- **BT-1 cold boot from segments**: 79.08 GiB resident vs 79.06 projected,
  KV +3.67 GiB, 1616 s to serve, coherent output.
- **BT-2 warm restart**: **0.0 GiB fetched** vs 295.8 cold, identical posture
  digest, 5.4×.
- **M4 swap engine**: forced re-tier moves real weights,
  `delta_bytes_per_rank: 0`, cardinality held, rollback exact.
- **Language routing is real**: EN-vs-ZH top-K Jaccard **0.321** vs EN-vs-EN
  **0.879** (chance 0.053), with divergence rising with depth (Spearman
  −0.4168).
- **Selection signal**: frequency ranks experts **0.96** as well as
  `mass × error`; error alone **0.00**. Hot experts quantize *best*
  (Spearman −0.7786, 75/75 layers) because LDLQ fits each expert on its own
  traffic.
- **Tensor-level FQ answered — NO** (`/home/…/results/bt/TENSOR-LEVEL-FEASIBILITY.md`):
  mass is per-expert by construction, so inside an expert `mass × Δloss`
  collapses to `Δloss`; there is no online signal below the expert. TP-rank
  main effect 0.0000. gate-vs-up is physically impossible (two N-halves of one
  FC1 GEMM, trellis bits a CuTe `const_expr`).

### NOT proven — do not claim

- **Sustained installs under load** (BT-6c). One install verified; stability
  across intervals never shown.
- **Quality delta** (BT-7). Nobody has measured whether the converged posture
  actually improves output. This is what reviewers will attack first.

### Open work — the complete register

The live task list is **session-only and dies with the Claude session**, so it is
reproduced here in full. 39 open items. IDs match the in-session task list while
it exists. Grouped by kind; roughly priority-ordered within each group.

#### A. In flight right now

- **#64 — BT-6c: sustained installs under load.** *In progress.* Pass bar:
  installs **repeat** across intervals with **zero** `invalid swap`. One install
  is not enough — BT-6 already showed one. Two bugs were fixed to get here
  (`40b6f5e8a` one-engine + publish, `505ffaa7b` the `NameError`); neither is
  verified live yet. Run on demo2, eager, `VLLM_FQ_LIVE_APPLY=1`, with
  `swap_evidence.py both --phase math:1800 --concurrency 8`.

- **#73 — OOM-1: concurrent boots hit the 1280 GiB cgroup ceiling.** Rule now in
  force: **serve concurrently, load serially**. But a *single* load reached
  671 GiB at 47/76 layers (~14.3 GiB/layer → ~1085 GiB extrapolated), and
  `memory.events` shows `oom 16` for only 2 kills, which means much of it is
  **not reclaimable** — likely mmap references to segment files held open for
  the whole boot. Investigate in order: (1) sample
  `/sys/fs/cgroup/memory.stat` for `file_mapped` vs `inactive_file` across a
  boot; (2) if confirmed, munmap/`MADV_DONTNEED`/O_DIRECT in the segment read
  path — a 79 GiB resident model should not need ~1 TB of page cache;
  (3) test `VLLM_FQ_PREFETCH_DEPTH` first, it is free; (4) make
  `serve-demo1.sh` refuse to boot while a peer is loading, as it already does
  for busy GPUs; (5) add `memory.events`/`memory.current` to
  `runs/health/sweep.sh`.

- **#65 — DEMO2-GRAPHS: a CUDA-graphs instance has never served.** Four attempts;
  none crashed — the **memory preflight refused them all**. Graphs cost ~1.9 GiB
  of device budget (88.32 vs 90.22 GiB), squeezing projected KV to 0.95 GiB
  against the 2.00 GiB floor. `policy-demo1-graphs32.json` (K4 capped at
  32/layer, 2,658 → 1,530 experts, +1.24 GiB → 2.19 GiB projected KV) clears it
  and was loading past layer 33 when the OOM killed it. Retry after #64, on an
  idle box.

#### B. Correctness and observability debt

- **#70 — OBS-2: `/fq/layer` and `/fq/state` report the LOOP's view.** Both
  resolve through `_loop_state(worker) → state.tier_of`, not the engine's
  `MixedLayerState.tier1_globals` that the kernel indexes. When they diverge the
  endpoint actively misleads. Add a device-truth RPC returning `tier1_globals`
  per layer, exposed as `/fq/layer/{n}?source=device` or a parallel field, so a
  mismatch is visible instead of silently resolved in the loop's favour.
  **Every past verification that used `/fq/layer` as an "independent surface"
  needs re-reading in this light**, including the M4 by-hand swap proof and the
  BT-6 install proof.

- **#66 — TUNE-1: exercise `POST /fq/tune` against a live serve.** The endpoint
  and `worker_tune()` exist and are unit-tested; they have never been driven
  against a running model. Confirm a tunable actually changes loop behaviour
  mid-run (e.g. `jaccard_floor`, `dwell_steps`) and that all four ranks agree
  afterwards.

- **#43 — M5-O: report the borrowed-buffer loader hazard upstream.** EXL3 quant
  methods retain every loaded tensor until `process_weights_after_loading`, and
  their two "copies" are identity ops (`t.contiguous() is t`;
  `t.to(t.device) is t`). So **any** borrowed-buffer loader corrupts an EXL3
  checkpoint, with or without FQ: `fastsafetensors` with `world_size>1` is
  incompatible — that is every TP serve — and `instanttensor` with
  `INSTANTTENSOR_COPY=0` likewise. Worst in mixed-bitrate, where
  `create_weights` drops trellis from the copy set. This is an upstream GG/EXL3
  bug and deserves its own report, separate from our PR.

#### C. Memory and growth

- **#58 — GROW-1: one-slot reserve per layer, BOTH directions, one op in
  flight.** Michel's choice: reserve only the buffer needed to grow **one**
  expert to the maximum K the constraints allow, limit to one growth at a time,
  and work out what (if anything) must be reserved for shrinking too.
- **#59 — GROW-2: count the reserve in the memory preflight.** The preflight
  models steady state; a growth reserve and a re-prepare transient are both
  invisible to it today.
- **#60 — GROW-3: apply unpaired promotions, refuse beyond the reserve.** Today
  every swap is paired (K3↔K4) so cardinality is fixed. Growth means an
  *unpaired* promotion, which is the only way a layer's bit budget can rise.
  Must fail closed once the reserve is spent.

- **#72 — K2-BOOT: per-layer tier pairs so the model boots small and upgrades
  live.** *Michel's idea; strongest version of the progressive story.* All-K2
  experts are 43.48 GiB/rank vs all-K3 64.85 — **21.4 GiB less to move before
  first token (33%)**, and that same 21 GiB is what squeezes KV for #65. K2 is
  fully encoded (75/75 on HF) and the loop already prices it. **Shape it as
  per-layer PAIRS, not N tiers** — see §7.6: a layer's bit budget is conserved,
  so a within-layer ladder is a growth problem (#58–60), not a tier-count one.
  Work: (1) `swap.py:561-563` replace the `(K3,K4)` equality guard with a
  validated pair (both in {2,3,4,5}, ascending), fail-closed otherwise;
  (2) `swap.py:707,736-742` lift `tier_bits` from engine-wide to per-layer,
  sizing staging from the widest pair present (`ExpertStage` is already
  bits-parameterized); (3) `admin.py:1902` drop the hardcode; (4) boot demo1
  with shallow layers on `(2,3)` and deep on `(3,4)`, measure TTFS against
  BT-1's 1616 s, report the KV headroom freed. Tests: pair validation, staging
  sizing, a `(2,3)` apply round-trip in `test_swap_cpu.py`.

#### D. The headline demos

- **#35 — M5-H / Scenario 1: FQ rediscovers the coder quant's expert choices.**
  Analyse the K-distribution of the 3.42bpw "coder" quant (its high-K experts
  were chosen with a coding corpus). Use its footprint as the loader's memory
  envelope. Boot plain 3.0bpw (all K3) with cached segments + HF access,
  restricted to K3/K4. Replay the MTP78 activation corpus + GSM8K. **Hypothesis:
  FQ converges on the same experts the 3.42bpw quant picked.** Measure Jaccard
  against the reference. If it holds, downloading N separate 3.x bpw full quants
  is wasted bandwidth — which is the whole economic argument.

- **#36 — M5-I / Scenario 2: boot flat K3, grow to the fitted posture live.**
  Note the design constraint Michel asked about: a literal `tiers=((3,256),)`
  boot **cannot** grow — zero K4 slots and the engine never reallocates. The
  version that works today allocates the K4 slots but fills them with an
  information-free choice, so the loop must discover the right occupants. The
  fitted policy's per-layer max is already exactly 56, so `((3,200),(4,56))`
  lines up. 13 slots/layer works today; 56 needs the growth work or a compacted
  policy.

- **#62 — R10-CMP: score our policy against R10's optimal allocation.**
  `brandonmusic/GLM-5.2-EXL3-TR3v4-3.5bpw-MTP78` ships 75
  `R10_FROZEN_DECISIONS_LAYER_*.json` — per-tensor bit assignments from a
  DP-optimal exact-budget knapsack on `mass × measured-Δloss` with held-out
  rows. A **better** Scenario 1 reference than the coder quant because it is
  optimal rather than another heuristic. Compute: (a) how much of R10's total
  gain a per-expert fixed-cardinality approximation captures; (b) per-layer
  overlap between our K4 set and R10's high-bit tensors; (c) whether the loss
  curves are monotone often enough that fixed cardinality is safe. Answers
  "what does expert granularity cost vs tensor granularity" with numbers.

#### E. Battle tests remaining

- **#47 — BT-3: offline restart.** `HF_HUB_OFFLINE=1` against the primed cache.
  Proves the warm path needs no network at all.
- **#48 — BT-4: deliberately degraded boot.** Request a K the box cannot supply
  and confirm the K-fallback ladder degrades gracefully rather than failing.
- **#49 — BT-5: convergence repays the deficits.** From BT-4's degraded state,
  let `ConvergenceWorker` fetch what exists and encode on the fly what does not,
  live-swapping via `fq_converge_layers`. Measure time-to-converge and per-layer
  progress; assert the final composition matches the desired posture.
  **Verify the swap actually installed — this RPC has previously reported
  success it never performed.**
- **#69 — BT-5 prerequisite: nothing drains the encode queue.** `encode-queue.jsonl`
  records on-the-fly encode requests but no consumer processes them. BT-5 cannot
  pass until something does.
- **#51 — BT-7: quality delta.** GSM8K on the degraded arm vs the converged arm,
  **same 250-item subsample and seed** for a paired comparison (250 items =
  ±2% stderr, so unpaired 1–2 point deltas are noise). Baseline to beat:
  **89.2% flexible-extract on flat K3.** *This is the only test that asks
  whether re-tiering improves output rather than just moving bits, and it is
  the claim reviewers will attack first.*
- **#52 — BT-8: restart after convergence.** Does the hydrated posture persist
  across a restart? (Related to #32, M5-E.)
- **#53 — BT-9: kill -9 mid-swap.** Crash safety of the live path. The engine
  stages fail-atomic and the slabs are a cache, so the claim is that a torn
  layer is recoverable by re-applying or rebooting — untested.
- **#32 — M5-E: prove hydrated-expert state survives restart.**

#### F. Measurement and analysis

- **#61 — FLAT-1: re-measure the collector window effect with the fixed Jaccard
  guard.** The original "15× the sample buys +0.014" used the corrupted metric
  (§7.10 retraction 2). The delta may survive; the levels do not.
- **#39 — Tensor-level FQ: ANSWERED, one offline experiment left.** Online
  per-slice re-tiering is **not** worth it (see
  `results/bt/TENSOR-LEVEL-FEASIBILITY.md`). What remains is cheap and offline:
  assemble two equal-byte checkpoints from existing segments — **A** = 26
  experts × 3 projections, **B** = proxy-optimal FC1/FC2 split (66 `down_proj` +
  6 gate/up pairs) — and run one KLD eval each. `slice_nmse` and
  `slice_proxy_err` disagree in **sign** about `down_proj` and predict opposite
  winners (**B/A = 1.169** under proxy, **0.795** under nmse), so it cannot come
  back ambiguous. If B wins, the action is a down-heavy offline **tier shape** —
  a policy change, not a granularity change.
- **#68 — ARCHIVE-1: fold heatmap samples into a Parquet dataset.** Nine+ JSON
  blobs at ~78 KB that must be decoded one at a time. In Parquet LONG form
  (`run, step, layer, expert, count, tier`) the recurring analyses — EN/ZH
  contrast, top-K stability, R10-CMP — become `SELECT` statements at ~31 KB per
  sample. Keep the live endpoint as base64-in-JSON; that decision is measured
  and correct (`heatmap-format-analysis.md`).
- **#41 — M5-M: live expert-activation heatmap endpoint + operator page.**
  Endpoint exists and is instrumented; the operator-facing page does not.
- **#12 — Phase 0d/0e: specialization claim + rollback threshold.** Blocked by
  #11.

#### G. Publication and upstream

- **#33 — M5-F: open the evidence-rich PR.** See §11 for the full gate. The base
  must be **hand-retargeted to `dev/gilded-gnosis`**.
- **#34 — M5-G: answer the EPLB duplicate-work challenge in the PR.** Reviewers
  will ask why this is not expert-parallel load balancing. Have the answer
  written before submitting.
- **#42 — M5-N: document the TP4-only artifact constraint** on the HF card and
  in the PR. Segments are rank-sliced; they are not portable to another TP
  degree.
- **#23 — Community publication:** reassembly-verified segments + a public tools
  repo + the card.
- **#24 — Refresh the public repo** with real GLM-5.2 data (charts, claims) once
  available.
- **#27 — Prior-art-driven evolution:** two-producer interop + supply-chain
  economics evaluation.
- **#17 — M5: hardening + release as a GG image tag.**
- **#18 — HUMAN (Michel): post the RFC #49702 comment before 2026-08-18.**
  *Only Michel can do this one.*

#### H. Campaign and infrastructure

- **#11 — Phase 0c per-expert dKL sensitivity campaign.** *Paused deliberately*
  2026-08-11 15:00Z to free disk and GPUs, not stalled. K2 75/75 and K4 75/75
  complete and published; K5 24/75 deprioritised (K5 cannot be served on SM120
  anyway — 109,568 B shared memory vs the 101,376 B opt-in limit). Paused
  cleanly via `pause-when-k4-done.sh`: waited for the 75th done-JSON rather than
  killing mid-encode, drained the encoder, stopped the supervisor, ran a final
  publish. **Resume by restarting `campaign-supervisor.sh` — done-JSONs make it
  idempotent.**
- **#20 — Multi-K campaign**: encode GLM-5.2 at K2/K3/K4/K5 and publish
  segments. Largely done except K5.
- **#38 — M5-K: fetch K5 layers 3–10 (49 GB) pruned locally, or scope the claims
  to 12 layers.**
- **#13 — M1: stats collector + persistence.** *In progress.*
  `exl3_fungible/stats.py`: `BaseRouter.set_capture_fn` binding (graph-safe
  `scatter_add`, capture-fn chaining per `gg-integration-surface.md`),
  window/decay, stats dump. **Gate: T1 green on 1 GPU and TP4; overhead <0.5%
  decode tok/s at cc8.**
- **#14 — M2: policy engine in dryrun.** Blocked by #13.

## 9. Preemption recovery checklist

The box can vanish. Only `/home` survives. On a new instance:

1. `uptime`; confirm `tmux list-sessions` fails (session gone).
2. Verify persistence: `/home/mbelleau/rootfs/gg-v20-r33`,
   `/home/mbelleau/src/gg-vllm`, `/home/mbelleau/glm52-segments`,
   `/home/mbelleau/cache/fq-demo1`,
   `/home/mbelleau/protensors-work/vllm-voipmonitor`, `/home/mbelleau/venvs/fq`,
   and `/home/mbelleau/.fq_env` (must still be `600`).
3. Rebuild tmux:
   ```bash
   tmux new-session -d -s fq -n claude
   for w in campaign fragprune demo1 pause serve13 bt6 heatmap serve2; do
     tmux new-window -d -t fq -n "$w"; done
   ```
4. Check for **partial captures**: `find /home/mbelleau/glm52-capture -name "*.partial"`.
5. Verify the rootfs deploy matches source (§4); re-deploy if not.
6. `git status` both repos; commit and push stranded run artifacts.
7. Re-arm the log monitors and the 10-minute sweep cron.
8. Verify all 8 GPUs free (`nvidia-smi --query-compute-apps`), then relaunch.

---

## 10. Working discipline (Michel's rules)

- **`uv` for all Python.** Never bare `pip`.
- **A tmux window per long job**, on-disk resumable state, **commit + push after
  every completed step**.
- **Artifacts >100 MB go to HF**, with the sha256 recorded in the committed
  report.
- **Work against fresh GG/b12x HEADs.**
- **Tests land with the code.**
- **Keep the GPUs busy.**
- **Anything touching correctness (KLD), runtime memory, or performance (both
  prefill and decode) must be made performant (CUDA graphs) AND instrumented.**
- Proxy models: small real-weight GLM-5.2-arch "Fruit" models exist for cheap
  iteration; **GLM-5.2 itself is what proves a milestone.**
- Verify artifacts, not liveness. Flag disk below 180G (moot at 1.2T free).
- Report failures plainly, including your own.

---

## 11. The PR

Materials in `/home/…/runs/pr/`:

| File | Size |
|---|---|
| `PR-BODY.md` | 584 lines / 32 KB |
| `SUBMISSION-CHECKLIST.md` | 365 lines |
| `SEPARATE-REPORTS.md` | 357 lines |
| `COMMITS.md` | 126 lines |
| `FILED-ISSUES.md` | 72 lines |

Evidence: `/home/…/runs/m5-serve/results/bt/` — `BT-1-AND-2.md`,
`BT-6-INSTALLED.md`, `ROUTING-FLATNESS.md` (*with its retraction banner*),
`SELECTION-SIGNAL.md`, `R10-ALLOCATOR.md`, `TENSOR-LEVEL-FEASIBILITY.md`.
Design: `/home/…/runs/m5-serve/` — `peer-review.md` (981),
`admin-api-spec.md` (1424), `topology-neutrality.md` (543),
`loader-compatibility.md` (357), `HF-CARD.md` (434), `BATTLE-TEST-PLAN.md` (217),
`growth-design.md`, `layer-78-constraint.md`, `k5-shared-memory-limit.md`.

**Before submitting — required by
`/home/mbelleau/protensors-work/vllm-voipmonitor/AGENTS.md`; breaching it can get
you banned from vllm-project:**

- Run the duplicate-work checks (`gh issue view`, `gh pr list --search`).
- **Retarget the PR base by hand to `dev/gilded-gnosis`.**
- **Pure code-agent PRs are not allowed.** A human must understand and defend
  every line. The body must state that AI assistance was used, list the test
  commands run and their results, and explain why it is not a duplicate.
- Answer the **EPLB duplicate-work challenge** (#34): reviewers will ask why this
  is not expert-parallel load balancing.
- **Scope the claim honestly.** R10 (Brandon's allocator) already solves *offline*
  allocation optimally at tensor granularity. Our contribution narrows to
  **delivery, restart, and live re-tiering**. Claiming more will not survive
  review — see `/home/…/results/bt/R10-ALLOCATOR.md`.

---

## 12. First 15 minutes on this project

```bash
# 1. Whole-system state
bash /home/mbelleau/protensors-work/vllm-voipmonitor/research/fungible-quant/runs/health/sweep.sh

# 2. What happened recently
cd /home/mbelleau/protensors-work/vllm-voipmonitor && git log --oneline -15
cd /home/mbelleau/src/gg-vllm && git log --oneline -10

# 3. The plan and the last snapshot
cat /home/mbelleau/protensors-work/vllm-voipmonitor/research/fungible-quant/runs/m5-serve/SESSION-STATE.md
cat /home/mbelleau/protensors-work/vllm-voipmonitor/research/fungible-quant/runs/m5-serve/BATTLE-TEST-PLAN.md
```

Then read, in this order:
1. `/home/…/results/bt/ROUTING-FLATNESS.md` — for the retraction habit.
2. `/home/…/results/bt/SELECTION-SIGNAL.md` — why the policy works at all.
3. `/home/…/runs/m5-serve/layer-78-constraint.md` — the 76-vs-75 trap.
4. `/home/…/runs/m5-serve/growth-design.md` — why growth is hard.
5. `/home/…/runs/m5-serve/peer-review.md` — the objections already raised.

(`/home/…/` = `/home/mbelleau/protensors-work/vllm-voipmonitor/research/fungible-quant/`.)

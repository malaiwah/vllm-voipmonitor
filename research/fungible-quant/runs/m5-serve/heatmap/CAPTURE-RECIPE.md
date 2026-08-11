# CAPTURE-RECIPE — collecting the four real per-axis routing dumps

What `make_axis_panels.py` needs, and does not yet have: **one
`VLLM_FQ_DUMP_STATS` jsonl per corpus axis**, each holding only that axis's
traffic. This file is the exact sequence an operator runs once the GPUs are
free. Nothing here may be run while GPUs 0–3 are serving or 4–7 are encoding.

Estimated cost: **4 × (~6 min boot + ~15 min replay + ~1 min drain) ≈ 1 h 30**,
from the measured 915 s of the axis-3 replay at concurrency 16
(`results/k3-fq/replay-code-REAL.json`). Four axes × 3,057 prompts.

## State today

| axis | dump | status |
|---|---|---|
| axis1_general | — | **missing** |
| axis2_legal | — | **missing** |
| axis3_code_agentic | `results/k3-fq/stats-code-axis.jsonl` | real, 13 records, no `mass_is_real` field (predates it) |
| axis4_reasoning_termination | — | **missing** |
| (not an axis) full corpus | `results/k3-fq/stats.jsonl` | real, all four axes mixed |
| (not an axis) truncated | `results/k3-fq/stats-INVALID-truncated-corpus.jsonl` | **retracted**, do not use |

So the committed figure is a layout preview:
`axis-panels.SYNTHETIC.svg` + `axis-panels.SYNTHETIC.json`, with axis 3 real
and the other three fabricated. It is watermarked, hatched and renamed, and
`make_axis_panels.py` will not emit an unmarked figure while any panel is
fabricated. Running §2–§4 below replaces it with `axis-panels.svg`, which
carries no watermark — that absence is the signal that the figure is measured.

---

## 0. Why one boot per axis (the "reset between")

The collector has **no reset entry point today** — `FqStatsCollector` exposes
`step`, `decayed`, `summary`, `mass_is_real`, and nothing that zeroes the
accumulators from outside. So the reset is a **process boundary**: buffers are
allocated at bind time, and a fresh serve starts at zero. One boot per axis.

There is a second, provable option if a reboot per axis is too expensive: the
window is a **finite ring**. `window_len=64` slots × `window_stride=32` engine
steps = **2,048 engine steps** retained, and `decayed()` reads only those
slots. After 2,048 engine steps of axis-N traffic, every retained slot holds
axis-N and the previous axis contributes **exactly zero** — not "approximately
zero", zero, because its slots have been overwritten. If you go this route you
must (a) drive ≥ 2,048 engine steps of the new axis before the record you keep,
and (b) keep only the LAST record of each axis's window, which is what
`make_axis_panels.py` reads by default. A pause does **not** clear anything:
`step()` only advances when the engine steps, so idling preserves the previous
axis's window indefinitely.

**Use one boot per axis unless GPU time forces otherwise**, and if you take the
ring-drain route, say so in the figure's caption.

## 0.1 Why `dryrun`, not `live`

Boot the serve in **`dryrun`** mode (decisions logged, nothing applied) with the
uniform-K3 policy. If the loop re-tiers mid-capture, axis 2 is measured on a
different model than axis 1 and the panels stop being comparable — the picture
would then be showing "traffic × whatever the loop did", which is not the
question. Fixed tiering for all four axes; `tier_of` in every dump must be
uniform K3.

---

## 1. Preconditions

```bash
# nothing of ours on the GPUs, and nothing already on the port
nvidia-smi --query-compute-apps=pid,used_memory --format=csv
curl -sf http://127.0.0.1:8000/health && echo "PORT 8000 IS BUSY — STOP"

# the corpus is present and byte-identical to the reference quant's calibration
RUN=/home/mbelleau/protensors-work/vllm-voipmonitor/research/fungible-quant/runs/m5-serve
PY=/home/mbelleau/venvs/fq/bin/python
$PY $RUN/harness/load_mtp78_corpus.py --json | head -20
#   expect sha256 cf247acc7c5da9f0600c7d6ab3b7c2fcfc54ec30b794e3b6047559285fa44df4
#   expect 12,228 rows, 3,057 per axis
```

## 2. Capture, one axis at a time

```bash
RUN=/home/mbelleau/protensors-work/vllm-voipmonitor/research/fungible-quant/runs/m5-serve
OUT=$RUN/results/axis-panels        # NEW directory; do not write into k3-fq
PY=/home/mbelleau/venvs/fq/bin/python
mkdir -p $OUT

for AXIS in axis1_general axis2_legal axis3_code_agentic axis4_reasoning_termination; do
  echo "=== $AXIS ==="

  # --- 2a. fresh serve = zeroed accumulators. dryrun = fixed tiering.
  setsid env \
    FQ_MODEL=/home/mbelleau/glm52-k3-assembled \
    FQ_PORT=8000 \
    VLLM_FQ_POLICY=$RUN/policy-k3-uniform.json \
    VLLM_FQ_DUMP_STATS=$OUT/stats-$AXIS.jsonl \
    VLLM_FQ_GATE_MASS=1 \
    FQ_INTERVAL=100 \
    $RUN/serve-glm52.sh dryrun > $OUT/serve-$AXIS.log 2>&1 &
  SERVE_PID=$!
  SERVE_PGID=$(ps -o pgid= -p $SERVE_PID | tr -d ' ')

  # --- 2b. wait for readiness (boot is ~6 min: assembly + JIT)
  for i in $(seq 1 180); do
    curl -sf http://127.0.0.1:8000/health >/dev/null && break; sleep 5
  done
  curl -sf http://127.0.0.1:8000/health >/dev/null || { echo "BOOT FAILED"; break; }

  # --- 2c. replay EXACTLY one axis, verbatim corpus bytes
  $PY $RUN/replay_mtp78.py \
      --base http://127.0.0.1:8000 --axis $AXIS \
      --concurrency 16 --max-tokens 8 \
      --out $OUT/replay-$AXIS.json 2>&1 | tee $OUT/replay-$AXIS.log
  # expect: prompts=3057 ok=3057 failed=0, sha_ok=True

  # --- 2d. let the loop dump at least one interval AFTER the last request,
  #         so the final record covers a full window of this axis only
  sleep 60

  # --- 2e. stop this serve; the next axis gets a clean process
  kill -TERM -- -$SERVE_PGID
  for i in $(seq 1 60); do kill -0 -- -$SERVE_PGID 2>/dev/null || break; sleep 2; done
  sleep 10
done
```

`VLLM_FQ_GATE_MASS=1` is worth the extra `scatter_add_` here: with it the dumps
carry **real gate mass** and `mass_is_real: true`, and the figure can be
rendered with `--signal mass` instead of hit count. Without it, `mass` is
aliased to `count` and the tool says so. Either is publishable; silently
mixing them is not.

`FQ_INTERVAL=100` matches the interval used for the archived dumps (100 engine
steps per loop interval → one dumped record per 100 steps). A 15-minute replay
produced 13–51 records there; anything above ~25 records is plenty, since only
the last one is read by default.

## 3. Verify each dump BEFORE rendering

```bash
$PY - <<'EOF'
import json, pathlib
OUT = pathlib.Path("/home/mbelleau/protensors-work/vllm-voipmonitor/"
                   "research/fungible-quant/runs/m5-serve/results/axis-panels")
for f in sorted(OUT.glob("stats-axis*.jsonl")):
    recs = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
    r = recs[-1]
    sums = [sum(row) for row in r["count"]]
    tiers = {t for row in r["tier_of"] for t in row}
    print(f"{f.name:<44} records={len(recs):<4} layers={len(r['layers'])} "
          f"({r['layers'][0]}-{r['layers'][-1]}) experts={len(r['count'][0])} "
          f"mass_is_real={r.get('mass_is_real', 'ABSENT')} tiers={sorted(tiers)} "
          f"per-layer-total dev={max(abs(s - sums[0]) for s in sums) / sums[0]:.1e} "
          f"total={sum(sums):.3e}")
EOF
```

Every line must show:

| field | required value | why |
|---|---|---|
| `layers` | 75, `3-77` | layer 78 is MTP and is not instrumented; a dump with 76 rows means something changed and the reference join will be off by a layer |
| `experts` | 256 | |
| `mass_is_real` | `True` (or `False`, but **never** `ABSENT`) | the archived `results/k3-fq/*.jsonl` predate the field; a fresh capture must carry it. Never infer it by comparing `count` to `mass` |
| `tiers` | `[3]` | dryrun kept the tiering fixed; anything else means the loop applied swaps and the axes are no longer comparable |
| per-layer-total dev | `< 1e-9` | top-k routing gives every layer the same total; if this trips, per-layer share and per-run share stop being the same picture (DESIGN §5.1) |
| `records` | ≥ 10 | fewer means the replay ended before the loop dumped a full window |

Also check `replay-$AXIS.json` reports `prompts: 3057, ok: 3057, failed: 0` and
that the four `corpus_sha256` values are identical. A failed request is a
prompt whose routing never happened, and 3,057 is the axis's exact row count.

## 4. Render the flagship figure

```bash
cd $RUN/heatmap
$PY make_axis_panels.py \
  --axis axis1_general=$OUT/stats-axis1_general.jsonl \
  --axis axis2_legal=$OUT/stats-axis2_legal.jsonl \
  --axis axis3_code_agentic=$OUT/stats-axis3_code_agentic.jsonl \
  --axis axis4_reasoning_termination=$OUT/stats-axis4_reasoning_termination.jsonl \
  --reference $RUN/reference-coder-quant.json \
  --signal mass \
  --out $RUN/heatmap/axis-panels.svg \
  --json $RUN/heatmap/axis-panels.json
```

With all four dumps present there is **no `--allow-synthetic`**, no rename and
no watermark — the absence of the watermark is what tells a reader the figure
is measured. Use `--signal count` if `VLLM_FQ_GATE_MASS` was off (or if
`mass_is_real` is not `true`), and `--order axis:axis3_code_agentic` to sort
the columns by the code axis instead of the pooled mean — the reference quant
is the *Coder* build, so that ordering answers "do the other three axes look
like the corpus this quant was tuned for?".

Cross-check the `vs ref` column against the standalone scorer; they must agree
to the last digit, since both call the same helpers:

```bash
$PY $RUN/score_convergence.py --reference $RUN/reference-coder-quant.json \
    --stats $OUT/stats-axis3_code_agentic.jsonl --signal count
# mean Jaccard 0.3597 for the archived code-axis dump
```

## 5. What the answer will look like

The figure is honest in both directions, and the number to read first is the
**off-diagonal mean** of the pairwise matrix:

* **≥ 0.90** — the axes select nearly the same experts. Printed as
  `NULL RESULT`. One static allocation serves all this traffic and the
  runtime-re-tiering case has to be made on something other than corpus mix.
* **~0.27** — the chance floor for this reference's cardinality (analytic
  0.2652, sampled 0.2641). Overlap at chance means the axes share nothing.
* **in between** — the interesting case; the per-axis `vs ref` column then
  says which axis the human's Coder quant was actually tuned for.

For calibration, two numbers already measured on this model: two *human*
builds from the same calibration data agree at **0.657–0.671**, and the same
corpus replayed twice through a truncated variant of itself gives per-layer
Spearman **0.85–0.94**. So ~0.9 means "same traffic", and ~0.35 means
"genuinely different routing".

## 6. Do not

* Do not re-use `results/k3-fq/stats.jsonl` as an axis: it is the **full
  corpus** (all four axes mixed), and `stats-INVALID-truncated-corpus.jsonl` is
  the retracted run that replayed 160-character display stubs
  (`results/k3-fq/CONVERGENCE-RESULTS.md`).
* Do not append two axes to one `VLLM_FQ_DUMP_STATS` path. The tool reads the
  last record, which would be axis B contaminated by whatever of axis A is
  still inside the 2,048-step window.
* Do not compare a `--signal mass` panel with a `--signal count` panel. The
  tool applies one signal to the whole figure for exactly this reason.
* Do not run this while the encode campaign holds GPUs 4–7 *and* a serve holds
  0–3. `serve-glm52.sh` pins `CUDA_VISIBLE_DEVICES=0,1,2,3` and uses private
  JIT caches, but the boot still needs those four GPUs entirely to itself.

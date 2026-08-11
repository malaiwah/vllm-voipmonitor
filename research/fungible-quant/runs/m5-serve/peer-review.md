# Adversarial peer review — M5 evidence campaign

Scope: everything written in the hour before this review, in priority order —
`runs/m5-serve/{swap_evidence,make_charts}.py` and their tests,
`runs/m5-serve/{serve-glm52,run-evidence}.sh`, the histc sentinel fix in
`gg-vllm/.../exl3_fungible/stats.py`, `runs/0c-campaign/publish_window.py`, and
`runs/health/sweep.sh`.

Method: every finding below was reproduced by running something, not by reading.
Probe scripts, their outputs, and the pre/post-fix regression runs are quoted
inline. Nothing was run on a GPU and no serve was started; the gg-vllm working
tree was read only — its fixes are proposed as diffs.

Baseline state at review time: `pytest test_swap_evidence.py test_make_charts.py`
was **red** (1 failed, 20 passed) — see MAJ-1.

---

## Summary table

| # | Sev | File | One line |
|---|-----|------|----------|
| CRIT-1 | critical | `serve-glm52.sh:54` | empty `VLLM_FQ_TRUST_SIGNERS` silently disables signature **and** predicate trust — strictly weaker than not setting it |
| CRIT-2 | critical | `run-evidence.sh:34-41` | `cleanup()` kills only the API server; EngineCore + 4 TP workers survive holding GPU memory |
| CRIT-3 | critical | `run-evidence.sh:14,128` | no `set -e`; the script exits 0 after any post-probe failure — the "empty-but-plausible result directory" its own header claims to prevent |
| CRIT-4 | critical | `run-evidence.sh:113` | guards on `harness/run-gpqa.sh`, which does not exist (`eval_gpqa.sh` does) — the quality eval is **always** silently skipped |
| MAJ-1 | major | `swap_evidence.py:78` | prompt corpus is seeded on salted `str.__hash__` (not reproducible across processes) and 3 of 4 families cap at 285 distinct prompts — **fixed** |
| MAJ-2 | major | `swap_evidence.py:230` | a repeated family double-counts an earlier phase's requests and reports 2x tok/s — **fixed** |
| MAJ-3 | major | both py files | `NaN`/`Inf` reach the timeline: NaN erases a chart series silently, Inf crashes the renderer — **fixed** |
| MAJ-4 | major | `swap_evidence.py:148` | `startswith` also matches `_created`/`_sum`/`_count`/`_bucket` — **fixed** |
| MAJ-5 | major | `swap_evidence.py:153` | rate on the wall clock with no counter-reset guard → negative / NTP-inflated tok/s — **fixed** |
| MAJ-6 | major | `serve-glm52.sh:61` | `VLLM_FQ_ARTIFACT_DIR` is the same path for `off`/`dryrun`/`live`; the baseline result dir gets the live run's decision log |
| MAJ-7 | major | `publish_window.py:206` | `return 1` on "nothing to publish" is now the *common* case and it disables the supervisor's capture prune, including the emergency disk-pressure one |
| MAJ-8 | major | `publish_window.py:80` | skip is filename-only: a corrupt or differently-provenanced remote layer is never re-uploaded; undocumented, no override |
| MAJ-9 | major | `publish_window.py:214` | `index-k*.json` is rebuilt from local state and uploaded; the skip makes "remote has it, local state does not" normal, so the published index can lose layers |
| MAJ-10 | major | `make_charts.py:172` | cumulative swaps and expert occupancy share one "experts" axis |
| MIN-1..13 | minor | various | parser tolerance, sweep stamp race, log rotation, etc. |

Cleared (checked, no defect): the histc fix (§4), SVG escaping, the two panels'
x-domain identity, real thread concurrency, the parser against a real
multiprocess exposition, encoder shard write atomicity, and the `.pub` file
format vs the resolver's `keyid`.

---

## 1. `swap_evidence.py`

### MAJ-1 (major, will definitely bite) — prompt corpus: not reproducible, and 285 prompts deep
`swap_evidence.py:78-82` (pre-fix)

```python
rng = random.Random(hash((family, i)) & 0xFFFFFFFF)
```

`str.__hash__` is salted per interpreter. Six runs, six corpora:

```
$ for i in 1..6; do PYTHONHASHSEED=random python -c "...build_prompt('math',0)"; done
unique of 24: 24 | math#0 param:  at 47:0
unique of 24: 24 | math#0 param:  at 16:0
unique of 24: 24 | math#0 param:  at 86:0        <- a different corpus every process
```

The module docstring says "a fixed corpus keeps runs comparable". It was not.
Two consequences:

1. **The suite was red on arrival.** `test_prompt_families_are_distinct_and_formatted`
   asserts 24 distinct prompts out of 24 draws; the collision probability over
   12 same-template pairs with `a ∈ [3,97]` is ~12%, so the test passes or fails
   depending on the salt. It was failing (`assert 23 == 24`) when I started.
2. **The corpus ceiling defeats the experiment.** Three of the four families
   interpolate only `{a}`:

```
$ distinct prompts over i = 0..50000
  math            :  43691
  code            :    285      <- 3 templates x 95 values of {a}
  prose_multiling :    285
  biomed          :    285
```

A 420 s phase at concurrency 24 (`run-evidence.sh:92-98`) issues on the order of
1000–2000 requests. Every prompt is re-issued 4–7 times, and vLLM's prefix cache
then serves the prefill from cache. Cached prefill routes no tokens — it
understates exactly the routing pressure the domain shift exists to create. This
is the same class of error that made the M2 dryrun propose zero swaps, and the
existing test's own comment names it ("a repeated prompt would let the prefix
cache serve it and understate real routing pressure") while testing only 6 draws.

**Fixed.** Seed is now `blake2b(f"{family}:{i}")` (stable across interpreters,
platforms and Python versions), and every draw carries a varying leading clause
so it is a prefix-cache miss from the first token.

New tests: `test_prompt_corpus_does_not_run_out_under_a_real_phase` (5000 draws
must be 5000 distinct) and `test_prompt_is_deterministic_ACROSS_PROCESSES`
(three subprocesses with `PYTHONHASHSEED` 0/1/12345 must agree).

### MAJ-2 (major, latent but reproduced) — per-phase double counting
`swap_evidence.py:199,230-231` (pre-fix)

`results` is one list for the whole run and `phase_end` selects out of it with
`r["family"] == family`. Any phase that repeats an earlier family re-counts it.
Measured against a stub server, `--phase math:2 --phase code:2 --phase math:2`
at concurrency 12:

```
phase_end family=math  requests=  84 ok=  84 toks=  840 wall=2.03 tok_s=413.83
phase_end family=code  requests=  92 ok=  92 toks=  920 wall=2.06 tok_s=446.48
phase_end family=math  requests= 180 ok= 180 toks= 1800 wall=2.02 tok_s=891.88
                          ^^^ actually issued 96          ^^^ twice the real rate
```

A wrong tok/s written into the evidence JSONL is precisely the
plausible-but-wrong outcome this review exists to catch. Today's `run-evidence.sh`
uses four distinct families so it does not fire, but an A/B that repeats a family
(the obvious next experiment) silently produces doubled numbers.

Also in the same block: `wall_seconds` and the denominator of `tok_s` were two
separate `time.time()` calls, so the published `tok_s` was never exactly
`completion_tokens / wall_seconds`.

**Fixed.** Fresh `results` list per phase, one monotonic clock read used for both
fields, and `workload()` now returns totals. New test
`test_a_repeated_family_is_not_double_counted` (asserts both the count and the
`tok_s == toks/wall` identity).

### MAJ-3 (major) — `NaN`/`Inf` flow from `/metrics` into the chart
`swap_evidence.py:100-103`, `make_charts.py:68,91` (pre-fix)

`float("NaN")` and `float("+Inf")` parse fine and were stored. Both are legal in
the exposition format, and `fq_jaccard` is a `|A∩B|/|A∪B|` gauge — 0/0 at boot
before any policy exists is a real NaN, not a hypothetical.

Two distinct downstream failures, both reproduced:

```
== fq_view when a gauge is NaN ==
{'swaps_by_layer': {'1': nan, '2': 3.0}, 'swaps_total': nan, ...}

== render with a NaN in swaps_total ==
  'nan' in svg: True
  well-formed XML: yes (but coords are nan)     <- the line silently disappears

== render with an INF in decode_tok_s ==
  RAISED OverflowError : cannot convert float infinity to integer   <<< CRASH
```

The NaN case is the dangerous one: the SVG still parses, every renderer just
drops the path, and a chart with no swap line reads as "no swaps happened". The
Inf case crashes in `nice_ticks`' `int(abs(raw))` — after the whole evidence run
is already spent.

`json.dumps` also writes bare `NaN`/`Infinity`, which is not valid JSON for any
reader other than Python's own — the timeline would break `jq` and every JS
consumer.

**Fixed at both ends.** `parse_metrics` drops non-finite values and *reports*
them (they land in `row["scrape_warnings"]["nonfinite"]`, so the information is
not lost, just kept out of the arithmetic). `make_charts` gained a `finite()`
predicate used by `nice_ticks`, `Panel.fit`, `Panel.line`, `Panel.frame` and
`occ_at`.

### MAJ-4 (major) — token counter matched by prefix
`swap_evidence.py:148` (pre-fix): `[v for k, v in flat.items() if k.startswith(name)]`

`vllm:generation_tokens` is a prefix of `vllm:generation_tokens_created` (the
~1.7e9 unix timestamp prometheus_client emits beside a Counter in single-process
mode) and of `_sum`/`_count`/`_bucket` if the metric is ever exposed as a
histogram. Summing those into the counter adds a large constant; the derived
*rate* stays right until a new label set appears mid-run, at which point it
spikes by ~3e8 tok/s. Reproduced:

```
$ series(flat, "vllm:generation_tokens")     # with _created/_sum/_count/_bucket present
  old startswith -> [1770000000.0, 9.0, 2.0, 2.0, 500.0]  sum = 1770000513.0
  new exact      -> [500.0]
```

The primary path was verified safe against a **real** exposition: I built one with
`prometheus_client 0.26.0` in multiprocess mode using the exact metric shapes
from `loop.py:198-234` and `vllm/v1/metrics/loggers.py:708`, and confirmed
`vllm:generation_tokens_total`, `vllm:num_requests_running`, all four `fq_*`
gauges and both `fq_*_total` counters parse and aggregate correctly, with no
`_created` series and no `pid` label (the gauges use
`multiprocess_mode="mostrecent"`). So this is defence for the fallback names, not
a live bug — but it is a fallback that fires exactly when someone is already
confused about why the chart is empty.

**Fixed** — `series()` matches `name` or `name{`. New test
`test_series_matches_the_exact_metric_name_only`.

### MAJ-5 (major) — counter reset and wall-clock rate
`swap_evidence.py:153-155` (pre-fix)

```python
if prev_tok is not None and t0 > prev_t:
    row["decode_tok_s"] = (tok - prev_tok) / (t0 - prev_t)
```

Answering the specific questions:

* **first sample** — correctly produces no rate. Fine.
* **missed scrape** — `prev_tok`/`prev_t` are not updated on the error path, so
  the next success averages over the whole gap. Correct behaviour, keep it.
* **metric name differs from all three** — no `gen_tokens_total` key at all, no
  rate, and (pre-fix) nothing said so. The chart just had an empty top panel. See
  the new `render()` guard below.
* **counter reset** — unguarded. A negative rate is possible and its blast radius
  is worse than it looks: `Panel.fit` only inspects the max, so `ymax` is
  unchanged and the point lands far outside the 560-tall viewBox:

```
=== render with a NEGATIVE decode_tok_s (counter reset) ===
  out-of-viewBox path points: [('517.8', '65463.4')]
```

  and an all-negative series inverts the axis outright — the ticks read `0` while
  the plotted values are −100 to −109, i.e. a falling series drawn as if rising.
* **absurd rate** — yes. `time.time()` is the wall clock; an NTP step during a
  half-hour run can make `t0 - prev_t` a few milliseconds and turn 600 tok/s into
  600 000 tok/s.

**Fixed.** The rate is derived on `time.monotonic()` (the row timestamp stays on
the wall clock, which is what the phase rows need); a backwards counter sets
`row["counter_reset"] = True` and emits no rate. Test
`test_decode_rate_is_never_negative_on_a_counter_reset` drives a stub whose
counter goes 1000 → 5.

`Panel.fit` was additionally hardened so a negative value widens the domain
downward instead of inverting the axis — defence in depth, since it is the chart's
job not to lie about data it is handed.

### MIN-1 (minor, fixed) — parser tolerance
Probed with a hostile exposition. Pre-fix behaviour:

| input | pre-fix | post-fix |
|---|---|---|
| `foo_bucket{le="0.5"} 1 # {trace_id="abc"} 0.5 …` (exemplar) | dropped silently | parsed |
| `foo_ts 42 1520879607789` (timestamp) | dropped silently | parsed |
| `  spaced 7` (leading whitespace, legal) | dropped silently | parsed |
| `foo{a="x}y",b="z"} 4` (`}` in a label value) | dropped silently | dropped, **counted** |
| `foo{a="x\"y"} 3` (escaped quote) | parsed | parsed |
| `dup 1` / `dup 2` (duplicate series) | last wins | last wins (MIN-2) |

Exemplars/timestamps/whitespace now parse. A `}` inside a label value still
cannot be handled by a regex with `[^}]*`, and switching to a greedy `.*` makes
exemplar lines mis-parse *silently and wrongly* (it grabs the exemplar's value),
which is worse. Instead, unparsable non-comment lines are now **counted** and the
count is written to `row["scrape_warnings"]["unparsed"]`, so a silent drop becomes
a visible one. Test `test_unparsable_lines_are_counted_not_silently_dropped`.

**MIN-2 (minor, not fixed):** duplicate series are last-wins rather than summed.
Prometheus treats duplicates as an error and the multiprocess collector never
emits them, so this is theoretical; noted for completeness.

### CLEARED — the workload really is concurrent
Threads + blocking urllib: `urlopen` releases the GIL in the socket read, so
concurrency is real. Measured in-flight requests at a threaded stub server:

```
peak concurrent in-flight at the server: 12 (asked for 12)
```

### CLEARED — thread safety of `results` and `counter`
Both mutations are inside the same `lock`, and the post-join read happens after
every worker has exited. Correct. (The *selection* out of `results` was wrong —
MAJ-2 — but that is not a data race.)

### MIN-3 (minor, theoretical, not fixed) — two writers on one JSONL
In `both` mode the scraper and the workload each hold their own `open(out, "a")`.
Both `write()`+`flush()` a whole line, and O_APPEND writes to a regular file are
atomic with respect to the offset under `i_rwsem`, so interleaving is safe in
practice even for the ~8 KB sample rows (76 layers of occupancy). Left as-is;
worth a single writer if the row size ever grows.

### New guard — a run that generated nothing now says so
A workload where every request fails still leaves a complete, plausible
timeline: samples, phase rows, a chart with a flat line. `main()` now returns 3
with `FATAL: no request succeeded (N attempted)`. Test
`test_a_workload_that_never_succeeds_exits_nonzero`. Each scrape row also carries
`fq_present`, so a baseline serve and a live serve whose metric names moved are
distinguishable after the fact.

Dead parameter `scrape_loop(..., phases=None)` removed — it was never written to
and invited a caller to pass phases expecting them to be recorded.

---

## 2. `make_charts.py`

### `nice_ticks()` — audited against the specific inputs asked about

```
nice_ticks(0, 0)            = [0]                     ok
nice_ticks(0, 1e-9)         = [0.0]                   one tick only (MIN-5)
nice_ticks(0, 0.0001)       = [0.0]                   one tick only
nice_ticks(-5, 5)           = [-4,-2,0,2,4,6]         ok
nice_ticks(-7, -1)          = [-6,-4,-2,0]            emits a tick past hi (MIN-5)
nice_ticks(0, 1e9)          = [0,2e8,…,1e9]           ok
nice_ticks(0, 1e300)        = [0,2.5e299,…,1e300]     ok
nice_ticks(-1e300, 1e300)   = huge Python ints        ok but ugly
nice_ticks(0, inf)          = OverflowError           CRASH  (MAJ-3)
nice_ticks(0, nan)          = [0, nan]                NON-FINITE TICK  (MAJ-3)
nice_ticks(nan, 1)          = ValueError              CRASH  (MAJ-3)
```

Direct answers:
* `mag = 10 ** (len(str(int(abs(raw)))) - 1) if abs(raw) >= 1 else 0.1` — the
  `< 1` branch floors `mag` at `0.1`, so **step is never zero**. For `raw` at
  1e-9 that means a single tick and no usable scale, but not a degenerate axis.
* **step is never infinite** — `raw = inf` raises before a step is computed.
* **the tick list is never empty** — `out or [lo, hi]` covers it, and I could not
  construct a finite input that reaches the fallback.
* **negatives**: `start = int(lo/step)*step` truncates toward zero, so for
  `lo = -7, step = 2` the first tick is −6, not −8; and the `v <= hi + step*0.5`
  bound admits one tick past `hi`. Both are cosmetic — `Panel.frame` filters
  ticks outside the plot rect — and unreachable from the current call site
  (`ymin` was always 0). Left as-is; flagged (MIN-5).

**Fixed:** `nice_ticks` returns `[0.0]` for non-finite bounds instead of raising.
Test `test_nice_ticks_survives_non_finite_bounds`.

### `Panel.fit()` — audited

```
all zero                 ymax=1.0     ok, `or 1.0` covers it
all None / empty         ymax=1.0     ok
all negative [-5,-3]     ymax=-3.45   ymin=0 > ymax: axis INVERTED
nan                      ymax=nan     every coordinate becomes nan
inf                      ymax=inf     OverflowError in nice_ticks
```

So `or 1.0` handles exactly the case it was written for and nothing else.
**Fixed:** non-finite values are excluded, `ymax = top*1.15 if top > 0 else 1.0`,
and `ymin = bot*1.15 if bot < 0 else 0.0` so a negative series is drawn inside
the panel with correct ticks rather than off the bottom edge. Tests
`test_fit_never_inverts_the_axis_on_negative_values`,
`test_fit_ignores_non_finite_values`,
`test_a_negative_sample_stays_inside_the_viewbox`.

### CLEARED — SVG escaping
`esc()` covers `&`, `<`, `>`, `"` and is applied to every user-controlled string:
title, subtitle, y-labels, phase family names, x-tick labels, legend labels.
Probed with `</svg><script>alert(1)</script> & "quoted" <b>` as *both* the title
and a family name:

```
raw '<script' leaked: False
raw '</svg>' count (should be 1): 1
well-formed with hostile text: yes
```

`'` is not escaped, but every attribute is double-quoted, so that is correct, not
an oversight. The `CSS` constant contains no literal `%` that would collide with
its `%`-formatting. No defect.

### CLEARED — the two panels share one x domain
Both are constructed with the same `L`, `pw`, `t0`, `t1`, so `px()` is
identical by construction; verified numerically at three points, and now locked
by `test_the_two_panels_share_one_x_domain`.

### MIN-4 (minor, fixed) — phase bands drawn outside the plot
`t0`/`t1` come from **sample** rows only. A `phase_start` written a few
milliseconds before the scraper's first row (a genuine race in `both` mode —
`main()` starts the scraper thread and immediately calls `workload`) lands left
of the plot:

```
=== phase row BEFORE the first sample ===
  dashed line x: ['52.1']  (panel x0=70)
```

The dashed rule and the family label then overprint the y-axis. Fixed by clamping
band edges to the plot rect. Test
`test_phase_band_before_the_first_sample_stays_inside_the_panel`.

### New guard — a timeline with no throughput at all is refused
`render()` now raises `SystemExit` if no sample carries a finite `decode_tok_s`.
Previously such a timeline (every scrape errored, or the counter name moved)
produced a chart with an empty top panel and no complaint — indistinguishable
from an idle serve. Test `test_a_timeline_with_no_throughput_at_all_fails_loudly`.

### MAJ-10 (major, design, NOT fixed) — two units on one axis
`make_charts.py:172-173` fits the bottom panel to
`swaps_total ∪ occupancy(K5) ∪ occupancy(K4)` and labels the axis "experts".
`swaps_total` is a **monotonically increasing cumulative counter**; the occupancy
series are **bounded instantaneous counts**. The module docstring is explicit that
dual-axis charts are banned, which is right — but the fix chosen (one shared
axis) puts a quantity that grows without bound next to one that does not. If
`swaps_total` reaches a few thousand while `occupancy(K5)` sits at a few hundred,
`ymax` is set by the counter and the occupancy lines flatten onto the baseline —
the chart would show "swaps happened" and hide "which experts moved", which is
the more interesting half of the claim.

I did not change this: it is a layout decision, not a bug, and the right answer
(a third stacked panel for the cumulative counter, or plotting the per-interval
swap *delta* instead of the cumulative total against occupancy) changes what the
figure asserts. Recommend deciding it deliberately before the PR, ideally by
rendering the real timeline once and looking.

---

## 3. `serve-glm52.sh` and `run-evidence.sh`

### CRIT-1 — an empty `VLLM_FQ_TRUST_SIGNERS` silently weakens trust
`serve-glm52.sh:54`

```bash
export VLLM_FQ_TRUST_SIGNERS=$(cat "$HOME/.fq_keys/fq_signing.pub" 2>/dev/null || true)
```

The resolver (`fragments.py:495-502`) branches on `is not None`, not on
truthiness:

```python
signers_raw = environ.get(FQ_TRUST_SIGNERS_ENV)
if signers_raw is not None:
    signers = [s.strip().lower() for s in signers_raw.split(",") if s.strip()]
else:
    manifest_key = self.manifest.get("signer_pubkey")
    signers = [str(manifest_key).lower()] if manifest_key else []
self.trust_signers = frozenset(signers)
self.trust_enabled = bool(self.trust_signers)
```

An exported-but-empty value therefore yields `signers == []`, `trust_enabled ==
False`, and `_evaluate_attestation` short-circuits to legacy sha-only:

```python
if not self.trust_enabled:
    shas = _attestation_expert_shas(text)
    return (shas, None) if shas else (None, "no-attestation")
```

**This means the empty value is strictly WEAKER than not setting the variable at
all** — unset falls back to the manifest's `signer_pubkey`, empty falls back to
nothing. Proven against the real resolver (`/tmp/.../test_empty_signer.py`,
3 passed):

```
unset  -> trust_enabled=True,  rogue-signed fragment REJECTED ("signer not-trusted")
empty  -> trust_enabled=False, trust_signers=set()
          rogue-signed fragment ACCEPTED, origin = fetched
          untrusted predicate ("rumour-of") ACCEPTED   <- VLLM_FQ_TRUST_PREDICATES bypassed too
```

So a missing `.pub` file turns a serve that pulls expert fragments from a public
HF repo over the network into one that accepts any attestation carrying an
`expert_sha256` map, from any key, under any predicate — while
`VLLM_FQ_VERIFY=all` and `VLLM_FQ_TRUST_PREDICATES=repack-of,encode-of` are still
printed in the env dump at `serve-glm52.sh:86`, which reads as if trust were on.

The `.pub` file does exist now (65 bytes = 64 hex chars + newline, starting
`a58b`, matching `Signer.pub_hex`'s format and the published fingerprint), so
this is not currently firing. The failure mode still matters: it is silent, it
degrades security, and `|| true` is the only thing standing between the campaign
and it.

**Yes, the script should fail loudly.** Proposed:

```diff
--- a/research/fungible-quant/runs/m5-serve/serve-glm52.sh
+++ b/research/fungible-quant/runs/m5-serve/serve-glm52.sh
@@
-    # Trust: only fragments signed by our pinned key, and only honest rungs.
-    export VLLM_FQ_TRUST_SIGNERS=$(cat "$HOME/.fq_keys/fq_signing.pub" 2>/dev/null || true)
+    # Trust: only fragments signed by our pinned key, and only honest rungs.
+    # An EMPTY value here is not "no signer configured" — the resolver
+    # branches on `is not None`, so an empty string suppresses even the
+    # manifest's own signer_pubkey anchor and drops to legacy sha-only,
+    # accepting any attestation from any key under any predicate. Empty is
+    # strictly weaker than unset, so refuse to boot rather than export it.
+    FQ_PUB=$HOME/.fq_keys/fq_signing.pub
+    if [ ! -s "$FQ_PUB" ]; then
+      echo "FATAL: $FQ_PUB missing/empty — refusing to serve with trust off" >&2
+      exit 3
+    fi
+    VLLM_FQ_TRUST_SIGNERS=$(tr -d '[:space:]' < "$FQ_PUB")
+    case "$VLLM_FQ_TRUST_SIGNERS" in
+      [0-9a-fA-F]*) [ ${#VLLM_FQ_TRUST_SIGNERS} -eq 64 ] || {
+        echo "FATAL: $FQ_PUB is not a 64-hex ed25519 pubkey" >&2; exit 3; } ;;
+      *) echo "FATAL: $FQ_PUB is not hex" >&2; exit 3 ;;
+    esac
+    export VLLM_FQ_TRUST_SIGNERS
```

Separately worth considering upstream: `fragments.py` should treat an
*explicitly empty* `VLLM_FQ_TRUST_SIGNERS` as an error, or at minimum log
`WARNING: trust filtering disabled`, rather than silently returning to legacy
behaviour. Right now nothing in the log distinguishes the two modes.

### CRIT-2 — `cleanup()` orphans the TP workers
`run-evidence.sh:31-41`

`serve-glm52.sh` ends in `exec "$GG" python -m vllm...` and `gg-run.sh` ends in
`exec env ... "$target"`, so `SERVE_PID` really is the API server — that part is
right. But vLLM v1 with `--tensor-parallel-size 4` and
`VLLM_WORKER_MULTIPROC_METHOD=spawn` runs EngineCore and four workers as separate
processes, and `cleanup()` signals only the one PID. Reproduced with a
process-shaped stand-in (no GPUs involved):

```
SERVE_PID=413532  my_pgid=413529  serve_pgid=413529
workers before: 5
workers AFTER cleanup(): 5   <-- these hold the GPU memory
```

Note `serve_pgid == my_pgid`: the script is non-interactive, so job control is
off and the backgrounded serve stays in **run-evidence.sh's own process group** —
`kill -- -$PGID` would kill the script too. The fix has to create a new group:

```diff
-FQ_MODEL=$CKPT FQ_PORT=$PORT "$RUN/serve-glm52.sh" "$MODE" \
-  > "$OUT/serve.log" 2>&1 &
+FQ_MODEL=$CKPT FQ_PORT=$PORT setsid "$RUN/serve-glm52.sh" "$MODE" \
+  > "$OUT/serve.log" 2>&1 &
 SERVE_PID=$!
+SERVE_PGID=$SERVE_PID          # setsid makes the child its own group leader
 cleanup() {
   say "shutting down serve pid=$SERVE_PID"
-  kill "$SERVE_PID" 2>/dev/null
+  kill -TERM -- "-$SERVE_PGID" 2>/dev/null      # whole group: engine + TP workers
   for _ in $(seq 1 30); do kill -0 "$SERVE_PID" 2>/dev/null || break; sleep 2; done
-  kill -9 "$SERVE_PID" 2>/dev/null
+  kill -KILL -- "-$SERVE_PGID" 2>/dev/null
+  # last-resort sweep; orphaned CUDA contexts have wedged this box before
+  pkill -9 -f "VLLM::EngineCore" 2>/dev/null
+  say "post-cleanup vllm procs: $(pgrep -cf 'vllm.*api_server|VLLM::EngineCore' || echo 0)"
 }
-trap cleanup EXIT
+trap cleanup EXIT INT TERM
```

The `INT TERM` addition matters too: bash does not reliably run an `EXIT` trap
when it is killed while blocked in the 100-minute readiness `sleep`.

### CRIT-3 — `set -uo pipefail` without `-e`: the script always exits 0
`run-evidence.sh:14`, and the tail at `:128-129`

The header says "Deliberately fail-loud: … the script stops rather than producing
an empty-but-plausible result directory." That is true only for the four explicit
`exit 1`s during boot and probe. Everything after the probe continues on failure,
and the script's last command is `ls -la … | tee`, whose status is `tee`'s 0. So:

* `swap_evidence.py` crashing → the run continues, charts are attempted, `DONE` is
  logged, exit 0.
* `make_charts.py` raising `SystemExit("no samples in timeline")` → `DONE` is
  logged anyway, exit 0.
* `mkdir -p "$OUT"` at line 24 has already created the directory, so a caller or
  a later sweep sees `results/<tag>/` with a `run.log` ending in `DONE` and no
  chart.

Minimum fix:

```diff
-set -uo pipefail
+set -euo pipefail
+fail() { say "FATAL: $*"; exit 1; }
```
plus explicit `|| fail …` on the steps that are legitimately allowed to fail
(`grep` on no match at :108, the optional `cp -a` at :105, the final `/metrics`
scrape at :102 which already has its own `|| say WARN`), and a final
`say "DONE"` that is only reached on success. Also worth asserting the deliverable
exists before declaring victory:

```bash
[ -s "$OUT/swap-timeline.svg" ] || fail "no chart produced"
[ "$(wc -l < "$OUT/timeline-main.jsonl")" -ge 10 ] || fail "timeline too short"
```

### CRIT-4 — the quality eval never runs
`run-evidence.sh:113`

```bash
if [ "${FQ_SKIP_EVAL:-0}" != 1 ] && [ -x "$RUN/harness/run-gpqa.sh" ]; then
```

`runs/m5-serve/harness/` contains `eval_gpqa.sh`, `eval_gsm8k.sh`,
`decode_bench.sh`, `preflight.sh`, `saturate.py`, `stub_server.py`,
`subsample.py`, `README.md` — there is **no `run-gpqa.sh`**. The guard is always
false, so the campaign takes the else branch and logs

```
skipping eval (FQ_SKIP_EVAL=0, harness present=no)
```

which reads like a deliberate choice. The M5 evidence campaign would therefore
produce zero quality numbers, and the `--max-model-len 32768` reasoning in
`serve-glm52.sh:98-100` ("a truncated GPQA reasoning trace scores as WRONG")
would be defending a step that never executes. Fix is to point at the real script
and to make a *missing* harness fatal rather than skippable:

```diff
-if [ "${FQ_SKIP_EVAL:-0}" != 1 ] && [ -x "$RUN/harness/run-gpqa.sh" ]; then
+if [ "${FQ_SKIP_EVAL:-0}" = 1 ]; then
+  say "eval skipped by FQ_SKIP_EVAL=1"
+elif [ -x "$RUN/harness/eval_gpqa.sh" ]; then
   say "quality eval (GPQA Diamond)"
-  "$RUN/harness/run-gpqa.sh" "$BASE_URL" "$OUT/gpqa.json" \
+  "$RUN/harness/eval_gpqa.sh" "$BASE_URL" "$OUT/gpqa.json" \
     2>&1 | tee -a "$OUT/run.log"
 else
-  say "skipping eval (...)"
+  fail "harness/eval_gpqa.sh missing — refusing to call this run complete"
 fi
```
(check `eval_gpqa.sh`'s actual argument contract before wiring it — I did not run it.)

### MAJ-6 — `VLLM_FQ_ARTIFACT_DIR` is shared across arms
`serve-glm52.sh:61` hardcodes `$RUN/artifacts` for every mode, with no per-tag
override and no clearing at boot. `run-evidence.sh:104-107` then does
`cp -a "$RUN/artifacts" "$OUT/fq-artifacts"`.

Consequence: run `live` then `off`, and `results/off/fq-artifacts/` contains the
**live** run's decision log and committed policy. For an A/B whose whole point is
"FQ off vs FQ on", shipping the on-arm's swap decisions inside the off-arm's
evidence directory is a provenance hazard that a reviewer would (correctly) call
fabrication.

```diff
-    export VLLM_FQ_ARTIFACT_DIR=$RUN/artifacts
+    export VLLM_FQ_ARTIFACT_DIR=${FQ_ARTIFACT_DIR:-$RUN/artifacts/$MODE-$(date -u +%Y%m%dT%H%M%SZ)}
```
and have `run-evidence.sh` export `FQ_ARTIFACT_DIR=$OUT/fq-artifacts` so the loop
writes straight into the run's own directory and the `cp -a` disappears.

### Other `run-evidence.sh` notes
* **MIN-12 (minor):** `$OUT` is interpolated into a `python -c` string literal at
  `:67-74`; a `'` in `TAG` (operator-supplied `$3`) breaks or injects. Use
  `$PY - "$OUT/probe.json" <<'EOF'` and read `sys.argv[1]`.
* Charts are rendered only for `timeline-main.jsonl`; `timeline-baseline.jsonl` is
  recorded and never drawn or summarised.
* The probe path is genuinely fail-loud and correct: `curl -sf` failing writes an
  empty `probe.json`, `json.load` raises, `PROBE_PARSE_FAIL` is matched, exit 1.
  Empty/whitespace content is also caught by the `""` case. Good as written.

**Answering "is any step capable of producing an EMPTY-but-plausible result
directory?"** — yes, three ways: CRIT-3 (any post-probe failure still logs `DONE`
and exits 0), CRIT-4 (the eval silently no-ops), and a workload where every
request fails but the scraper keeps writing rows. The third is now fixed in
`swap_evidence.py` (exit 3) and `make_charts.py` (refuses a throughput-free
timeline); the first two need the shell changes above.

---

## 4. The histc sentinel fix — verified independently, **no defect found**

`stats.py:137-156`. I re-derived the claim from scratch rather than trusting the
comment. `bins = E+1, min = 0, max = E+1` makes every bin exactly 1.0 wide: bin
`i` covers `[i, i+1)` for `i < E`, and bin `E` is `[E, E+1]` (closed at max, per
histc's documented last-bin rule). Slicing `[:E]` therefore keeps exactly the
real experts.

Tested on CPU under the GG runtime (`torch 2.12.0+cu132`, `CUDA_VISIBLE_DEVICES=""`),
fast path vs the weighted `scatter_add_` path:

```
E=8:  in range / id==E / id==E+1 / id==E+2 / id<0 / mixed / huge(2^31-1) / empty
      -> all match, all OOR dropped
exhaustive, every id in [-3, E+3], E ∈ {2,3,8,255,256,257,1024}:  mismatches = none
E=256:  arange(0,256) -> all-ones, sum=256;   id==E x1000 -> 0;   id==E+1 x1000 -> 0
E=8192: arange(0,8192) -> all-ones, sum=8192; id==E x1000 -> 0;   id==E+1 x1000 -> 0
```

**Float precision at large E.** CPU histc self-corrects against computed bin
edges, so the CPU result proves little about the GPU. I therefore simulated the
CUDA kernel's own binning formula (`SummaryOps.cu::getBin`,
`bin = (int)((v - min) * nbins / (max - min))` in `acc_type<float,true>` = float,
with the `bin == nbins → bin -= 1` clamp) in numpy float32:

```
E=   256: misbinned ids = [] (0)
E=  4096: misbinned ids = [] (0)
E=  8192: misbinned ids = [] (0)
E= 16384: misbinned ids = [] (0)
```

This is worth stating explicitly because it is *not* obvious: for E=8192 the
intermediate product `v * (E+1)` reaches 6.7e7, well past float32's 2^24 exact
range. It survives because `E+1 = 8193` needs only 14 mantissa bits, so
`v*(E+1)` is either exact or rounds within half an ulp of the correct quotient.
It is a property of this particular `bins == max` choice, not a general one — a
future change to the bin count could break it silently, which argues for keeping
the exhaustive check as a test.

**"ids strictly between E and E+1 cannot occur for integer inputs"** — confirmed
and trivially true: the input is `topk_ids` (integer), `.to(torch.float32)`
preserves every integer below 2^24 exactly (verified: exact at E−1 for E=256,
8192 and 2^24; first failure at 2^24+2, far outside any plausible expert count),
so no value can land in the open interval. Even if one did, it would fall in
bin E and be sliced off — the fix is robust to that case anyway.

**Fast path vs weighted path semantics for in-range ids** — identical: histc adds
+1 per occurrence, `scatter_add_` adds `ones_like` per occurrence, both into a
float32 buffer. The only difference is where OOR ids go (histc drops them, the
weighted path parks them in the overflow slot at index `E`), and `step()`
(`stats.py:193`) copies `count_buf[:e]`, so the overflow slot never reaches the
window either way. Consistent.

**Two things I checked that are not in the comment but should be:**

1. **No host sync, so it is CUDA-graph safe.** `_histc_cuda` only falls back to
   `self.min()/self.max()` — a device→host copy that would break graph capture —
   when `minvalue == maxvalue`. Passing `min=0, max=E+1` avoids that branch by
   construction. This is load-bearing for the whole M1 design (the capture fn
   must be recordable into CUDA graphs) and is currently an implicit dependency
   on an ATen implementation detail. Suggest a one-line comment.
2. **float32 accumulation headroom.** `count[:E].add_(hist[:E])` accumulates
   across a whole `window_stride` (32 steps) before `step()` rolls and zeroes.
   Worst case at the M5 serve settings (`max_num_batched_tokens` ~8192, topk 8)
   is ~2.1M increments per window per layer if a single expert took everything —
   about 8x below 2^24. Comfortable, but it is a ~200x margin rather than an
   infinite one, and it shrinks linearly with `window_stride`. Worth naming.

Proposed comment-only diff (I did not apply it — the gg-vllm tree has live
uncommitted work):

```diff
--- a/vllm/model_executor/layers/quantization/exl3_fungible/stats.py
+++ b/vllm/model_executor/layers/quantization/exl3_fungible/stats.py
@@
                 # fall outside the range and are dropped by histc itself.
                 # Exact for id values < 2^24 in fp32.
+                #
+                # Two properties this relies on, both verified rather than
+                # assumed (peer review 2026-08-11):
+                #  * Passing an explicit min/max avoids _histc_cuda's
+                #    min==max fallback, which does a device->host copy and
+                #    would make this capture fn ungraphable.
+                #  * The CUDA bin formula is (int)(v*nbins/(max-min)) in
+                #    fp32. With nbins == max == E+1 the product v*(E+1) is
+                #    exact or rounds within half an ulp of v for every
+                #    integer v <= E+1, checked exhaustively up to E=16384.
+                #    A different bins/max relationship would NOT be safe.
+                #  * count[:E] accumulates a whole window_stride before
+                #    step() zeroes it: ~2.1M increments worst case at the
+                #    M5 serve settings, ~8x under fp32's 2^24 exact range.
                 flat = topk_ids.flatten().to(torch.float32)
```

I also ran the package's CPU suite to make sure nothing else regressed:

```
$ CUDA_VISIBLE_DEVICES="" gg-run.sh python -m pytest tests/exl3_fungible/ -q --noconftest \
    --ignore=…test_swap_gpu.py --ignore=…test_swap_t5_gpu.py
142 passed, 1 skipped in 1.85s
```

Recommend adding the exhaustive OOR check as a real test in
`tests/exl3_fungible/test_stats_cpu.py` (there is currently one histc-related
assertion at line 137); the E=8192 case in particular is the one that would
silently regress.

---

## 5. `publish_window.py` — the skip-already-published logic

### The layer-number parse
`publish_window.py:80`: `int(f.stem.replace("tr3-layer-", ""))` over
`work.glob("tr3-layer-*.safetensors")`.

Robust for the files that actually exist. I checked the writer: the encoder
bundle (`encode_tr3_v31.py:292-304`, :2042-2045) writes
`tr3-layer-NNN.safetensors.tmp` then `os.replace`, and writes the `.done.json`
marker afterwards, also atomically. `tr3-layer-NNN.safetensors.tmp` does **not**
match the glob, so **a half-written shard can never be staged** — my initial
concern here was unfounded, and the concurrency between encoder and publisher is
safe.

**MIN-9 (minor):** any unexpected name (`tr3-layer-.safetensors`,
`tr3-layer-007-retry.safetensors`) raises an unhandled `ValueError` that aborts
the publish for *all* Ks; the supervisor logs only `publish failed (retry next
pass)` and retries forever. `int(f.stem.removeprefix("tr3-layer-"))` inside a
`try` that warns and skips would be strictly better. Related:
`rebuild_remote_manifest` does `f.split(".k")[1][0]` — one character — so it
silently mis-parses K≥10. `KS = (2,4,5)` today, so latent.

### Could it skip a layer that exists remotely but is CORRUPT or from a different provenance?
**Yes, both.** `already` is a set of *filenames* from `list_repo_files`. Nothing
compares bytes, sizes, capture fingerprints or encoder shas. A layer uploaded
from an earlier capture run, or truncated by a failed upload, is skipped forever
and there is no `--force`.

**Is that an acceptable trade?** For the disk-thrash problem it solves, mostly
yes — regenerating 236 GB on a box with 250 GB free while an assembly runs is a
worse failure. But it is not *documented as a trade* anywhere: the code comment
explains only the disk rationale, and there is no escape hatch. Two cheap
improvements, in order of value:

1. **Compare digests, not names.** `HfApi().list_repo_tree(REPO, expand=True)`
   returns `lfs.sha256` per file. The local `state.json` already stores each
   segment's sha. Skipping only when the shas agree turns "never re-upload a
   fixed version" into "re-upload exactly the broken ones", for the same one API
   call.
2. **`--force` / `FQ_REPUBLISH=layer-042.k5,…`**, plus a docstring line saying in
   as many words: *a segment already present on the remote is trusted by name; if
   you need to replace one, delete it on the Hub or pass --force.*

### MAJ-7 — `return 1` on "nothing to publish" disables the prune
`publish_window.py:206-207` returns 1 when `total == 0`. The supervisor
(`campaign-supervisor.sh:140-158`) treats that as failure:

```bash
if $PY "$CAMP/publish_window.py" >> "$CAMP/publish-auto.log" 2>&1; then
  …prune this window's capture…
else
  log "publish failed (retry next pass)"
fi
```

and the disk-pressure path at :103-104 is worse — it is an `&&` chain:

```bash
$PY "$CAMP/publish_window.py" >> "$CAMP/publish-auto.log" 2>&1 && \
  find "$CAPTURE" -maxdepth 1 -name 'layer_*' -mmin +120 -exec rm -rf {} +
```

Before the skip commit, "nothing to publish" was rare (every layer was
regenerated, so `total > 0` almost always). **After it, "everything is already on
HF" is the normal steady state** — which means the emergency disk reclaim is now
disabled precisely when the campaign is caught up and has nothing new to upload.
That is a regression against the 455→213 GB problem the prune was written for,
and it will present as "the campaign wedged on disk" with `publish failed` in the
log and no explanation.

```diff
-    if not total:
-        print("nothing to publish", flush=True)
-        return 1
+    if not total:
+        # Success, not failure: the supervisor gates its capture prune (and
+        # the emergency disk reclaim) on this exit status, and with the
+        # skip-already-published check "nothing new" is the steady state.
+        print("nothing to publish — remote is already complete", flush=True)
+        return 0
```
If a distinct signal is wanted, use a dedicated code (e.g. 4) and have the
supervisor treat `0|4` as prunable.

### MAJ-9 — `index-k*.json` can be clobbered by the same class of bug the docstring fixed
`publish_window.py:214-216` uploads the whole `OUT` folder with
`ignore_patterns=["state.json", "*.part", ".huggingface*", "fq-manifest.json"]`.
`index-k*.json` is **not** in that list, and `fq_repack.index_for_k`
(`fq_repack.py:254`) rebuilds it "purely from the K's own state entries".

Today this is safe: `state.json` lives in `OUT`, is excluded from upload, and
therefore persists across windows carrying every layer ever repacked locally.
But the skip logic makes *"the remote has a layer that local state does not"* the
normal condition. If `state.json` is ever lost or the segment dir is rebuilt on a
fresh path, the next publish rebuilds `index-kK.json` from a partial state and
uploads it over the complete remote one — the resolver then cannot find layers
whose bytes are still sitting in the repo. That is bug #2 from this file's own
docstring, one level down.

Cheapest hardening: add `"index-k*.json"` to `ignore_patterns` and upload each
index explicitly only after merging it with the remote copy — or, mirroring the
manifest fix, rebuild the indexes from the remote inventory. At minimum, refuse
to upload an index with fewer layers than the remote one already has.

### MIN-10 / MIN-11
* The encoder writes `file_sha256` into every `layer-NNN.done.json` and the
  publisher never reads it. `fq_repack.verify_source_shard(shard, declared)`
  cross-checks only when `declared` is non-None, and the staged temp dir has no
  `MANIFEST.sha256`, so `declared` is always None — a free integrity check on the
  exact bytes about to be attested is being left on the table.
* `reattest_encode_of` globs *all* local attestations for a K and calls
  `seg.read_bytes()` on any still carrying `repack-of`. If a publish ever dies
  between repack and re-attestation and the segment is later pruned, the next run
  raises `FileNotFoundError`. Narrow, but it aborts the whole publish.

---

## 6. `runs/health/sweep.sh`

### MIN-6 — the stamp file races
`sweep.sh:16-27`. `grew()` does read → `grep -v > "$STAMP.tmp"` → `mv` → append,
six times per sweep, with a **fixed** temp path and no lock. Two concurrent
sweeps:

```
entries after a clean seeding sweep:     6  (expect 6)
mv: cannot stat '/tmp/claude-1000/racetest-stamp.tmp': No such file or directory
mv: cannot stat '/tmp/claude-1000/racetest-stamp.tmp': No such file or directory
entries after two CONCURRENT sweeps:     8  (expect 6)
```

One sweep's `.tmp` gets moved out from under the other, producing duplicate rows,
stderr noise, and — under a less lucky interleaving — *lost* rows for keys the
other sweep had just rewritten. The observed harm is always in the
under-reporting direction (a lost row reads as `first-look`, never as a false
`STALLED`), and `grep … | tail -1` still picks the newest duplicate, so the
verdicts stayed correct in my run. Minor, but trivially fixable:

```diff
 STAMP=/tmp/claude-1000/fq-sweep-sizes
 mkdir -p "$(dirname "$STAMP")"; touch "$STAMP"
+exec 9>"$STAMP.lock"; flock 9        # one sweep at a time
@@
-  grep -v "^$key " "$STAMP" > "$STAMP.tmp" 2>/dev/null; mv "$STAMP.tmp" "$STAMP"
+  local tmp; tmp=$(mktemp "$STAMP.XXXXXX")
+  grep -v "^$key " "$STAMP" > "$tmp" 2>/dev/null; mv "$tmp" "$STAMP"
```

### MIN-7 — rotation/truncation reads as STALLED (then gets masked)
Confirmed:

```
--- log rotation / truncation:
  rotated+growing log -> STALLED
```

`grew()` only tests `now -gt prev`, so any size *decrease* is `STALLED`. `report()`
then relabels `STALLED` to `quiet` when the process is alive, so a busy job whose
log was just rotated is reported as `quiet` rather than as a false alarm — the
honest-labelling logic accidentally covers this. Still wrong information. One
line:

```diff
   if [ -z "$prev" ]; then echo "first-look"
+  elif [ "$now" -lt "$prev" ]; then echo "rotated"
   elif [ "$now" -gt "$prev" ]; then echo "+$((now - prev))B"
   else echo "STALLED"; fi
```

### MIN-8 and small stuff
* `report m5-serve … "$RUNS/m5-serve/results/live/serve.log"` hardcodes the tag
  `live`; an `off` or `dryrun` evidence run reports `no-log`.
* `FREE=$(df … | tr -dc 0-9)` yields an empty string if `df` fails, and
  `[ "$FREE" -lt 180 ]` then errors to stderr. `${FREE:-0}` fixes it.
* `pgrep -fc "vllm.*api_server"` counts the API server *and* every TP worker, so
  `alive=5` is normal for TP4 — fine, but worth a word in the header so nobody
  reads it as five serves.
* The `[.]` bracket idiom in the patterns is `grep` habit, not needed for
  `pgrep -f` (which never matches itself), and harmless.

---

## What I actually changed

Only `runs/m5-serve/*.py`. Each change has a test that fails without it.

| File | Change |
|---|---|
| `swap_evidence.py` | stable blake2b prompt seed + per-draw distinct prefix (MAJ-1); exemplar/timestamp/whitespace-tolerant regex + unparsable-line counter (MIN-1); non-finite values dropped and reported (MAJ-3); new `series()` exact-name matcher (MAJ-4); monotonic-clock rate + counter-reset guard (MAJ-5); per-phase results list + single clock read (MAJ-2); `fq_present` per sample; `main()` exits 3 when nothing succeeded; dead `phases` param removed |
| `make_charts.py` | `finite()` predicate wired into `nice_ticks`/`fit`/`line`/`frame`/`occ_at` (MAJ-3); `fit` no longer inverts the axis on negatives; phase bands clamped to the plot rect (MIN-4); `render()` refuses a timeline with no throughput sample |
| `test_swap_evidence.py` | +9 tests |
| `test_make_charts.py` | +8 tests |

Verification:

```
$ pytest test_swap_evidence.py test_make_charts.py -q
39 passed in 3.72s
$ for s in 0 1 7 99; do PYTHONHASHSEED=$s pytest -q …; done
39 passed  (x4)          # the flaky corpus test is now deterministic

# same tests against HEAD's swap_evidence.py / make_charts.py:
17 failed, 22 passed     # every new test fails without its fix
```

Not changed, deliberately: the shell scripts (proposed as diffs above), anything
under `/home/mbelleau/src/gg-vllm` (live uncommitted work — comment-only diff
proposed), `publish_window.py` and `sweep.sh` (outside the m5-serve scope I was
given; fixes proposed inline).

## Recommended order of work before the PR

1. CRIT-1 (`serve-glm52.sh` fails closed on a missing pubkey) — a security
   property that is currently one `|| true` away from off.
2. CRIT-2 (`setsid` + process-group kill) — protects the box, and the campaign
   sharing it.
3. CRIT-4 then CRIT-3 (`eval_gpqa.sh`, then `set -e` + deliverable assertions) —
   without these the campaign can report success having measured no quality and
   drawn no chart.
4. MAJ-7 (`return 0` on nothing-to-publish) — unblocks the disk prune.
5. MAJ-6 (per-arm artifact dir) before any A/B is recorded.
6. MAJ-10 (decide the bottom panel's scale) by rendering the real timeline once
   and looking at it.

# Adversarial review — the heatmap feature

**Date** 2026-08-11 · **Scope** `heatmap.py` (endpoint), `heatmap.html` (page),
`make_axis_panels.py` (flagship SVG), their tests, `DESIGN.md`,
`ENDPOINT-SPEC.md`, `README.md`, `CAPTURE-RECIPE.md`.

**Premise of the review:** this is destined to be a flagship image in an
upstream PR, so the worst outcome is not a crash — it is a picture that renders
beautifully and states something the data does not say. Everything below was
hunted with that bias, and every claim here is backed by a measurement against
the real dumps in `../results/k3-fq/`, not by reading.

**Verdict: DEFECTS_FOUND.** Three "pretty and wrong" defects in the page, one
in `DESIGN.md`'s measurements, two minor issues in the endpoint. All six are
fixed. One further defect — the colour of a never-routed cell — is *reported
and not fixed*, because fixing it is a palette decision and nobody on this box
can see the page render.

The two headline attacks, cross-rank double counting and normalisation
dishonesty, **survived**. See §1 and §2.

---

## Test suites — verbatim summary lines

```
$ cd /home/mbelleau/src/gg-vllm && CUDA_VISIBLE_DEVICES="" gg-run.sh \
      python -m pytest tests/exl3_fungible/ -q --noconftest
493 passed, 10 skipped, 1 warning in 9.35s

$ cd runs/m5-serve/heatmap && python -m pytest test_heatmap_math.py -q
55 passed in 2.86s

$ cd runs/m5-serve/heatmap && python -m pytest test_make_axis_panels.py -q
44 passed in 0.16s

$ cd runs/m5-serve/heatmap && python -m pytest test_heatmap_math.py test_make_axis_panels.py -q
99 passed in 2.92s
```

Baseline before this review: `491 passed, 10 skipped` / `49 passed` /
`44 passed`. This review adds **2** endpoint tests and **6** page tests. Both
optional JS dependencies are installed, so nothing was silently skipped:
`quickjs` and `esprima` both import, and `pytest --collect-only` reports 55
collected with 0 skips.

---

## 1. Cross-rank double counting — **NOT PRESENT**

Under TP4 the gate is replicated, so four ranks are four copies of one number,
not four quarters of it. Summing would inflate every cell 4×.

Traced end to end rather than taken on trust:

| Link | Evidence |
|---|---|
| The collector is sized to the **global** expert count, identical on every rank | `integration.py:157` — `routers[0][1].global_num_experts` |
| It histograms the **gate's** `topk_ids`, i.e. the replicated decision | `stats.py:224-227`, `stats.py:241-244` |
| The endpoint declares the rule and does not sum | `heatmap.py` `MERGE_RULE = "rank0-canonical"`, `_merge()` takes one rank's arrays |
| Ranks are proved to be replicas of the **real** `FqStatsCollector`, then the endpoint over four of them is proved to return 1× | `test_heatmap_cpu.py::test_tp_ranks_are_replicas_so_the_merge_must_not_sum_the_real_collector` |
| Divergence is surfaced, not papered over | digest per rank, `ranks.agree`, a caption warning, rate-limited ERROR log |
| The rule is printed **on the figure**, so an exported PNG answers "is this 4× too big?" | `heatmap.html:titleLines` → `ranks N agree=… merge rank0-canonical` |
| DP > 1 is refused rather than reporting one replica as the whole serve | `501 dp_not_supported` |

One real gap was found here and fixed — see **F-06**.

## 2. Normalisation dishonesty — **NOT PRESENT**

The plotted value is `log2( E · count(l,e) / Σₑ count(l,e) )` on a **fixed**
±4 domain that is never derived from the data. Measured, not asserted:

* Scaling a real dump by **71×** (the actual spread across the archived runs)
  changes the magnitude panel by **exactly 0.0** in every one of 19,200 cells.
* The compare panel between a run and the same run ×71 is
  **max |Δ| = 2.8e-14** — i.e. compare mode is *not* showing "run B was longer".
* Two dumps 71× apart share `lo,hi = (-4.0, 4.0)`
  (`test_domain_is_fixed_never_autoscaled`).
* `delta` conserves within every layer (`max|Σₑ Δ| < 1e-12`), so the panel reads
  literally as traffic moving between experts. `symlog` and `logratio` do not
  conserve, are offered anyway, and both raise a banner and a caption line
  saying so.

The one auto-scaled element in the figure is the **right-hand gutter's second
bar** (`max×` for a magnitude panel, `moved` for a compare panel). Its bounds
are printed underneath it, so it is labelled rather than hidden, but two
exported figures' gutters are not comparable to each other. Reported below as
**N-05**, not fixed.

## 3. mass vs count — one real hole, fixed

The server side is airtight: `mass_b64` is populated only when
`mass_is_real and "mass" in include`, and the aliased case ships `mass: null`
plus an explicit warning. `mass_is_real` is always read from the flag, never
inferred from `count == mass` — which matters, because in **every archived
dump `count == mass` exactly** and the flag is **absent** (verified: all four
files, no `mass_is_real` key).

The page drops an aliased `mass` array at parse time so nothing downstream —
above all the CSV export, which leaves the page and loses every on-screen
caveat — can publish count under a column named `mass`. `updateMassGate`
disables the control and forces `metric=count`.

**But `updateMassGate` only inspects the *current* frame**, and a figure can be
built entirely out of snapshots. That is **F-03**, and it is the single worst
thing found in this review: a compare panel differencing real gate mass against
a count fallback. Fixed.

## 4. Colour correctness — **CORRECT**, with one measured exception

| Requirement | Measured | Verdict |
|---|---|---|
| Sequential is one hue, no rainbow | ColorBrewer Purples-9, L\* 98.7 → 18.1 strictly monotone | PASS |
| Diverging has a neutral gray midpoint | `RAMP_DIV[3] = #C1C1C1`, Lab a\* = −0.0, b\* = 0.0 | PASS |
| …and zero lands on it exactly | `LUT_N = 257` (odd) ⇒ `lutIndex(0,−D,+D) = 128` ⇒ `#C1C1C1`, for D ∈ {1,2,4,8} | PASS |
| Diverging is lightness-symmetric | L\* 38.9 / 51.9 / 66.0 / **78.1** / 66.2 / 51.9 / 37.9 | PASS |
| Tier ramp is ordinal, evenly spaced | L\* 90.1 / 72.0 / 53.9 / 36.0 — steps 18.1 / 18.0 / 17.9 | PASS |
| Scale bounds always displayed | mag `1/16× … 16×`; mismatch `−1 / 0 agrees / +1`; compare `−D … +D`; tier K2–K5 with "· absent" marks | PASS |
| No colour defined only inside a media query | every token in `@media (prefers-color-scheme: dark)` and in `:root[data-theme="dark"]` is also defined on bare `:root`; `body` carries an explicit `background:var(--bg)` | PASS |
| Ramps interpolated in Lab, not sRGB | JS Lab LUT matches an independent Python Lab LUT on all 257×2 stops | PASS |

**The exception is the "never routed" colour** — `#FFFFFF`, ΔE76 **1.66** from
the palest ramp swatch and ΔE **0** from the exported figure's ground. That is
**F-04**, reported and not fixed.

## 5. The synthetic watermark — **HOLDS**

Attacked directly. `--allow-synthetic` output carries the string 3× (solid
orange banner + repeated diagonals), every synthetic panel is hatched (210
`<line class="hatch">`) and tagged `— SYNTHETIC PLACEHOLDER, NOT MEASURED`, the
file is renamed `*.SYNTHETIC.svg`, the JSON sidecar is renamed too and carries
`"synthetic": true`, and stdout leads with `*** SYNTHETIC LAYOUT PREVIEW ***`.

Attempts that all failed to produce an unmarked placeholder:

* every CLI flag and flag combination (`--order`, `--signal`, `--all-intervals`,
  `--title ""`, `--subtitle ""`, `--cell 0`, `--cell 1`) — still 3–5 marks;
* `--no-watermark` and four other guessed names — `SystemExit`, and the strings
  are asserted absent from the source;
* blanking the `WATERMARK` constant — the writer's guard compares against a
  **literal**, so it fails closed;
* stubbing `_watermark()` to return `[]` — the banner alone still marks it, and
  stubbing `render()` too trips the guard.

The only way through is to set `Panel.synthetic = False` by hand in Python,
i.e. to deliberately write false provenance — not a defect in the guard.

One cosmetic weakness found: **N-04**, the banner text overflows its highlight
rect on narrow figures.

## 6. Offline mode vs a plain Python read of the same file

Loaded every archived dump into the page's own code (whole script, stub DOM,
QuickJS) and compared the numbers it *displays* against NumPy:

| dump | cells | max abs diff, `share × uniform`, cell-for-cell | Σ count (page) | Σ count (NumPy) | dead / clip-lo / clip-hi (page = NumPy) |
|---|---|---|---|---|---|
| `stats-code-axis.jsonl` | 19 200 | 1.78e-14 | 69 990 461.170763 | 69 990 461.170763 | 0 / 752 / 25 |
| `stats.jsonl` | 19 200 | 8.88e-15 | 505 302 611.043496 | 505 302 611.043497 | 0 / 23 / 0 |
| `stats-synthetic.jsonl` | 19 200 | 2.84e-14 | 7 164 062.024370 | 7 164 062.024370 | 42 / 1082 / 5 |
| `stats-INVALID-truncated-corpus.jsonl` | 19 200 | 3.20e-14 | 71 587 110.063419 | 71 587 110.063419 | 1 / 767 / 20 |

Also cross-checked and matching: per-layer entropy, `max × uniform`, the
row-sum invariant, the `aria-label` summary, the layer table and the CSV export
(`mass_is_real` travels as a **column**, correctly `unknown` for the archived
dumps). The page's arithmetic is right.

---

## Findings

### F-01 · MAJOR · `heatmap.html` · **FIXED**
### The column-mean marginal strip is destroyed by a single never-routed cell

`derive()` stores `-Infinity` for a dead cell (it is drawn *off* the ramp).
`drawPanel` averaged `pv.vals` straight, so **one** dead layer turned the whole
column's mean into `-Infinity`, and `lutIndex` clamps that to index 0 — the
palest swatch on the ramp.

**Failure scenario (measured, real data).** `stats-code-axis.jsonl` record 0
has three dead cells, in columns E65 / E67 / E155. Their true floored column
means are **−0.71 / −0.86 / −0.68**; E155 peaks at **3.5× uniform** in one
layer. All three rendered as `rgb(252,251,253)` — the *coldest* columns on a
figure whose marginal strip exists to say which experts are hot overall.
Marginals are ON by default.

**Fix.** New pure `colMeans(vals, L, E, lo)` floors dead cells at the panel's
own domain edge — exactly what `derive` already does to every other sub-1/16×
value and what `make_axis_panels.to_scale` does for the same figure.
Post-fix those columns render `rgb(180,179,214)` / `rgb(183,183,217)` /
`rgb(179,178,214)`.

**Tests.** `test_column_mean_floors_dead_cells_at_the_domain_edge` (minimal),
`test_column_mean_marginal_survives_a_single_never_routed_cell` (real dump vs
NumPy), and `test_rendered_marginal_strip_is_not_flattened_by_a_dead_cell` —
the end-to-end one, which renders the page and reads back the fill colours the
strip actually received. Reverting only the `drawPanel` call site fails it with
`assert 'rgb(252,251,253)' != 'rgb(252,251,253)'`.

### F-02 · MAJOR · `heatmap.html` · **FIXED**
### Log-ratio compare paints an on/off switch as "no change"

`panelValues` computed
`x = (da.rel>0 && db.rel>0) ? log2(db/da) : 0`. Zero is the **neutral-gray
midpoint** of the diverging ramp — "nothing happened". So an expert that went
from dead to hot, or hot to dead, was painted as the one thing it certainly is
not.

This directly contradicts the page's own guidance: selecting log-ratio raises a
banner reading *"Use it only for 'which experts switched on or off'."*

**Failure scenario (measured, real data).** Comparing
`stats-code-axis.jsonl` record 0 → `stats-synthetic.jsonl` last record, **39
cells** have exactly one side dead. Every one of them mapped to LUT index 128,
`#C1C1C1`.

**Fix.** Saturate: `+Infinity` when B switched on, `-Infinity` when B switched
off (`lutIndex` clamps both onto the ends of the ramp, which is the honest
reading of an infinite ratio); dead on *both* sides stays 0, which really is no
change. Finite cells are bit-identical to before.

**Test.** `test_log_ratio_saturates_an_expert_that_switched_on_or_off` — asserts
switched-on cells hit `LUT_N-1`, switched-off cells hit `0`, neither hits the
neutral index, both-dead stays 0, and the finite cells are unchanged to 1e-12.

### F-03 · MAJOR · `heatmap.html` · **FIXED**
### The compare panel silently differenced gate mass against count

`updateMassGate` inspects only `currentFrame()`. A figure built from snapshots
can therefore hold frames with different mass availability, and `derive()`
falls back to `count` *silently* for any frame whose aliased mass was dropped.
Nothing checked that the two sides of a compare agreed on which array they were
reading — the panel refused a shape mismatch but not a metric mismatch.

**Failure scenario.** Serve booted with `VLLM_FQ_GATE_MASS=1`; the live frame
has real mass, so the `mass` control stays enabled. Slot A = a snapshot with
real mass. Slot B = a snapshot of an archived dump whose mass was aliased and
dropped. `cmpA=A, cmpB=B`, metric `mass`. `currentFrame()` is the live frame
and passes the gate, so nothing intervenes. Reproduced on a fixture whose two
frames have **identical counts**: the honest answer is a perfectly neutral
field, and the panel instead produced **19 197 of 19 200 non-zero cells, max
|Δ| = 8.27** — a fully saturated diverging panel depicting traffic that did not
move. Stack mode had the same hole, with the added twist that the figure header
describes `panels[0]` only.

**Fix.** New pure `metricOf(frame)` returns the array `derive()` will actually
read. `buildPanels` now refuses a compare whose two sides disagree, and drops a
stacked magnitude panel that disagrees with `panels[0]`, in both cases naming
it in `PANEL_DROPS` — which the caption already prints. The caption line was
generalised from "shape differs" to "cannot share this figure's basis
(L×E cells, metric M)".

**Tests.** `test_compare_refuses_to_difference_gate_mass_against_count` (with a
control proving the panel *is* drawn, and differences to exactly 0.0, when both
sides agree) and
`test_stacked_panels_that_fall_back_to_count_are_dropped_not_mislabelled`.

### F-04 · MAJOR · `heatmap.html` + `DESIGN.md` · **REPORTED, NOT FIXED**
### A never-routed cell is invisible

`DEAD_RGB = [255,255,255]`, specified by `DESIGN.md` §5.2 as "the pure
background colour `#FFFFFF` so 'never routed' is not confused with 'rarely
routed'". Measured, it achieves the opposite:

* `#FFFFFF` vs the palest ramp stop `#FCFBFD`: **ΔE76 = 1.66**, below the
  ~2.3 JND. "Never routed" and "rarely routed" are the same colour.
* `#FFFFFF` vs the light figure ground and the **PNG export ground** (both
  `#FFFFFF`): **ΔE = 0**. In the exported flagship a dead cell is a hole.
* It is also indistinguishable from the **752 / 23 / 1082 / 767 cells clipped
  at the bottom** of the domain (see F-05), which are painted `#FCFBFD`. The
  pale end of the figure conflates *dead*, *clipped-low* and *merely cold*.

`make_axis_panels.py` — the *other* renderer of this same design — explicitly
rejects this choice in a code comment and uses a neutral gray (`#b0b0aa` light
/ `#6e6e66` dark, ΔE 27.4 from `#FCFBFD`). The two implementations disagree.

**Why not fixed here.** It is a palette change with light/dark implications and
no one on this box can see the page render; picking a swatch blind is how the
next reviewer inherits a different wrong colour. `DESIGN.md` §5.2 now carries
the measurement and an OPEN marker, and `README.md` warns to tick **flag dead
cells** (crimson) before exporting a figure that has any. Recommended
resolution: adopt `make_axis_panels.py`'s neutral gray in both renderers, or
default `flagDead` on for `exportLight`.

### F-05 · MAJOR · `DESIGN.md` · **FIXED**
### "Clips essentially nothing" counted only the upper clip

§5.2 claimed the ±4 domain "covers 99.87% / 100.00% / 99.97% / 99.90% of
cells". Those are the **high-side** clips alone. Counting both ends:

| dump | clipped low | clipped high | real coverage | claimed |
|---|---|---|---|---|
| `stats-code-axis.jsonl` | 752 (3.92%) | 25 (0.130%) | **95.95%** | 99.87% |
| `stats.jsonl` | 23 (0.12%) | 0 | **99.88%** | 100.00% |
| `stats-synthetic.jsonl` | 1082 (5.64%) | 5 (0.026%) | **94.34%** | 99.97% |
| `stats-INVALID-truncated-corpus.jsonl` | 767 (3.99%) | 20 (0.104%) | **95.90%** | 99.90% |

Up to **5.6% of the flagship's cells are saturated**, all at the pale end,
where they are also within ΔE 1.66 of "never routed" (F-04). A PR reviewer
reading "clips essentially nothing" would not go looking for that.

`heatmap.html` itself is honest — it prints both counts separately in the
caption and both in the `aria-label` — so this is a documentation defect only.
Both occurrences in `DESIGN.md` (§5.2 and the appendix summary) are corrected
with the measured numbers and marked as corrections.

### F-06 · MINOR · `heatmap.py` · **FIXED**
### `?reduce=all` could serve a non-rank-0 array under a `rank0-canonical` label

`_merge` took `next(r for r in results if r.get("canonical"))`. With
`reduce=all` **every** rank sets `canonical: True` (each ships its arrays), so
that expression is "the first entry in the list", and `_rank_results` preserves
whatever order it is handed.

**Failure scenario.** An executor that returns worker results in any order
other than rank order puts, say, rank 3's arrays in the envelope's top-level
`count` while `merge_rule` says `rank0-canonical`. With `reduce=rank0` the
behaviour is correct (only rank 0 is canonical), so the bug is invisible in the
default path and appears only in the debug path an operator reaches *because*
they already suspect the ranks disagree.

**Fix.** Prefer rank 0 among canonical results, falling back to the previous
behaviour. New helper `_as_rank`.

**Test.** `test_reduce_all_still_takes_rank_0_when_results_arrive_out_of_order`
— a `FakeEngineClient` that reverses the result list, with rank 2 perturbed so
the arrays are distinguishable. Reverting the fix fails it with
`canonical rank 3`.

### F-07 · MINOR · `heatmap.py` · **FIXED (documentation + pinning test)**
### `_require_gates` claimed to defend the token; it does not

The docstring said the worker-side check stops `POST /collective_rpc` from
"bypassing `VLLM_FQ_HEATMAP` **and the token** entirely". It re-checks the env
gates only. The token is an HTTP header validated at the router and nothing of
the HTTP request reaches `collective_rpc`.

**Failure scenario.** `VLLM_SERVER_DEV_MODE=1` + `VLLM_FQ_HEATMAP=1` +
`VLLM_FQ_HEATMAP_TOKEN=…` set. `GET /fq/heatmap` without the header → 403.
`POST /collective_rpc {"method":"fq_heatmap_sample","args":["{\"op\":\"sample\"}"]}`
→ **the full 19 200-cell matrix**, no token.

**Why the fix is documentation, not enforcement.** `/collective_rpc` forwards
*any* worker method, so anyone who can reach it already has strictly more reach
than the heatmap token grants; bolting a token into the RPC args would move the
secret into the RPC payload without closing anything. The honest statement is
that the token protects the heatmap **route**, and `/collective_rpc` must not
be exposed to anyone who must not read the matrix. The destructive scope is
*not* in this position: `op=zero_collector` has its own env gate, which this
path does honour, and the test pins that too.

**Test.** `test_the_worker_gate_is_env_only_the_token_does_not_reach_it` — a
behaviour pin, not a red-then-green test; it fails if someone changes the
enforcement without updating the docstring and the spec together.

---

## Notes — real but not worth a fix in this pass

* **N-01 · `heatmap.py` · one `_CUM` per process, keyed by nothing.** Two
  clients polling `?include=cum` with *different* `layers` selectors make
  `integrate()` see a changed row axis and rebase to zero on every alternate
  poll, so both see a cumulative that keeps restarting. `cum_since_step`
  reports it honestly, so it is confusing rather than wrong. Keying the
  accumulator by the layer tuple would fix it.
* **N-02 · `heatmap.html` · `rowSumOk` is vacuously true when layer 0 is
  empty.** `if(rowSum[0]>0)` guards the whole loop, so `dev` stays 0 and the
  caption prints "< 1e-9 — per-layer share equals per-run share" even if other
  layers disagree. Reachable only in the first seconds of a serve.
* **N-03 · `heatmap.html` · the offline path does not validate the tier
  domain.** The endpoint rejects a tier outside `occupancy_table.TIERS =
  (2,3,4,5)` (`encode_tier`); a raw `VLLM_FQ_DUMP_STATS` record does not go
  through that, and an out-of-domain K renders as an unlabelled
  `rgb(170,170,170)` with no legend key. The panel label does list it in
  `K{…}`.
* **N-04 · `make_axis_panels.py` · the synthetic banner overflows on narrow
  figures.** At `--cell 1` the banner text is ~605 px inside a `pw = 256` px
  orange rect; the 349 px overflow is `.warntx` `#ffffff` on the `#fcfcfb`
  light surface, i.e. invisible. The headline `SYNTHETIC LAYOUT PREVIEW`
  itself (~145 px) still sits inside the rect, and the diagonal marks and the
  hatch are unaffected, so the figure stays marked. Sizing the rect to the text
  rather than to the panel width would fix it.
* **N-05 · `heatmap.html` · the gutter's second bar is auto-scaled per
  figure.** `max×` uses `[1, max(2, max maxRel)]` and `moved` uses
  `[0, movedMax]`, so gutter bars are not comparable between two exported
  figures. Both ranges are printed beneath the bar, so it is labelled, not
  hidden. The main colour scales are correctly fixed.
* **N-06 · `heatmap.html` · `band9()` was dead code with a wrong comment**
  (it claimed the legend used it; the legend goes through the 257-entry LUT).
  Comment corrected; the function is kept because the maths test buckets cells
  with it to compare against `DESIGN.md` §5.2's measured table.
* **N-07 · the plotted magnitude is `Float32Array`.** Agreement with a float64
  reference is ~1e-8, five orders below one LUT step (8/256 = 0.031). Harmless,
  but it is why the new column-mean test uses a 1e-6 tolerance and not 1e-12.

---

## What is still NOT verified

* **No human, and no rasteriser, has seen this page render.** There is no
  browser on this box. Everything here is arithmetic, colour maths done in Lab
  against an independent implementation, and pixel bytes read back out of a
  stubbed `ImageData`. **Layout is unverified**: label/ramp collisions, text
  overflow, wrapping, the legend's hardcoded `x+312` offsets against the actual
  rendered width of its labels, and whether the 3 px cells read at all on a
  real display. F-04 in particular is a *measured* colour-difference claim, not
  an observation of the rendered page.
* **The live endpoint was not exercised against the running serve.** GPUs 0–3
  are serving and GPUs 4–7 are encoding; every `/fq/heatmap` sample stalls the
  engine step loop, so no HTTP request was issued to it. The endpoint is
  exercised only through FastAPI's `TestClient` against fake workers running
  the **real** `worker_sample`, plus real `FqStatsCollector` instances for the
  cross-rank test.
* **The cross-rank merge is proved for TP, not measured on this TP4 serve.**
  The proof is: replicated-gate reasoning traced through `integration.py` and
  `stats.py`, plus four real collectors driven with identical batches. Nobody
  has diffed rank 0 against rank 3 on the live serve.
* **No CUDA-graph capture path was run.** All tests are CPU.
* **The flagship four-axis figure is still a placeholder.** Only
  `axis3_code_agentic` has a real dump; the other three panels in
  `axis-panels.SYNTHETIC.json/.svg` are fabricated, correctly watermarked, and
  every overlap number involving them is meaningless. See `CAPTURE-RECIPE.md`.
* **`mass` has never been observed real.** All four archived dumps have
  `count == mass` exactly and no `mass_is_real` key, so the real-gate-mass path
  is tested only against synthesised fixtures.
* **F-07 was not closed, only documented.** A serve that exposes
  `POST /collective_rpc` exposes the routing matrix regardless of the heatmap
  token.

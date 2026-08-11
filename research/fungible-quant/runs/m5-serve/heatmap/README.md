# FQ expert heatmap — operator page

`heatmap.html` is a single self-contained file: no build step, no CDN, no fonts,
no external image, zero network requests except the ones you point it at. Open
it from disk and it works.

It draws the **75 layer × 256 expert = 19,200-cell** activation matrix for
GLM-5.2 on a `<canvas>`, with the K-tier overlay, a derived tier↔heat mismatch
panel, and a diverging compare panel. The visual design is specified in
[`DESIGN.md`](./DESIGN.md); the wire format it consumes is specified in
[`ENDPOINT-SPEC.md`](./ENDPOINT-SPEC.md).

> **Status.** The maths, the render path and the pixels are tested (47 tests,
> below). **No human has seen this rendered in a browser** — there is no browser
> and no rasteriser on the build box. Layout collisions and typography are
> unverified; everything that decides *which colour a cell gets* is verified
> cell-by-cell against an independent NumPy implementation.

---

## Quick start

### Offline — works right now, no serve needed

1. Open `heatmap.html` in a browser (double-click; `file://` is fine).
2. Under **Offline — load a stats dump**, pick any file from
   `../results/k3-fq/*.jsonl`.
3. The page indexes the record boundaries (a few seconds for the 93 MB
   synthetic dump), then shows the **final** interval. Use the record selector
   to scrub through intervals.

All four archived dumps load: `stats-code-axis.jsonl`, `stats.jsonl`,
`stats-synthetic.jsonl`, and `stats-INVALID-truncated-corpus.jsonl` (shape only
— its contents come from a broken replay and mean nothing).

### Live — against a serve

1. Put the serve's base URL in **Endpoint** (default `http://127.0.0.1:8000`)
   and press **Connect**.
2. The page polls `GET {endpoint}/fq/heatmap` and revalidates with `If-None-Match`,
   so most polls cost a ~200-byte `304`.

Server side, `/fq/heatmap` needs its gates open:

```bash
export VLLM_SERVER_DEV_MODE=1          # serve-glm52.sh already sets this
export VLLM_FQ_HEATMAP=1               # the surface's own opt-in
export VLLM_FQ_HEATMAP_TOKEN=...       # optional; paste it into the token box
export VLLM_FQ_GATE_MASS=1             # optional; without it `mass` is not real
```

**CORS is the thing that will bite you.** A page opened as `file://` sends
`Origin: null`, and vLLM's `build_app` installs CORS middleware only when you
ask for it. If the fetch fails while the serve is plainly up, that is why — the
error banner says so. Start the serve with `--allowed-origins '*'` (dev only),
or serve the page from the serve's own origin.

**And a second, quieter CORS effect:** `ETag` is not a CORS-safelisted response
header, so cross-origin JavaScript cannot read it unless the serve sends
`Access-Control-Expose-Headers: ETag`. Without that the revalidation path is
silently off and every poll transfers the full ~33 KB sample instead of a
~200 byte `304`. That is costly, not broken — the page raises a warning banner
the first time it notices.

**The endpoint may not exist yet.** `heatmap.py` is specified in
`ENDPOINT-SPEC.md` but is a separate deliverable. Until it lands you will get
`404`. The page keeps retrying with backoff and stays usable in offline mode
meanwhile.

### What the page accepts on the wire

`parseSample()` handles **two** shapes, so it works against the real endpoint
and against anything simpler you bolt on:

| Shape | Arrays | Where it comes from |
|---|---|---|
| `fq-heatmap/1` envelope | base64 `bf16`/`f32`/`u8`, `layout: layer-major` | `ENDPOINT-SPEC.md` §3.4 |
| raw `VLLM_FQ_DUMP_STATS` record | nested JSON arrays `count` / `mass` / `tier_of` | `loop._dump_stats`, the archived dumps |

Either way it validates before it draws: every array must decode to exactly
`num_layers × num_experts` cells, and `tier_of` must have exactly as many rows
as `layers`. A short array is refused, never rendered as a heatmap shifted by a
cell.

---

## Controls

### Source

| Control | What it does |
|---|---|
| **Endpoint** | Base URL of the serve. Editable, persisted in `localStorage`. |
| **Token** | Sent as `X-FQ-Heatmap-Token`. |
| **Connect** | Drops the cached `ETag`, resets the failure counter, probes `/fq/heatmap/meta`, fetches. |
| **Poll** | 1 / 2 / 5 / 10 s, or off. The decayed matrix only changes on a window roll (~15 s at the observed cadence), so 2 s already oversamples it. |
| **Pause** | Stops polling. The current frame stays on screen. |
| **Fetch once** | One request, ignoring the timer. |
| **Offline file** | Loads a `.jsonl` dump. Indexes line offsets by streaming the Blob and only parses the selected record, so a 93 MB dump does not have to fit in a JS string. |
| **record** | Scrub through the intervals in the loaded dump. |
| **back to live** | Drops the file and returns to the live sample. |

### View

| Control | What it does |
|---|---|
| **Metric: count \| mass** | `mass` is **disabled and labelled "not recorded"** when the sample reports `mass_is_real: false`. When the field is *absent* (all four archived dumps) it is selectable but labelled **"unknown — may alias count"**, and the figure header repeats the warning. It never shows `count` under a label that says `mass`. |
| **Column order: native \| sorted** | Native expert id (the live default — operators look for a specific expert) or sorted by heat (the flagship default — readers look for a pattern). |
| **ref:** | Which frame defines the sort. `pooled` = mean over the panels in the figure; or a named slot. **One** permutation is computed and applied to every panel; its hash is printed on the figure. |
| **Cell** | `fit` picks the largest integer cell that fits the column; 2–12 px are explicit. Always an integer — see "Why integer cells" below. |
| **K-tier strip** | The tier matrix, same column order, pixel-aligned under the magnitude panel. |
| **mismatch** | The derived `rank(heat) − rank(tier)` panel. |
| **compare** | The diverging A→B panel. |
| **stack slots** | Draw every filled snapshot as its own magnitude panel — this is the flagship layout. |
| **marginals + gutter** | Column-mean strip above each magnitude panel; per-layer entropy and max-share bars in the right gutter (Spearman ρ and traffic-moved for a compare panel). |
| **flag dead cells** | Never-routed cells are drawn pure white by default; this makes them crimson so you can find them. |
| **tier strip in native order** | Draws the tier strip in expert-id order even when the magnitude panels are sorted. Use it when tiers are heat-derived, where a sorted tier strip degenerates into a trivial staircase (`DESIGN.md` §2.5). |

### Snapshots and compare

| Control | What it does |
|---|---|
| **RESET — mark baseline** | Snapshots the current frame into slot **A** and switches the compare panel on as `A → live`. **Client-side only: the server is not touched.** |
| **mark / ×** on slots A–D | Capture or clear a named snapshot. |
| **Compare A → B** | Any two of `live`, `A`, `B`, `C`, `D`. |
| **compare metric** | `Δ share` (default, conserving), `symlog Δ` (boosts weak differences, warned), `log2 ratio` (warned). |
| **domain** | ±1 / ±2 / ±4 / ±8. Always printed on the legend. Changing it is a *stated* choice, not auto-scaling. |

Why RESET is client-side, per `ENDPOINT-SPEC.md` §7.2: the same counters feed
the policy loop, the `fq_jaccard` gauge and every other viewer, so a
destructive reset by one tab silently rewrites what everyone else sees. A
client baseline also lets you hold four of them at once and diff any pair with
no round trip.

**It is "change since mark", not "traffic since reset".** `count` is a
λ-decayed window, so the difference can be negative (an expert that cooled
off), and once ~640 engine steps have passed the baseline has decayed out of
the current window and the difference converges back to the plain current
value. The page says this in a banner when you press RESET.

A server-side reset does exist, under **Advanced**, behind an explicit
"I understand" checkbox and a confirm dialog: `scope: "heatmap"` (harmless —
rebases the endpoint's own cumulative accumulator) and `scope: "collector"`
(destructive — the card spells out exactly what it does to the policy loop).
Neither is wired to any default control.

### Export

| Control | Output |
|---|---|
| **PNG** | The whole figure — title, provenance, panels, gutters, legends, caption — as it is on screen, re-rendered on a **light ground** so the image is stable regardless of your theme. |
| **CSV** | `layer,expert,count,mass,tier,share_x_uniform,log2_rel` for every cell of the current frame. This is the table-view twin of the canvas. |
| **permutation** | The exact column permutation as JSON, with its hash and the reference that produced it. Commit this next to any sorted figure. |

Hovering a cell gives the exact record — layer id, expert id, count, mass,
share ×uniform, log2, rank in layer, K tier, and (in compare) both sides and
the delta. Clicking pins the layer into the **Layer detail** table below.

---

## Producing the 4-axis comparison image

This is `DESIGN.md` §6.1: one panel per corpus, one shared permutation, one
tier strip, one compare panel, one caption.

1. Open the page. You do not need a serve.
2. Load `../results/k3-fq/stats-code-axis.jsonl` → let it settle on the last
   record → **mark** slot **A**.
3. Load `../results/k3-fq/stats.jsonl` → **mark** slot **B**.
4. Load `../results/k3-fq/stats-synthetic.jsonl` → **mark** slot **C**.
5. Load `../results/k3-fq/stats-INVALID-truncated-corpus.jsonl` → **mark**
   slot **D**. *(Label it as invalid in the caption — its contents are
   meaningless; it is here as the fourth axis of the shape comparison.)*
6. Tick **stack slots**, **K-tier strip**, **compare**. Set **Compare** to
   `A → C` (code-axis vs synthetic is the interesting pair; `A → B` is
   code-axis vs full corpus).
7. Set **Column order** to `sorted` with **ref: pooled**.
8. Set **Cell** to `4 px`.
9. Press **PNG**, then press **permutation** and commit the JSON next to the
   image.

The caption the page renders onto the figure already carries the
normalisation, the row-sum invariant, the clipped-cell counts, the permutation
hash and its reference, the mass state, the conservation check and the
per-layer Spearman/overlap summary. **Do not crop it off** — a heatmap without
its scale and its normalisation is decoration.

One sentence you must keep in any prose around the image, because the picture
will mislead without it:

> The dark→light gradient in the sorted panels **is the sort**, not a finding.

---

## What the numbers mean

**Normalisation.** `value(l,e) = log2( E · count(l,e) / Σₑ count(l,e) )` — the
expert's share of its own layer, in multiples of uniform routing (`1/E`), log2,
clipped to **±4** (1/16× to 16×). Run length cancels exactly, which matters: the
four archived corpora differ by **71×** in total assignments, so any raw-count
rendering would be a rendering of transcript length. The domain is **fixed** and
never auto-scaled — that is what makes two frames comparable, and it is why the
picture right after a reset is correctly pale rather than falsely confident.

**Per-layer == per-run.** Top-k routing puts the same total through every
instrumented layer, so per-layer share and per-run share are the same picture up
to a constant. The page **asserts** this at load (`max|Σₑ − Σₑ[0]|/Σₑ[0] < 1e-9`,
measured 1.2e-16 on the real dumps) and says `VIOLATED` on the figure if a shared
expert or a varying `top_k` ever breaks it.

**Clipping is reported at both edges.** `DESIGN.md` quotes 99.87 % domain
coverage; that figure counts only the *upper* tail. Measured on the code-axis
dump the real split is **752 cells (3.92 %) below 1/16×** against **25 (0.13 %)
above 16×** — the cold tail is 30× the hot one, and the page prints both numbers
in the caption. `test_clipping_is_counted_at_both_edges` pins this.

**`count` is not a token count.** It is a λ-decayed sum over a bounded ring
(`window_len=64`, `stride=32`, `decay=0.95`), returned as float64. Values are
floats, the horizon is ~640 engine steps, and traffic older than that is already
gone. Nothing in the UI formats it as an integer or calls it "tokens".

**`mass_is_real` is a flag, never an inference.** A uniform router legitimately
produces `count == mass`, so comparing the arrays cannot tell you anything. All
four archived dumps predate the field entirely, so the page renders them as
`unknown` — **absent ≠ false**.

**Layer ids are data.** The row axis comes from the sample's `layers` array,
never from `row + 3`. Policy files carry **76** layer keys (3–78, including the
MTP layer) while stats carry **75** (3–77), so a positional join is off by one at
the bottom; the page takes `tier_of` from the sample and refuses a sample where
`len(tier_of) != len(layers)`.

**Why integer cells.** Fractional cell widths make the browser resample, and
resampling a heatmap averages neighbouring experts into colours that correspond
to no data point. The matrix is written into a 256×75 `ImageData` at one pixel
per cell and blitted with `imageSmoothingEnabled = false` at an integer scale;
`test_blit_is_an_exact_integer_scale_with_smoothing_off` checks every cell size.

**Colour.** Magnitude is ColorBrewer Purples-9 — one hue, light→dark, L\*
monotone, so it degrades to a valid grayscale ramp under any CVD and under
photocopying. Tier is a single-hue green at even 18-unit L\* spacing, mapped by K
value and never reassigned by frequency. Compare and mismatch use a
blue↔**neutral gray**↔orange diverging ramp whose midpoint is a true neutral
(L\* 78, a\* = b\* = 0) and whose L\* is symmetric and monotone outward, so there
is no bright halo inventing a feature at zero. All three are interpolated in
CIELAB, not sRGB, into 257-entry LUTs — odd, so that zero lands *exactly* on the
neutral swatch.

---

## Tests

```bash
cd research/fungible-quant/runs/m5-serve/heatmap
/home/mbelleau/venvs/fq/bin/python -m pytest test_heatmap_math.py -q   # 49 passed
/home/mbelleau/venvs/fq/bin/python test_heatmap_math.py                # same, no pytest
```

Optional dependencies, each skipped individually with a reason when missing:

```bash
uv pip install --python /home/mbelleau/venvs/fq/bin/python esprima quickjs
```

What the suite actually does:

* **Structure** — the HTML parses and every tag balances; there is **no**
  external `src`/`href`/`@import`/`<link>`/`url()`; the theme tokens exist on
  bare `:root`, under `@media (prefers-color-scheme: dark)` **and** under
  `:root[data-theme="dark"]`; wide content scrolls in its own container.
* **Syntax** — all 64 KB of inline JavaScript is parsed by `esprima`, a real
  ECMAScript parser. (Consequence: the page avoids `?.`, `??` and bare
  `catch {}`, which esprima 4 cannot parse — no loss, they are also the ES2020
  features older browsers lack.)
* **Colour** — the ramps in the page match `DESIGN.md` literal-for-literal
  (a marked constants block is parsed out of the HTML, so a ramp cannot drift
  silently); L\* monotonicity, the true-neutral midpoint, even tier spacing, and
  `lut(0) == #C1C1C1` on every compare domain.
* **Cross-implementation** — the page's own arithmetic is sliced out of the HTML
  at the `==FQ-HEATMAP-PURE-END==` marker, executed in **QuickJS**, and compared
  against an independent NumPy implementation written from `DESIGN.md`:
  the 257-entry LUTs match entry for entry, and `derive()` matches on `rel`,
  `rowSum`, `entropy`, `maxRel`, `dead`, `clipLo` and `clipHi` for a real dump.
* **Colour buckets** — the colour index of **all 19,200 cells** of
  `stats-code-axis.jsonl`, pinned by a golden digest plus six hand-checked cells
  with their RGB. The nine-band histogram reproduces `DESIGN.md` §5.2 exactly:
  `[2028, 2739, 3995, 4802, 3501, 1470, 487, 153, 25]`.
* **Render path** — a stub canvas is installed and the **whole page script** is
  run in QuickJS. `render()` is exercised across seven panel configurations; the
  bytes written into the `ImageData` are read back and compared to the reference
  colours; the blit is asserted to be an exact integer scale with smoothing off;
  hover resolves to the right layer id and expert; PNG/CSV/permutation exports
  fire; and a failed poll is asserted to keep the last frame, dim it, show a
  CORS-aware error and grow its backoff.
* **Hazards** — `mass_is_real` absent loads as unknown; it is never inferred
  from array equality; a `tier_of`/`layers` length mismatch is refused; layer ids
  are taken from the data; a row-sum violation is surfaced; a truncated bf16 blob
  is refused rather than shifted; one permutation is shared across all panels and
  the reference demonstrably changes it.
* **Interop with the real endpoint** — the pure encoders are lifted out of
  `gg-vllm`'s `exl3_fungible/heatmap.py` by AST (no `import vllm`, so no built
  `vllm._C` is needed), an envelope is built with the **endpoint's own** bf16
  encoder, and the **page's own** parser decodes it. The two decoders agree
  bit-for-bit; worst error against the f32 source is **0.38868 %**, matching the
  0.38898 % the spec measured. On the colour ramp that quantum moves 1,276 of
  19,200 cells by **at most one** 257-step LUT index — which is the entire
  justification for bf16, checked rather than asserted. Skipped when `gg-vllm`
  is not in the tree.
* **Corroboration** — the page's own compare code recomputes `DESIGN.md` §5.4's
  measured statistics from scratch: Spearman median **0.366** (doc: 0.37),
  top-32 overlap median **0.281** (doc: 28 %), row conservation **4.4e-13**
  (doc: 7e-14).

### What is *not* tested

Anything only a browser can answer: font metrics and therefore text wrapping and
label collisions, real `devicePixelRatio`, actual scroll behaviour, the `File`
picker, `toDataURL` output, and whether the thing simply looks good. **Open it
and look before trusting any of the layout.**

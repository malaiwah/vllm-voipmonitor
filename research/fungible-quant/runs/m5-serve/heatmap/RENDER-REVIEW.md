# RENDER-REVIEW — first actual look at the heatmap page and the SVG figures

Until now nothing in this directory had ever been displayed. There was no browser
and no rasteriser on the box, so every visual claim in `DESIGN.md`, `README.md`
and `REVIEW.md` was unverified. This is the first viewing.

All PNGs referenced below are in `renders/`. Reproduce with `./render.sh`.

---

## 1. What was installed, and how

No root, no docker, no namespaces. Everything went into `$HOME`.

| Piece | How | Size |
|---|---|---|
| Playwright 1.62.0 + Chromium 151.0.7922.34 (headless shell) | `uv venv ~/venvs/render` then `uv pip install playwright` + `playwright install chromium` | 656 MB cache + 138 MB venv |
| 21 missing shared libraries | `apt-get download` + `dpkg-deb -x` into `~/.local/chromium-libs`, reached via `LD_LIBRARY_PATH` | 6.8 MB |
| DejaVu + Liberation fonts, and a fontconfig config | same, into `~/.local/share/fonts` + `~/.local/etc/fonts/fonts.conf` | 7.0 MB |

Two things were not obvious going in:

- **Chromium would not start.** `ldd` reported twelve missing `.so` files
  (`libnss3`, `libnspr4`, `libgbm1`, `libatk*`, `libxkbcommon0`, …). `apt-get`
  needs no root as long as every `Dir::` is redirected to a writable path, so
  the packages were downloaded and unpacked by hand. The system package lists
  were also stale enough to 404, so `apt-get update` was re-run against a
  private lists directory first.
- **The box has no fonts at all** — no `/usr/share/fonts`, no `/etc/fonts`, no
  `fc-list`. Chromium starts without them but draws no text, which would have
  produced screenshots that silently omitted every label, title and legend.
  DejaVu was installed and aliased to `sans-serif` / `ui-sans-serif` /
  `system-ui` / `monospace` / `ui-monospace` so the page's font stacks resolve.

`./render.sh --bootstrap` redoes all of the above from scratch.
Total footprint ≈ 810 MB, of which 12 MB is the committed `renders/`.

## 2. What was rendered

`heatmap.html` was driven through its real OFFLINE path — Playwright pushes the
real dump into the `<input type=file>`, the page indexes the 15 MB JSONL, and
record 18 (the final interval) is displayed. No mocking, no synthetic data.

30 PNGs: default view, count↔mass toggle, mismatch panel, dead-cell flag on/off,
COMPARE (axis1 vs axis2, all three compare metrics), legend close-ups — each in
both light and forced `prefers-color-scheme: dark` — plus both SVG figures at
1600 px. `renders/canvas-stats.json` holds per-view pixel histograms pulled
straight off the canvas with `getImageData`.

## 3. Verdict

**The flagship SVG is good and can go in a PR essentially as-is** (one cosmetic
table fix). **The heatmap page is not** — its default offline appearance is
corrupted by a compositing bug (S1) and two of its three legends are illegible
(S2, S3).

Answering the specific questions asked:

- **Uniform wash?** The figure *looks* washed out, and that is a real bug — but
  not a normalisation bug. The normalisation is correct; the canvas is being
  composited at 55% opacity (S1). With the dimming removed the same data has
  1.81× the contrast and clear structure. The row-sum invariant prints
  1.41e-15 and the ramp domain is fixed at ±4 and honestly labelled.
- **Do the four panels differ?** Yes, obviously and immediately. See §4.
- **Legend with numeric bounds?** Present in both artifacts. `1/16× · 1/4× ·
  1× uniform · 4× · 16×` on the magnitude ramp, `−4 … +4` on compare, `0.5–1`
  and `1–26.3` on the marginal strips. The bounds are there; two of the three
  legends are unreadable for layout reasons, not for missing content.
- **Dead cells invisible?** Confirmed, with a caveat that matters — see S4.
- **Light and dark?** Both render. Dark inverts the ramp's salience (S6).

## 4. Do the four panels look different? — yes

`renders/svg-flagship-4axis-1600px.png`

They are visibly, immediately distinguishable, and the differences match the
stated Jaccard numbers rather than contradicting them:

- **A axis1_general** — near-even lavender field, mild left-to-right gradient.
- **B axis2_legal** — strong gradient, dark concentrated left edge, washed-out
  right third, visible horizontal banding around L15–L27.
- **C axis3_code_agentic** — closest to A (their pairwise Jaccard is 0.602, the
  highest in the matrix — so the figure agrees with the table).
- **D axis4_reasoning_termination** — the most extreme: dense dark left block,
  very pale right, strong horizontal structure.

The headline `DIFFERENT: axes select substantially different experts (mean
pairwise 0.424 vs chance 0.265, 1.60x chance; range 0.347-0.602)` is supported
by what is on screen. B and D would never be mistaken for A.

The `R reference quant` panel and the per-panel Σ / max× / entropy / clipped
annotations all render cleanly. Provenance discipline in
`axis-panels.SYNTHETIC.svg` is exemplary — orange banner, diagonal
`SYNTHETIC LAYOUT PREVIEW` watermark, per-panel `NOT MEASURED` labels and `[SYN]`
markers in the table. There is no way to mistake it for the real figure.

## 5. Ranked visual defects

### S1 — CRITICAL — the offline figure is drawn at 55% opacity, and the export disagrees with the screen

`renders/light-13-DEFECT-stale-dimming-comparison.png` (top = as shipped, bottom = same data undimmed)

`poll()`'s `catch` adds `.stale` to `#figwrap` unconditionally
(`heatmap.html:1721`), and `heatmap.html:109` is
`.figwrap.stale canvas#fig{opacity:.55}`. In offline mode the live poll has no
serve to talk to, so it fails forever, so **every offline view is permanently
faded to 55%** — including the operator's screen and every screenshot pasted
into a PR. Measured on the heat field:

| | contrast (σ) | dynamic range |
|---|---|---|
| as shipped | 21.10 | [125, 255] |
| `.stale` removed | 38.21 | [20, 255] |

That is **1.81× of the contrast thrown away**, and it is the entire reason the
figure reads as a pale uniform wash. The dimming is meant to signal "the network
blinked, this frame is old" — but an offline frame loaded from disk is not stale
and has nothing to do with the poll.

Worse: CSS `opacity` is a display-time compositing effect, so `toDataURL()` is
unaffected. Verified — the exported PNG has σ = 38.21 while the screen shows
σ = 21.10. **What you see is not what you export.**

*Fix:* gate the class on `!S.offline` in the `catch`, or clear it whenever
`S.offline` is set. One line.

### S2 — HIGH — compare and mismatch legend captions collide with their own tick labels

`renders/light-10-legend-strip.png`, `renders/light-05-mismatch.png`

`rampTicks()` draws its labels at baseline `y+h+13` = `y+24`. Both callers then
draw a caption at `y+27` — **3 px lower, at a 10 px font**. The glyphs overlap.

The compare legend literally renders as `−4` struck through by `A hotter`, `−2`
overprinted by `hotter`, `0` overprinted by `B hotter`, and `B hotter →` running
into `+2`. The mismatch legend does the same: `0 agrees` is buried under
`over-tiered (wasted bits)`. **The key to reading the compare panel is
unreadable.** Needs ~14 px more, not 3.

### S3 — HIGH — legend titles run into the ramp swatch

`renders/light-06-legend-closeup.png`

Legend titles are drawn at `x`, the ramp bar at a hardcoded `x+312`, with no
measurement. `magnitude — count share ÷ uniform, log2, clipped` overruns 312 px
and the word `clipped` touches the gradient bar. `mismatch — rank(heat) −
rank(tier), within layer` does the same. Either measure the title with
`ctx.measureText` or move the ramp to a computed offset.

### S4 — HIGH — isolated dead cells are invisible at default settings

`renders/light-11-deadprobe-flag-off.png` vs `renders/light-12-deadprobe-flag-on.png`

**Confirmed, but the received description is slightly wrong and the correction
matters.** The colour maths is exactly as claimed —
ΔE(`#FFFFFF`, `#FCFBFD`) = **1.66** and ΔE(dead, export ground) = **0.00**, both
under the ~2.3 JND threshold. But a dead *block* is perfectly visible, because
it sits in a lavender field, not on the ground — it reads as a white hole by
shape.

What is genuinely invisible is a **single** dead cell. No real dump has any dead
cells at all (all four axes: **0 / 19200**), so a probe fixture was derived from
axis1 record 18 by zeroing a 20×30 block plus a one-cell-per-layer diagonal.
With the flag off the block is obvious and **the diagonal cannot be found at
all**. With `flag dead cells` on, the diagonal is unmistakable. The control
works correctly — it is simply off by default, and the realistic failure mode
(a handful of scattered never-routed experts) is exactly the case it hides.

Compounding it: the dumps *do* contain 39–249 cells per axis clipped below
1/16×, which render at `#FCFBFD`. So the pale specks visible all over the figure
are clipped-cold cells, and **a dead cell and a merely-very-cold cell are
indistinguishable** — 1.66 ΔE apart. The footer line `Dead (never routed): 0 —
drawn pure white` is the only thing telling you which you are looking at.

Also inconsistent: `flagship-4axis.svg` encodes the same concept as a **grey**
`never routed (off scale)` swatch, which is legible. The two artifacts disagree.

*Fix:* default `flagDead` to on, or take dead off pure white entirely and adopt
the SVG's grey.

### S5 — MEDIUM — the K-tier strip is 31% of the canvas and carries no information

`renders/light-02-default-figure.png`

Measured from the pixel histogram: `#73C189` — the flat K3 green — is
**31.16% of every pixel on the canvas**. The panel is a featureless rectangle
roughly the same height as the heatmap itself. The page already knows and says
so, in grey 10 px at the very bottom: `⚠ Tier is uniform (K3 everywhere): the
tier strip carries no information and the mismatch panel collapses to the heat
rank.` It is still on by default, and the mismatch panel it warns about (another
full-height panel of blue/orange noise) is a re-encoding of the heat rank.

A third of the figure is spent proving a constant. Auto-collapse both when
`tierVocab(f).length === 1`.

### S6 — MEDIUM — dark mode inverts the ramp's salience

`renders/dark-02b-default-figure-UNDIMMED.png`

Purples-9 is a sequential light→dark ramp designed for a white ground; the
figure ground flips to `#14141B` with the theme. ΔE from the ground per swatch:

`92.3, 87.5, 81.0, 71.8, 62.3, 56.9, 56.7, 64.2, 69.5`

**Non-monotonic, with the minimum at swatch 7 of 9.** On the dark ground the
*hottest* cells are the hardest to see and the *coldest* cells are the
brightest, highest-contrast thing on screen. An operator scanning dark mode for
hot experts has their eye pulled to precisely the wrong cells. Dark mode needs
its own ramp, not the light one on an inverted ground.

Separately, the theme button's tooltip claims *"figure panels always render on
their designed ground"*. That is false — `ground()` returns `--fig-ground`,
which the dark block overrides. Fix the tooltip or the behaviour.

### S7 — MEDIUM — flagship SVG: Jaccard table header collides with row 1

`renders/svg-flagship-4axis-1600px.png`

The header row (`a1 general`, `a2 legal`, `a3 code agent`, `a4 reasoning`,
`vs ref`) sits too close to the first data row: the descenders of `legal`,
`agent` and `reasoning` overlap `0.395`, `0.602`, `0.347`, `0.399`. Rows 2–4 are
fine. Readable, but sloppy in a figure that is otherwise clean — and it is the
table the headline claim rests on. Needs ~6 px more leading on that one row.
Same defect in `axis-panels.SYNTHETIC.svg`.

### S8 — LOW — offline mode opens with a red error banner for a serve nobody asked for

`renders/light-01-default-fullpage.png`

Loading a dump from disk produces two stacked banners (~90 px) above the figure:
a red `endpoint unreachable (1): network: Failed to fetch …` and a yellow
`offline mode: showing «…»`, with `error · 1 consecutive failures` in the header.
The red one is about `http://127.0.0.1:8000`, which the operator never chose to
use. Together with S1 the page reads as broken when it is working fine. Suppress
the net banner while `S.offline` is set.

### S9 — LOW — the dead swatch in the legend reads as an unchecked checkbox

An 11 px white square with a hairline border, next to the text
`dead (never routed)`, sitting in a row of controls. It looks like a checkbox,
not a colour sample — and with the flag off it is white-on-white, so it is a
colour sample showing nothing. Follows from S4.

## 6. Things that are right, and were verified rather than assumed

- Real data all the way through — 19 records indexed from a 15 MB JSONL via the
  page's own file picker; header reads `step 1900 · interval 19 · Σ assignments
  6.91e+8`.
- The **count ↔ mass toggle genuinely changes the figure** — 30.06% of pixels
  differ, max channel delta 129. `mass_is_real: real gate mass` is reported from
  the dump's flag, and the mass button correctly gates on it.
- **COMPARE works**: both slots fill from different axes, the diverging panel
  shows real blue/red structure, and the footer reports `Per-layer Spearman ρ
  min 0.03 / median 0.40 / max 0.60 · top-32 overlap median 28%`. All three
  compare metrics render.
- Marginal strips (`ent`, `max×`) carry numeric bounds.
- Nearest-neighbour blitting is honoured — cells are crisp, no resampled
  in-between colours at any zoom.
- Zero JS exceptions across all 30 renders. The only console noise is the
  expected `ERR_CONNECTION_REFUSED` from the live poll.
- The SYNTHETIC SVG cannot be mistaken for real data.

## 7. Reproducing

```sh
./render.sh                      # everything: heatmap light+dark, dead probe, both SVGs
./render.sh --only svg           # SVG figures only
./render.sh --only heatmap --scheme dark
./render.sh --only deadprobe     # derives the dead-cell fixture and shoots flag on/off
./render.sh --bootstrap          # re-install chromium + libs + fonts on a fresh box
```

`render_heatmap.py` holds the driving logic; `render.sh` supplies the
`LD_LIBRARY_PATH` / `FONTCONFIG_FILE` this box needs and filters chrome's dbus
noise.

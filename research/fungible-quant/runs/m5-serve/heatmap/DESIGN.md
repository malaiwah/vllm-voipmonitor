# FQ expert heatmap — design decision

Status: **decided, ready to implement**. No code yet; this document is the spec.
Author: research agent, 2026-08-11.
Scope: the live layer × expert activation heatmap, its K-tier overlay, its reset control, its
compare mode, and the flagship 4-way corpus image.

Everything numeric below was measured on the real dumps in
`research/fungible-quant/runs/m5-serve/results/k3-fq/*.jsonl` (75 layers × 256 experts,
four corpus replays). The validation renders that back the visual arguments were produced
throwaway during this study and are described inline; they are not committed.

---

## 0. The decision, in one page

| Question | Decision |
|---|---|
| How to show magnitude + K tier in one cell | **Do not.** Option (b): two matrices sharing an identical layout — a magnitude panel and a K-tier strip. Reject the bivariate fill (option c) outright. |
| The panel that actually answers the operator's question | A **third, derived panel**: tier↔heat *mismatch*. See §2.5. This is the money view and it dissolves the double-encoding problem rather than solving it. |
| Magnitude scale | `log2(share / uniform)`, domain fixed at **[−4, +4]** (i.e. 1/16× to 16× uniform), single-hue sequential light→dark. Domain is **fixed across all panels, never auto-scaled per panel.** |
| Magnitude ramp | ColorBrewer **Purples-9**: `#FCFBFD #EFEDF5 #DADAEB #BCBDDC #9E9AC8 #807DBA #6A51A3 #54278F #3F007D` |
| K-tier ramp (ordinal, K2<K3<K4<K5) | Single-hue green, even L\* spacing: **`#BBEFC7` `#73C189` `#289352` `#096231`**. Adjacent ΔE ≈ 20 under normal, protan, deutan *and* tritan vision. |
| Compare-mode metric | **Δ of share, in units of uniform share** (`256·share_B − 256·share_A`), *not* log-ratio. Linear, domain **±4**, hard-clipped with clip marks. |
| Compare-mode ramp | Blue ↔ **neutral gray** ↔ orange, symmetric and monotone in L\*: **`#005F9D #2D81C0 #7DA4CE #C1C1C1 #D89067 #C85B37 #9B3925`** (L\* = 39/52/66/**78**/66/52/38). |
| Rendering | One `<canvas>`, `ImageData` written per cell, `image-rendering: pixelated`, integer device-pixel cell size. Not 19 200 DOM nodes. |
| Column order | Two modes. Live view: **native expert id** (default). Flagship image: **shared pooled-rank order** — one permutation computed once from the pooled baseline, applied identically to every panel, and persisted alongside the image. |
| Normalisation | Per-layer share (which, for this model, equals per-run share up to the constant 75 — see §5.1), expressed relative to uniform `1/256`. |

---

## 1. Prior art

I read the sources marked **[read]**; the others are pointers I located but did not open in
full, and are labelled as such so nobody cites them through me.

### 1.1 The routing-analysis literature

**Switch Transformer** (Fedus, Zoph, Shazeer, 2021) — <https://arxiv.org/abs/2101.03961>.
The canonical framing of expert load as a *balance* problem: the auxiliary load-balancing loss is
the dot product of the fraction of tokens dispatched to each expert against the mean router
probability. Its visual language is **scalar-per-expert bar/line plots against a uniform
reference**, plus capacity-factor / token-drop rates. What worked: making "uniform" an explicit
drawn reference rather than an implied one. What does not transfer: it never had to draw
*L × E* at once — the expert counts are small.

**Mixtral of Experts** (Jiang et al., 2024) — <https://arxiv.org/abs/2401.04088>.
Routing analysis over Pile subsets at layers 0, 15, 31; the headline is a **negative result**
("no obvious pattern in expert assignment based on topic"), later summarised in the community as
routing driven by syntactic/positional rather than semantic structure. The design lesson is
uncomfortable and directly relevant to us: *a well-made expert-usage picture of a well-balanced
model is a picture of noise.* Any design that only looks good when there is structure is a
design that will lie by omission. Our validation render (§4.3) reproduces exactly this: the
full-corpus panel is near-featureless, and that is the correct rendering of the data.

**OLMoE** (Muennighoff et al., 2024) — <https://arxiv.org/abs/2409.02060>,
HTML: <https://arxiv.org/html/2409.02060v2> **[read, §5]**. The most careful public treatment.
Four analyses: router saturation, expert co-activation, domain specialisation, vocabulary
specialisation. Two things to steal:

* **Normalisation is defined as a formula in the caption, not left implicit.** Domain
  specialisation is `N_{E_i,D}^{(k)} / N_D` — the proportion of tokens from domain *D* for which
  expert *E_i* is in the top-k. Co-activation is `N_{E_i,E_j} / N_{E_i}`.
* **The uniform baseline is drawn on the figure.** Figure 22: "Horizontal gray lines correspond
  to random chance or uniform routing (8/64 = 12.5% per expert for OLMoE … 2/8 = 25% for
  Mixtral)."

What is criticised / does not scale: domain specialisation is drawn as **per-domain line plots
over 64 expert positions**, one panel per layer. At 64 experts and 3 layers that reads. At
**256 experts and 75 layers it is 75 panels of spaghetti** — it does not survive our scale. The
co-activation heatmaps hit the same wall and were only shown for "the 32 experts with the highest
maximum co-activation score", i.e. they aggressively subset rather than render everything.

**Expert-Choice routing** (Zhou et al., NeurIPS 2022) —
<https://papers.nips.cc/paper_files/paper/2022/hash/2f00ecd787b432c1d36f3de9800728eb-Abstract-Conference.html>
*[pointer]*. Reframes the load problem by construction (experts pick tokens, fixed bucket size).
Relevant to us because it establishes "max routing load" / imbalance ratio as the accepted
*scalar* summary — a number we should print next to each panel, not instead of it.

Critical counterweights worth reading before over-interpreting any picture we ship:
*The Myth of Expert Specialization in MoEs* — <https://arxiv.org/html/2604.09780v1> *[pointer]*,
and *What Gets Activated: Uncovering Domain and Driver Experts in MoE LMs* —
<https://arxiv.org/html/2601.10159> *[pointer]*. Both argue routing structure often reflects
representation geometry rather than semantics. A "brain activity under different traffic" caption
is a claim; the honest caption is "routing share differs by X".

### 1.2 Systems / serving tools (our closest neighbours)

**DeepSeek EPLB** — <https://github.com/deepseek-ai/EPLB> **[read]**. Expert load is exactly our
matrix: a 2-D tensor, rows = layers, columns = experts, values = numeric load. The README's own
example is `torch.tensor([[90, 132, 40, ...], [20, 107, 104, ...]])` for a 2-layer, 12-expert
model. Notably it ships **no heatmap** — only a *placement* diagram (experts → GPUs). The
load matrix itself is treated as an input to an algorithm, never as something a human looks at.
That gap is precisely what we are filling.

**vLLM EPLB** — <https://docs.vllm.ai/en/latest/serving/expert_parallel_deployment/>,
feature PR <https://github.com/vllm-project/vllm/pull/18343>, vllm-ascend design doc
<https://docs.vllm.ai/projects/ascend/en/latest/developer_guide/Design_Documents/eplb_swift_balancer.html>.
Keeps a **sliding window of expert load with shape `(window_size, num_moe_layers,
num_physical_experts)`** and rebalances periodically. Two design imports:
(1) the *windowed* view is the operator-relevant one, not the since-boot cumulative — which is
why the reset control matters; (2) load is measured as **token count per expert per forward
pass**, matching our `count`. There is no first-party visualisation, so no conventions to inherit
and none to violate.

**moe-viz** (Martin Alderson) — <https://martinalderson.com/posts/moe-expert-routing-visualization/>,
tool at <https://moe-viz.martinalderson.com> **[read]**. The nearest thing to what the operator
is asking for. Two stacked panels: a live per-token routing panel, and a **cumulative heatmap
across the whole generation**. Best finding, and the one that justifies the reset control:
"~25% of experts are inactive for any given prompt, but it's always a *different* 25%." What is
criticised: the author is explicit that it is a weekend project on a patched llama.cpp and "may
have serious mistakes"; there is no stated normalisation, no uniform reference line, and no
colour-scale documentation — so the picture cannot be compared across prompts of different
length. That is the exact failure mode §5 exists to prevent.

**MoE-Visualization-GPT20B** — <https://github.com/s4piru/MoE-Visualization-GPT20B> *[pointer]*.
Compares routing under two prompts via a **Jensen–Shannon divergence heatmap** over
(continuation token × router layer), plus a CSV of the largest per-layer expert shifts in
*model-normalized* routing weight. Directly relevant to compare mode: it reduces "two runs" to a
per-(layer, position) divergence scalar and then lists the movers as a table. We should do both —
a diverging picture *and* a ranked table of movers, because the picture cannot be read to a
specific expert id at 3 px.

### 1.3 Interpretability conventions

**Neuronpedia** — <https://www.neuronpedia.org/>. Per-layer scrollable heatmaps, neurons
colour-coded by activation; the durable convention it carries is *token-level colouring against a
signed scale*, and the practice of pairing every picture with the top-activating exemplars.
Transfers to us as: the heatmap is a navigation surface, and hovering must produce the underlying
record (layer, expert, count, mass, tier, share, rank).

**CircuitsVis / TransformerLens** — <https://github.com/TransformerLensOrg/CircuitsVis>.
`colored_tokens` takes values like `[0.123, -0.226]` — i.e. the interpretability default for
*signed* quantities is a diverging scale with a neutral zero, and for *unsigned* activations a
single-hue ramp. Our magnitude field is unsigned once normalised (share ≥ 0) and our compare
field is signed; the two must therefore look categorically different, which the palettes in §3
enforce.

### 1.4 Colour and matrix-layout method

* Rainbow considered harmful: <https://hess.copernicus.org/articles/25/4549/2021/>; empirical
  ranking of quantitative colormaps, *Somewhere Over the Rainbow* (CHI 2018),
  <https://idl.cs.washington.edu/files/2018-QuantitativeColor-CHI.pdf>.
* ColorBrewer (source of Purples-9 and of the "diverging schemes are centred on a neutral, and
  RdBu's odd-class midpoint is `#f7f7f7`, not pure white" convention): <https://colorbrewer2.org>.
* Okabe–Ito / Color Universal Design, popularised by Wong, *Nature Methods* 2011:
  <https://jfly.uni-koeln.de/color/>, <https://www.nature.com/articles/nmeth.1618>.
* **Colour discrimination collapses on small marks** — the single most load-bearing fact for a
  3-px cell. Wilke, *Fundamentals of Data Visualization*, "Common pitfalls of color use":
  <https://clauswilke.com/dataviz/color-pitfalls.html>; Datawrapper,
  <https://www.datawrapper.de/blog/colors>; Szafir, *Modeling Color Difference for Visualization
  Design* (VIS 2017), <https://danielleszafir.com/colordiff_vis2017.pdf> — perceived colour
  difference varies **inversely with mark size**.
* Bivariate fill is a known-hard encoding: Axis Maps,
  <https://www.axismaps.com/guide/bivariate-choropleth>; Joshua Stevens,
  <https://www.joshuastevens.net/cartography/make-a-bivariate-choropleth-map/> — "if you try to
  make bivariate color schemes that are simultaneously colorblind-safe, photocopy-safe, and print
  friendly, you're going to have a bad time." See also *Evaluating Encodings for Bivariate Edges
  in Adjacency Matrices*, <https://arxiv.org/pdf/2604.14791> *[pointer]* — the same problem in
  exactly our mark geometry.
* Matrix reordering: Behrisch et al., *Matrix Reordering Methods for Table and Network
  Visualization*, CGF 2016 —
  <http://iihm.imag.fr/blanch/teaching/infovis/readings/2016-Behrisch-Redordering.pdf>.
* Canvas vs SVG at scale: Apache ECharts handbook,
  <https://apache.github.io/echarts-handbook/en/best-practices/canvas-vs-svg/>.

---

## 2. The hard problem: two variables, one cell

### 2.1 What the two variables actually are

* **Magnitude** — continuous, heavy-tailed, unsigned. Measured on the real dumps: per-layer share
  relative to uniform spans **0 to ~28×**, median ≈ 0.6–0.9×, p99 ≈ 3–7.6×. 43–87% of cells sit
  within 0.5×–2× of uniform depending on corpus.
* **K tier** — **ordinal**, not categorical. K2 < K3 < K4 < K5 is a real order (bits of
  precision). The task framed it as categorical; I am deliberately treating it as ordinal, because
  an ordered encoding for an ordered variable is strictly more readable and, as §3.2 shows, it is
  also the only way to keep four levels separable under all three CVD types at 3 px. Current
  policies in this tree use `{3}` (uniform) and `{3, 5}` (mixed), from `bits_per_expert` in
  `policy-k3-uniform.json` / `policy-mixed-k3k5.json`; the design supports 2–5.

### 2.2 Option (a) — magnitude fill, tier as border / corner marker / texture

**Rejected on arithmetic, before any perceptual argument.**

At the cell sizes this matrix forces (§4), a border is not a decoration, it is most of the cell:

| cell size | interior after a 1 px border | border share of cell area |
|---|---|---|
| 3 px | 1×1 px | **89%** |
| 4 px | 2×2 px | **75%** |
| 6 px | 4×4 px | 56% |
| 8 px | 6×6 px | 44% |
| 12 px | 10×10 px | 31% |

The border destroys the magnitude channel it is supposed to annotate, and it does so worst
exactly where cells are smallest. Corner markers fare no better: a 2×2 px triangle in a 4 px cell
is 25% of the area and 2 px is below the threshold at which shape is discriminable at all.
Texture (hatching) needs ≥ 6–8 px of period to read as a pattern rather than as a lighter fill —
at 3 px it *aliases into a lightness change*, i.e. it silently corrupts the magnitude reading.

The only regime where (a) works is a zoomed detail view at ≥ 12 px/cell, where it is genuinely
good. Keep it there (§6.3) and nowhere else.

### 2.3 Option (c) — fill hue = K tier, fill lightness = magnitude

**Rejected. This is the trap.** I built it and looked at it; three independent failures.

1. **Hues have different intrinsic lightness, so "lightness = magnitude" is not comparable across
   tiers.** With Okabe–Ito hues, orange `#E69F00` is far lighter than blue `#0072B2`. A dim orange
   cell and a bright blue cell land at the same L\*. The operator's core comparison — "is this
   expert hotter than that one" — becomes *unanswerable across tier boundaries*, which is the only
   comparison that matters when deciding re-tiering.
2. **Under CVD the tier identity evaporates, and it takes the magnitude with it.** In the
   deuteranope simulation of the rendered matrix, the green and orange tiers collapse toward
   lavender and yellow of similar lightness. Lightness is the one channel CVD viewers keep intact
   — and this design has already spent it on magnitude, so there is nothing left to carry tier.
3. **It loses on small marks even for normal trichromats.** Perceived colour difference varies
   inversely with mark size (Szafir 2017; Wilke's "colors are much easier to distinguish when
   applied to large areas than to small ones"). At 3 px the hue mosaic wins the visual competition
   outright: in the render, the tier bands were obvious and the magnitude field inside each band
   was unreadable. You get a tier map with decorative noise, not two variables.

This is the documented failure of bivariate choropleths (16 colours for a 4×4 scheme; legends
that must be memorised; ColorBrewer declining to ship CVD-safe bivariate schemes) reproduced in a
matrix, at a mark size an order of magnitude smaller than a US county.

### 2.4 Option (b) — two matrices sharing one layout — **RECOMMENDED**

Magnitude panel (Purples, sequential) + K-tier strip (Greens, ordinal), **identical row order,
identical column order, identical cell geometry, vertically stacked and pixel-aligned** so the
eye can drop a vertical line between them.

Why it wins:

* **Each channel keeps a full dynamic range.** Magnitude gets the whole lightness axis of one hue;
  tier gets the whole lightness axis of another. Nothing is shared, so nothing is compromised.
* **CVD safety is structural, not lucky.** Both scales are single-hue sequential, so both degrade
  to monotone grayscale ramps under any CVD and under photocopying. Measured adjacent-step ΔE for
  the four tier steps: 22.7 / 20.7 / 20.5 normal; 19.3 / 18.6 / 19.5 protan; 19.6 / 18.9 / 18.2
  deutan; 35.7 / 29.6 / 24.2 tritan. No pair drops below ΔE ≈ 18 in any condition.
* **It survives being screenshotted small.** Shrinking a stack of two single-hue panels degrades
  gracefully — the ramps blur toward their means and the *large-scale* structure (which layer
  bands are hot, which tier bands exist) is the last thing to survive, which is the right thing to
  survive. Shrinking a bivariate mosaic averages *different hues* together and produces colours
  that are in neither scale, i.e. it manufactures a false reading. This is the decisive argument
  for docs.
* **The tier strip costs almost nothing in vertical space.** Tiers are far more spatially
  coherent than heat, so the strip can be drawn at a fraction of the magnitude panel's height and
  still read.

Cost, stated plainly: it uses more vertical space, and reading "the tier of *this* cell" requires
a saccade rather than being in the fill. For the flagship image that cost is real but small; for
the live view it is eliminated by the hover readout, which reports both numbers exactly.

### 2.5 What I am adding, because (b) alone still under-serves the operator

The operator's real question is not "what is the heat and what is the tier". It is **"where is the
tiering wrong?"** — which experts are burning K5 bits on cold traffic, and which hot experts are
starved at K2/K3. That is *one derived signed scalar per cell*, and it wants a diverging scale,
not a bivariate one:

```
mismatch(l,e) = zheat(l,e) − ztier(l,e)
    zheat  = rank of heat within layer l, mapped to [0,1]
    ztier  = rank of K tier within layer l, mapped to [0,1] (ties averaged)
```

`mismatch > 0` = **hot but under-tiered** (starved; candidate for promotion).
`mismatch < 0` = **cold but over-tiered** (wasted bits; candidate for demotion).
Zero = tier rank agrees with heat rank. Rendered with the §3.3 diverging ramp, neutral gray at 0.

Two honest caveats that must sit in the caption:

* If the *policy itself* assigns K by heat rank, this panel is near-uniform gray by construction —
  which is the correct answer ("policy is consistent with observed traffic"), and the interesting
  cells are the few that are not gray. Corollary discovered while validating: **if columns are
  sorted by heat and tiers were assigned by heat, the tier strip degenerates into a trivial
  staircase carrying no information.** In that configuration the tier strip should be drawn in
  *native expert-id order* (where it is informative) and the mismatch panel becomes the only
  useful tier view in sorted order.
* Rank-vs-rank deliberately discards magnitude. A layer where the hottest expert is 28× uniform
  and one where it is 2.6× produce the same mismatch picture. Print the per-layer max-share and
  entropy in the row gutter so the reader can tell those apart.

---

## 3. Colour specification (exact, non-negotiable)

All three scales below were computed in CIELAB, checked for sRGB gamut, and simulated under
protanopia, deuteranopia and tritanopia (Viénot/Brettel LMS projection). Numbers quoted are
measured, not asserted.

### 3.1 Magnitude — sequential, ONE hue, light → dark

**ColorBrewer Purples-9.** Light = cold, dark = hot.

```
#FCFBFD  #EFEDF5  #DADAEB  #BCBDDC  #9E9AC8  #807DBA  #6A51A3  #54278F  #3F007D
L*  98.7    94.1     87.5     77.4     65.4     54.9     40.6     27.6     18.1
C*   1.1     4.2      8.9     16.6     25.7     35.1     50.3     65.5     73.3
```

L\* is monotone decreasing across all nine stops, so the ramp is a valid grayscale ramp for any
CVD and for print. Interpolate in Lab (or Oklab), not in sRGB.

Purple, specifically, and not viridis: viridis is perceptually excellent but **multi-hue**. In
this system hue is a semantic channel — it distinguishes the three scales from each other at a
glance. Keeping the magnitude ramp strictly single-hue makes it unambiguous that *hue carries no
information within the magnitude panel*, which is what lets the tier strip and the compare panel
own their hues without confusion. Purple was chosen over blue so that blue stays reserved for one
pole of the diverging scale.

**Never** a rainbow, jet, or turbo here.

### 3.2 K tier — ordinal, ONE hue, four fixed steps

```
K2  #BBEFC7   L* 90
K3  #73C189   L* 72
K4  #289352   L* 54
K5  #096231   L* 36
```

Fixed mapping by K value, **never cycled and never reassigned by frequency**: if a run contains
only K3 and K5, they render as `#73C189` and `#096231` with K2 and K4 simply absent from the
legend. The legend always shows all four positions with the unused ones dimmed, so two images
from different runs are directly comparable.

Even L\* spacing of 18 units per step was chosen so the ordinal reading survives every CVD
collapse. Measured adjacent ΔE: normal 22.7/20.7/20.5 · protan 19.3/18.6/19.5 ·
deutan 19.6/18.9/18.2 · tritan 35.7/29.6/24.2.

If a fifth or sixth tier is ever introduced, re-derive the ramp at even L\* spacing across the new
count; do not squeeze extra hues in.

### 3.3 Compare mode — diverging, two hues, NEUTRAL GRAY midpoint

Blue (A hotter) ↔ neutral gray (no difference) ↔ orange (B hotter):

```
−4×      −2.7×     −1.3×      0        +1.3×     +2.7×     +4×
#005F9D  #2D81C0  #7DA4CE  #C1C1C1  #D89067  #C85B37  #9B3925
L*  38.9    51.9     66.0     78.1     66.2     51.9     37.9
```

Properties, all verified:

* The midpoint is a **true neutral**: L\*=78, a\*=b\*=0. Not white, not off-white, not a hue.
* **L\* is exactly symmetric and monotone outward from the centre** (78 → 66 → 52 → 38 on both
  wings). This is the part people get wrong: naïvely centring RdBu on a mid-gray produces
  near-centre stops that are *lighter* than the centre, which draws a bright halo around zero and
  invents a feature that is not in the data.
* Blue↔orange is the CVD-safe diverging pair. End-to-end ΔE after simulation: protan 64.6,
  deutan 86.2, tritan 141.7 — the two poles never converge. The sign of the difference remains
  readable to every viewer.
* Chroma is 0 at the centre and rises outward, so "no change" is visually quiet and "big change"
  is loud. On real data (§4.3) this produces a mostly-gray field with sharp sparks, which is the
  honest shape of the answer.

Direction convention, fixed once: **left/first-named corpus = blue, right/second-named corpus =
orange**, stated in the legend as `← A hotter · B hotter →`. Never flip it per panel.

---

## 4. Scale: 19 200 cells

### 4.1 Geometry

75 rows × 256 columns is a **3.41 : 1 landscape** block. Experts on x, layers on y — this is
forced, not chosen: 256 columns need the long axis, and layer depth reading top-to-bottom matches
how everyone already thinks about the stack.

| cell size | panel px | fits where |
|---|---|---|
| 1 px | 256 × 75 | sparkline / thumbnail only |
| 2 px | 512 × 150 | dense docs column |
| **3 px** | **768 × 225** | **default doc figure; 4 stacked panels ≈ 768 × 1000** |
| 4 px | 1024 × 300 | flagship at 1× ; 2×2 grid ≈ 2100 × 700 |
| 6 px | 1536 × 450 | live view on a 1600 px browser column |
| 12 px+ | 3072 × 900 | zoom/detail view only — this is where option (a) becomes legal |

For the flagship, render at **4 px/cell logical, 2× device pixel ratio** → an 8 px physical cell,
sharp on retina and still legible when the PNG is downscaled to a 700 px doc column.

Non-negotiable: **integer cell sizes and `image-rendering: pixelated`**. Fractional cell widths
make the browser resample, and resampling a heatmap silently averages neighbouring experts —
producing cells whose colour corresponds to no data point. This is the single most common way a
correct heatmap becomes a wrong one.

### 4.2 Rendering

**One `<canvas>`, written via a single `ImageData` buffer.**

* 19 200 DOM nodes (`<div>`/`<rect>`) is 4–6× past the ~3–5 k element ceiling where SVG/DOM
  degrades, and this is a *live* view that must repaint on every stats interval. Rejected.
* An `<img>` of a server-rendered PNG is fine for the flagship export and is the wrong choice for
  the live view (no hover, no local re-normalisation, a round-trip per repaint).
* Canvas gives O(cells) fill with no retained scene graph, sub-millisecond repaints at this size,
  and trivially exact hit-testing: `expert = floor((x − x0) / cell)`, `layer = floor((y − y0) /
  cell)`. Draw at 1 device pixel per data pixel into a 256×75 offscreen `ImageData`, then blit
  scaled with smoothing disabled — this keeps the redraw cost independent of zoom level.
* Accessibility: canvas is opaque to screen readers. Ship the underlying matrix as a downloadable
  CSV/JSON next to the figure and give the canvas a text summary (`aria-label` with layer count,
  expert count, max share, entropy). Do not pretend the canvas is accessible.

### 4.3 Ordering and aggregation — measured, not guessed

The question "should we sort experts by heat within a layer, and does that destroy expert identity
across panels?" has an empirical answer on this data.

**Finding 1 — there is no cross-layer structure along the expert-index axis, so sorting per layer
destroys nothing real.** Correlation of the per-expert-index heat vector between layer pairs
(code-axis corpus): mean **−0.000**, p95 **0.113**, max 0.575; mean **adjacent-layer** correlation
**0.011**. Expert 137 in layer 5 has no relationship to expert 137 in layer 40 — they are
different weight matrices with independently-trained routers. Any vertical stripe the eye finds in
the native-order heatmap is pareidolia. This is the licence to reorder columns per layer.

**Finding 2 — sorting transforms the picture from noise to signal.** In native expert-id order all
four corpus panels look like the same purple static (this is the Mixtral negative result showing
up in our data). Under a shared pooled-rank permutation, three of the four panels show a clean
dark→light gradient and the fourth (full-corpus) stays visibly flat — instantly communicating "the
full corpus routes closer to uniform than the others". That contrast is the whole flagship story
and it is invisible without ordering.

**Finding 3 — the permutation must be SHARED, not per-panel.** Per-panel sorting makes every panel
show the same monotone gradient by construction and destroys all comparison — it is the
visualisation equivalent of fitting on the test set. The rule:

> Compute **one** permutation, per layer, from a designated reference (pooled mean across all
> panels in the figure, or a named baseline corpus). Apply it identically to every panel including
> the tier strip and the compare panel. Persist the permutation as JSON next to the image and
> state the reference in the caption.

**What ordering costs, explicitly:**

* The x axis is no longer expert id. You cannot read "expert 137" off the picture; you need the
  hover readout or the persisted permutation. The live view therefore defaults to **native id
  order** (operators are looking for a specific expert) and the flagship defaults to **sorted**
  (readers are looking for a pattern).
* The picture becomes *baseline-dependent*. Two figures with different references are not
  comparable, and there is a real temptation to pick the reference that makes the story look best.
  Mitigation: the reference is named in the caption and the permutation is committed.
* A sorted panel makes the deviation from monotonicity the signal. Readers untrained on this will
  read the gradient itself as a finding; it is not — it is the sort. The caption must say so.

**Do not aggregate away either axis by default.** Layer-mean and expert-mean marginals are useful
and should be drawn as **thin marginal strips** on the top and left of the panel (a 256-wide
column-mean strip, a 75-tall row-mean strip), which cost ~10 px each and let a reader answer "is
layer 51 unusual?" without eyeballing 256 cells. Binning experts into groups of 4 to halve the
width is tempting and should be rejected for the flagship: at 256 columns and 4 px we are not
space-constrained, and binning hides exactly the single-expert spikes (up to 28× uniform) that
drive re-tiering decisions.

---

## 5. Normalisation — the part that decides whether the picture is honest

### 5.1 The structural fact that simplifies everything

Measured on all four dumps: **the per-layer total is identical for every one of the 75 layers,
to floating-point tolerance.** (`code-axis`: every layer sums to 933 206.149; `full`: 6 749 860.85;
`synthetic`: 95 520.827.) This follows from top-k routing — every token contributes exactly `k`
assignments in every instrumented layer — and it holds for real gate mass too when gate weights
are renormalised over the top-k.

Consequence: **per-layer share and per-run share are the same picture up to the constant 75.**
The choice everyone agonises over does not exist here. What remains is only the choice of
reference and of transform.

Guard rail: this invariant must be *asserted at load time*, not assumed. If a future model has a
shared/always-on expert, a layer with different `top_k`, or a partially-instrumented layer, the
invariant breaks and per-layer normalisation stops being equivalent to per-run. Fail loudly:
`assert max|rowsum − rowsum[0]| / rowsum[0] < 1e-9`, and if it trips, switch to per-layer share
and say so on the figure.

### 5.2 The recommendation

```
value(l,e) = log2( 256 · count(l,e) / Σ_e count(l,e) )        domain [−4, +4], clipped
```

i.e. **log2 of share expressed in multiples of uniform share (1/256)**. Zero means "exactly its
fair share"; +1 means 2× uniform; −1 means half.

* **Run length cancels exactly**, so a 95 k-assignment replay and a 6.7 M-assignment replay are
  directly comparable. This is the whole point: raw counts across our four corpora differ by
  **71×** (7.16 M to 506 M total), so any raw-count rendering is a rendering of transcript length.
* **The reference is meaningful and fixed.** Uniform routing is the load-balancing target the
  model was trained toward, and OLMoE's convention of drawing the uniform line on the figure is
  the right one — here it is the *midpoint of the ramp*, and it must be tick-labelled `1×`.
* **Log, because the distribution demands it.** Measured share/uniform: p50 ≈ 0.6, p95 ≈ 3.2,
  p99 ≈ 7.6, max ≈ 28 (code-axis). On a linear ramp the top 1% of cells consume the top half of
  the colour range and 99% of cells are crushed into the bottom fifth — the picture becomes "a few
  black dots on white". Log2 spreads it: the [−4,+4] domain lands 2028/2739/3995/4802/3501/1470/
  487/153/25 cells in its nine unit-wide bands.
* **Domain [−4, +4] clips the top tail and nothing else — but it clips a real slice of the
  bottom.** ~~"covers 99.87% / 100.00% / 99.97% / 99.90%"~~ **(corrected 2026-08-11, review):**
  those four figures are the *upper* clip only. Counting both ends, the domain covers
  **95.95% / 99.88% / 94.34% / 95.90%** of cells for code-axis / full / synthetic / truncated,
  because **3.92% / 0.12% / 5.64% / 3.99%** of cells sit *below* 1/16× of uniform and saturate at
  the pale end. Clipping must still be shown — mark saturated cells with a 1 px white or black pip
  at the cell corner in the ≥ 8 px zoom view, and report **both** clipped counts in the caption
  (`heatmap.html` already does; do not restate the one-sided number anywhere).
* **Zeros**: the synthetic dump has 42 dead cells. Floor at `1/16` of uniform (the domain edge)
  rather than dropping to −∞, and render true zeros with a distinct hatch or the pure background
  colour `#FFFFFF` so "never routed" is not confused with "rarely routed". Never silently add an
  epsilon without saying so.
  * **OPEN (review 2026-08-11): `#FFFFFF` does not achieve that separation and must change.**
    Measured: `#FFFFFF` vs the palest ramp stop `#FCFBFD` is **ΔE76 = 1.66**, below the ~2.3 JND,
    and vs the light/export figure ground `#FFFFFF` it is **ΔE = 0** — a never-routed cell is
    literally a hole in the exported PNG. It is also indistinguishable from the 3.9–5.6% of cells
    clipped at the bottom (§ above), so the pale end of the figure conflates *dead*, *clipped-low*
    and *merely cold*. `make_axis_panels.py` already rejects this choice for the same figure and
    uses a neutral gray (`#b0b0aa` light / `#6e6e66` dark, ΔE 27.4 from `#FCFBFD`); the two
    renderers of this design disagree. The page's "flag dead cells" control (crimson) is the
    workaround and is OFF by default. Not changed in the review commit: it is a palette decision
    and nobody can see the page render on this box.

### 5.3 What each choice hides — say this in the caption

| Choice | Hides |
|---|---|
| Per-layer / per-run share | **Absolute volume.** A layer during a 3-token burst and a layer during a 3 M-token soak look identical. Mitigate: print total assignments and the wall-clock window on the figure. |
| Relative-to-uniform | **How balanced the layer is overall.** A layer at 0.96 normalised entropy and one at 0.80 both centre on 1×. Mitigate: the per-layer entropy and max-share gutter (measured range here: entropy 0.80–0.99, max-share 2.6×–27.9× uniform). |
| Log2 | **Additive magnitude.** A jump from 0.03× to 0.3× looks identical to 2.4× to 24×, but only the second matters for tiering. This is exactly why compare mode must *not* use log-ratio (§5.4). |
| Clipping at ±4 | The extreme tail. Small in count (≤ 0.13% of cells) but these are precisely the K5 candidates. Always report the count. |
| `count` when `VLLM_FQ_GATE_MASS=0` | That mass is **aliased to count**. Read the collector's `mass_is_real()` / the dump's `mass_is_real` field — **never** infer it by comparing the arrays. Note: the four dumps in `results/k3-fq/` have **no `mass_is_real` key at all**; treat a missing key as unknown and label the figure `mass: unknown (field absent)` rather than defaulting to either answer. |
| EMA decay | The collector applies exponential decay (`decay=0.95` default), so `count` is a **decayed window**, not a lifetime total — the values are floats, not integers. The figure must state the effective window, and the reset control must be documented as "zero the accumulators", not "reset the display". |

### 5.4 Compare mode: the metric, and why not log-ratio

The intuitive choice is `log2(share_B / share_A)`. **It is the wrong default and I can show it.**

Rendered on real data (code-axis vs synthetic) with domain ±3, the log-ratio panel saturates
almost everywhere and reads as red/blue static: 48.6% of cells exceed |log2 r| = 1 and 17.4%
exceed 2. It does that because it treats a cell moving 0.03× → 0.3× uniform as exactly as
important as one moving 2.4× → 24×. The first is a rounding error in a cold expert; the second is
the headline.

Use instead the **difference of shares in units of uniform**:

```
delta(l,e) = 256·share_B(l,e) − 256·share_A(l,e)      linear, domain ±4, clipped
```

* **It conserves.** Measured: each row of `delta` sums to zero to 7e−14. The panel therefore reads
  literally as *"this much traffic moved from these experts to those experts"* — orange area is
  exactly balanced by blue area within every layer. No other candidate metric has this property,
  and it is what makes the picture trustworthy.
* **It is magnitude-weighted by construction**, so the top movers are the cells that matter. The
  ten largest |delta| cells on code-axis vs synthetic are all high-traffic: e.g. L18/E53
  2.4× → 27.9×, L51/E219 22.8× → 2.5×, L43/E180 4.7× → 24.7×.
* **Domain ±4 covers 97.2% of cells**; 23.6% exceed |delta| = 1 and 8.8% exceed 2. The resulting
  picture is mostly quiet gray with sharp sparks — which is the truth.

Offer **symlog** `sign(d)·log2(1 + |d|)` at domain ±3 as an explicit "boost weak differences"
toggle (covers 99.1%), clearly labelled as breaking the conservation reading. Offer log-ratio only
as a third, warned option for the specific question "which experts switched on or off", and gate
it to cells above a share floor.

**Always pair the compare picture with two scalars per layer and one table**, because a 3 px cell
cannot be read to an expert id:

* per-layer **Spearman rank correlation** of expert heat between the two runs — measured here:
  code vs synthetic **0.23/0.37/0.56** (min/median/max), code vs full 0.12/0.39/0.58, and code vs
  the truncated variant of itself **0.85/0.91/0.94**. That last number is the calibration: ~0.9
  means "same traffic", ~0.35 means "genuinely different brain".
* per-layer **top-32 overlap** — code vs synthetic: median **28%**.
* a ranked **table of the top ~50 movers** by |delta| with explicit `(layer, expert, share_A,
  share_B, delta, tier)`, mirroring what MoE-Visualization-GPT20B does with its expert-shift CSV.

---

## 6. Applying it to the requested product

### 6.1 Flagship 4-way image

Vertical stack, one panel per corpus axis, plus shared reference strips. Rendered at 4 px/cell,
2× DPR, PNG.

```
┌─ title + provenance: model, policy, VLLM_FQ_GATE_MASS state, window, total assignments ─┐
│ A  <corpus name>   [75×256 magnitude, Purples, log2 rel-uniform, −4..+4]  │ entropy/max gutter
│ B  <corpus name>   [same domain, same permutation]                        │
│ C  <corpus name>   [same domain, same permutation]                        │
│ D  <corpus name>   [same domain, same permutation]                        │
├───────────────────────────────────────────────────────────────────────────┤
│    K-tier reference strip  [Greens, K2..K5]   (native id order if tiers   │
│                                                are heat-derived — see §2.5)│
├───────────────────────────────────────────────────────────────────────────┤
│    compare  B − A   [diverging, delta in uniform units, ±4, gray at 0]    │
└─ legends: magnitude 1/16×—1×—16× · tiers K2..K5 · compare ←A · B→ ────────┘
caption: normalisation, reference used for the column permutation, clipped-cell count,
         per-panel Spearman vs A, mass_is_real state.
```

A 2×2 grid is acceptable if width is the constraint, but the vertical stack is better: 3.41:1
panels stacked share a common x, which is what makes "same column = same expert" readable.

### 6.2 Live view

* Default: **native expert-id order**, magnitude panel + tier strip, 6 px cells.
* Toggle: `count` ↔ `mass`. When `mass_is_real()` is false the mass option is **disabled with a
  tooltip**, not silently aliased. When the dump lacks the field entirely, label it `unknown`.
* Toggle: sorted ↔ native order, with the reference named.
* **Reset**: zeroes the collector accumulators. It must (a) confirm, since it is destructive and
  affects any concurrent consumer of the stats, (b) stamp the figure with "window since
  `<timestamp>` / `<n>` intervals", and (c) *not* rescale the colour domain — the domain is fixed
  at ±4 always, so the picture immediately after reset is pale and converges as traffic
  accumulates, which correctly communicates low confidence. Auto-rescaling after reset would make
  three tokens of traffic look like a fully-formed routing pattern.
* Hover: exact readout `L{layer} E{expert} · count · mass · share ×uniform · rank in layer · K{n}`.
* Colour domain is **never** auto-scaled from the data, in any mode.

### 6.3 Zoom / detail

Above ~12 px/cell, option (a) becomes legal and useful: magnitude fill + a 1 px tier border, with
the tier colour taken from §3.2. Restrict this to the zoomed pane and to a viewport of at most a
few hundred cells.

---

## 7. Hazards found while validating

1. **A permutation mismatch between panels is silent and catastrophic.** My first full-composition
   render sorted the magnitude panels but forgot the tier strip; the result looked entirely
   plausible and was meaningless. The permutation must be computed once and threaded through every
   panel, with a runtime assertion that every panel in a figure carries the same permutation hash.
2. **Layer 78 exists in policies but not in stats.** `policy-*.json` `bits_per_expert` has **76**
   layer keys (3–78, including the MTP layer); the stats dumps have **75** (3–77). Joining tiers
   from a policy file by position will be off by one layer at the bottom. Take `tier_of` from the
   dump, which is already aligned to `layers`, and assert `len(tier_of) == len(layers)`.
3. **`count` is a decayed float, not a count.** Do not label the axis "tokens" or format the
   readout as an integer.
4. **`mass_is_real` may be absent from a dump.** Absent ≠ false. Render `unknown`.
5. **The pretty picture is the sort.** The strongest visual feature of the flagship image is a
   gradient that the ordering created. If that sentence is not in the caption, the figure is
   misleading.

---

## Appendix — measured facts (real dumps, final interval of each file)

Source: `research/fungible-quant/runs/m5-serve/results/k3-fq/`, 75 layers (3–77) × 256 experts.

| run | file | total assignments | per-layer total | share/uniform p50 / p95 / max | entropy (norm.) min/med/max | dead cells |
|---|---|---|---|---|---|---|
| code-axis | `stats-code-axis.jsonl` | 7.00e7 | 933 206.15 | 0.57 / 3.19 / 27.9 | 0.804 / 0.889 / 0.965 | 0 |
| full | `stats.jsonl` | 5.05e8 | 6 749 860.85 | 0.92 / 1.78 / 12.8 | 0.959 / 0.978 / 0.991 | 0 |
| synthetic | `stats-synthetic.jsonl` | 7.16e6 | 95 520.83 | 0.74 / 2.77 / 22.8 | 0.849 / 0.913 / 0.982 | 42 |
| truncated | `stats-INVALID-truncated-corpus.jsonl` | 7.16e7 | 954 500 | — | — | — |

* Per-layer totals are **identical across all 75 layers** in every file (rel. tol < 1e-12).
* Total assignments span **71×** across runs — raw counts are not comparable without normalisation.
* Domain [−4,+4] on log2(share/uniform) covers 95.95 / 99.88 / 94.34 / 95.90 % of cells
  (corrected 2026-08-11 — the old 99.87 / 100.00 / 99.97 / 99.90 counted the UPPER clip only;
  clipped LOW is 752 / 23 / 1082 / 767 cells, clipped HIGH is 25 / 0 / 5 / 20).
* Cross-layer correlation of per-expert-index heat: mean −0.000, p95 0.113; adjacent layers 0.011.
* Per-layer Spearman between runs: code↔synthetic 0.23/0.37/0.56, code↔full 0.12/0.39/0.58,
  synthetic↔full 0.09/0.32/0.57, code↔truncated 0.85/0.91/0.94.
* Top-32 expert overlap, code vs synthetic: 0.12 / 0.28 / 0.50 (min/median/max per layer).
* `delta` rows sum to zero to 7e−14; |delta| > 1 for 23.6% of cells, > 2 for 8.8%, > 4 for 2.8%.
* Tier vocabulary in current policies: `{3}` uniform, `{3: 18688, 5: 768}` mixed.

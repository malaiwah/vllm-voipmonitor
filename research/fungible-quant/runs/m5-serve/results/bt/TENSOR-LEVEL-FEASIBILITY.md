# One level deeper than the expert — is tensor/slice-level allocation worth it?

**Short answer: the TP-rank level is dead, the gate-vs-up level is dead, and the
one split that is left — FC1 (gate+up) vs FC2 (down) — is worth somewhere
between +0.3% and +14% of the addressable error depending on which error metric
you believe, with our two metrics disagreeing in *sign*. None of it is an
online decision: routing mass is constant across an expert's slices, so the
per-projection choice is a static offline property with nothing for the loop to
track. Online per-slice re-tiering is not worth pursuing. A cheap offline
experiment settles the remaining question and is specified at the end.**

Data: `slice_nmse` and `slice_proxy_err` from our own K4 encode
(`/home/mbelleau/glm52-work-k4`, 75 layers), plus the K2 encode
(`/home/mbelleau/glm52-work-k2`, 75 layers) and the K5 encode
(`/home/mbelleau/glm52-work-k5`, 24 layers) — the same capture
(`/home/mbelleau/glm52-capture`, 1,050,468 tokens), so real *measured* Δloss
curves at slice granularity, not an assumed curve shape. 3,072 slices/layer =
256 experts × 3 projections × 4 TP ranks.

Runtime code read (read-only): `gg-vllm` `exl3.py`,
`exl3_fungible/swap.py`, `exl3_fungible/policy.py`, and b12x
`moe/_shared/kernels/w4a16/mixed_trellis.py` + `kernel.py`.

---

## 1. Is there real per-slice variance? — yes, but almost all of it is a constant

Variance decomposition of `log(error)` inside a layer (median over 75 layers,
fraction of total sum of squares; effects are expert main, projection main,
TP-rank main, expert×projection interaction, residual):

| effect | `slice_nmse` | `slice_proxy_err` |
|---|---|---|
| expert main | 0.1705 | 0.2166 |
| **projection main** | **0.7276** | **0.7305** |
| TP-rank main | **0.0000** | **0.0000** |
| expert × projection | 0.0941 | 0.0350 |
| residual (rank interaction + noise) | 0.0005 | 0.0034 |

Raw spreads (max/min), medians over all layers and experts:

| | `slice_nmse` | `slice_proxy_err` |
|---|---|---|
| across the 3 projections of ONE expert | 1.222 (p90 1.345) | 2.517 (p90 3.755) |
| across the 4 TP ranks of ONE projection | **1.0028** (p90 1.013) | **1.030** (p90 1.111) |
| across experts (per layer) | 1.363 | 3.946 |
| across all 3,072 slices (per layer) | 1.796 | 15.19 |

**The TP-rank dimension is empty.** Rank spread exceeds 1.10 in 0.28% of
(layer, expert, projection) triples under `slice_nmse` and 11.7% under
`slice_proxy_err`; exceeds 1.50 in 0.009% and 1.03% respectively. The rank main
effect is 0.0000 of the variance under both metrics. Rank-slices of the same
projection are statistically interchangeable, exactly as they should be — they
are equal-sized shards of one weight matrix quantized against the same Hessian.
**3,072 units per layer is really 768.** R10 already knows this:
`TensorId(layer, expert, projection)`, no rank, `TENSORS_PER_LAYER` = 768,
`UPGRADE_UNITS_PER_LAYER` = 384 (`r10-review/.../allocation.py:11-21`).

**The projection dimension is real but is mostly a constant.** 73% of the
within-layer variance is the projection main effect — the same three-number
offset applied to every expert. The genuinely per-(expert, projection)
information, the interaction term, is 9.4% (`nmse`) / 3.5% (`proxy`) of the
variance, residual sd 0.126 in log units (~13%).

That interaction is *real*, not noise: its rank order at the 4→5 bit step and
at the 2→4 bit step agree with Spearman **0.9986** (median over the 24 layers
with all three encodes). But it does **not** transfer across layers —
Spearman(residual at layer *l*, residual at layer *l+1*) = **0.0256** over 22
adjacent pairs. So it is a per-layer measured quantity or it is nothing, which
is another way of saying R10's per-layer probe is the right and only way to get
it.

## 2. Which projection is most sensitive? — the two metrics disagree in sign

Per-projection median error, K4 encode, median over 75 layers:

| | `slice_nmse` | ratio | `slice_proxy_err` | ratio |
|---|---|---|---|---|
| `gate_proj` | 6.208e-03 | 1.000 | 1.333e-03 | 1.000 |
| `up_proj` | 6.210e-03 | 1.000 | 1.542e-03 | 1.157 |
| `down_proj` | **5.117e-03** | **0.824** | **3.377e-03** | **2.533** |

Both orderings are essentially universal:

- `slice_nmse`: `down_proj` is the **best** projection in **75/75** layers and
  in 98.9% of individual (layer, expert) pairs.
- `slice_proxy_err`: `down_proj` is the **worst** projection in **75/75**
  layers and in **99.86%** of (layer, expert) pairs.

So `down_proj` is quantized *more accurately in weight space* and *less
accurately in Hessian-weighted space*. The amplification factor
`proxy_err / nmse` makes this explicit:

| | median `proxy/nmse` | p10 | p90 |
|---|---|---|---|
| `gate_proj` | 0.1975 | 0.082 | 0.282 |
| `up_proj` | 0.2364 | 0.099 | 0.344 |
| `down_proj` | **0.6299** | 0.344 | 0.776 |

`down_proj` residual error aligns **3.19×** more with the high-curvature
directions of its input covariance than `gate_proj`'s does. That is consistent
with `down_proj` being the projection whose inputs are post-SwiGLU
intermediates — sparse, heavy-tailed, ill-conditioned — and it is *also*
consistent with `down_proj` being the K-split under TP=4 (`suh` 512, `svh` 6144
in the segment header, versus `suh` 6144 / `svh` 512 for gate and up). **Our
data cannot separate those two explanations.** It only establishes that the
effect is large and universal.

**Caveat that matters, stated plainly.** `proxy_err` is
`tr(EᵀHE) / tr(WᵀHW)` (exllamav3 `exl3_lib/quantize.py:1089-1095`) — a
*relative* Hessian-weighted error, normalised by each tensor's own energy.
Comparing it across projections implicitly asserts that a 1% relative output
perturbation of `down_proj` costs the model the same as a 1% relative
perturbation of `gate_proj`. That is not established, and the SwiGLU makes it
doubtful in both directions (a gate perturbation is multiplicative and passes
through a nonlinearity; a down perturbation is additive into the residual
stream). R10 sidesteps this entirely by measuring a **held-out loss from an
actual probe** rather than any proxy. We do not have those curves locally.
**Everything numbered in §3 that relies on `proxy_err` is therefore a
hypothesis with a magnitude attached, not a measurement of loss.**

## 3. The routing signal does not exist below the expert

Routing is per-expert by construction: `expert_routed_count` has 256 entries,
and all 12 slices of an expert share one value. R10's own `build_curves`
enforces this — `mass=masses[tensor_id.expert]` and it *raises* if the three
projections of an expert declare different masses
(`allocation.py:87-91`, "probe mass drift").

Consequences, measured:

| | median over layers |
|---|---|
| sd of log(mass) across the 256 experts | 0.5018 |
| sd of log(Δloss) across all 768 tensors | 0.4713 |
| sd of log(Δloss) **within** one expert (3 projections) | 0.4150 |
| mass spread max/min per layer | 31.1× |

So within an expert the mass factor is a constant and the `mass × Δloss`
ranking collapses exactly to a `Δloss` ranking. **The per-projection decision
carries zero workload information.** It is a static property of the weights and
the calibration Hessian, fixed at encode time.

This is *not* in tension with `SELECTION-SIGNAL.md`'s "error-only ranking has
zero overlap with contribution ranking". That result is about choosing *among
experts*, where a 31× mass lever beats a 1.37× error lever. Below the expert
there is no mass lever at all, so error is all there is — and it is a lever of
the right size there (2.5× under `proxy`), because it is dominated by the
projection constant rather than by per-expert differences.

### What per-projection granularity actually buys, at equal bytes

Simulation on real measured Δ curves. Objective is R10's:
`gain = mass × (loss_floor − loss_upgraded)`, summed over a layer, with the
budget in upgrade units (gate/up/down slices are all `H×I/TP` elements, so one
unit costs the same bytes in every projection — this is why R10 can use a flat
unit cost). Reported as a fraction of the total gain available if *all* 768
tensors were upgraded. `Δ` measured as K2→K4 over all 75 layers; the K4→K5 step
on the 24 layers that have a K5 encode gives the same answers to within 1%.

Budget 78 units/layer — our live runtime's 26-K4-experts-per-layer budget:

| allocation, 78 units | `slice_nmse` | `slice_proxy_err` |
|---|---|---|
| 26 experts, all 3 projections (whole-expert, what we do today) | 24.17% | 24.12% |
| 78 freely chosen tensors (full per-projection) | 24.23% | 28.20% |
| **relative gain from per-projection** | **+0.3%** | **+14.0%** |
| layers where per-projection wins | 75/75 | 75/75 (p10 +10.4%, p90 +41.9%) |

Budget 384 units/layer — R10's 3.5 bpw budget:

| allocation, 384 units | `slice_nmse` | `slice_proxy_err` |
|---|---|---|
| 128 experts, all 3 projections | 67.60% | 69.38% |
| 384 freely chosen tensors | 67.98% | 75.41% |
| **relative gain from per-projection** | **+0.6%** | **+8.0%** |

Two sanity checks land where they should: the whole-expert 78-unit number,
24.1%, reproduces `SELECTION-SIGNAL.md`'s independently derived 24.1% ceiling;
and error-only ranking (no mass) captures only 62% of the achievable gain under
`nmse`, consistent with the expert-level finding.

### …and almost none of it needs a per-tensor measurement

Same budgets, but the allocator is only allowed a *global three-number
projection prior* (this layer's median Δ per projection) times the per-expert
mass — i.e. no per-tensor error measurement at all:

| | 78 units | 384 units |
|---|---|---|
| fraction of the per-projection optimum captured (`proxy`) | **96.3%** | **98.4%** |
| fraction captured (`nmse`) | 99.8% | 99.9% |
| set overlap with the per-projection optimum (384 units) | — | 0.888 |

Composition of the 78-tensor optimum under `proxy`: median
**{gate 6, up 9, down 64}**. Under `nmse`: **{gate 29, up 29, down 20}**.

So the *entire* per-projection story, under the metric that supports it, is
"upgrade `down_proj` first". The measured expert×projection interaction — the
only thing a static prior misses — is worth **3.7%** of an effect that is
itself worth 14%, i.e. about **0.5%** of the addressable error. That is below
the noise of anything we can measure end-to-end.

### Finer granularity does not fix the stability problem either

`ROUTING-FLATNESS.md`'s live concern was top-K churn. Since the slice score is
`mass_e × static_{e,p}`, the churn is inherited unchanged from the expert
ranking. Simulated by Poisson-resampling the routing counts at a fraction of
the calibration corpus:

| observation window | Jaccard, top-26 experts | Jaccard, top-78 slices |
|---|---|---|
| 5% of corpus | 0.9259 | 0.9259 |
| 20% of corpus | 0.9259 | 0.9500 |

Going finer buys no stability. There is no version of this where slice
granularity rescues an unstable online policy.

## 4. Runtime cost — whole-expert granularity is forced, but not by the copies

The *copy machinery* is already fine-grained. `SwapEngine.stage()` emits three
independent slab row copies per expert — `w13[0, slot]` (gate), `w13[1, slot]`
(up), `w2[slot]` (down) (`swap.py:973-981`) — and `apply()` does nothing but
`dst.copy_(src, non_blocking=True)` (`swap.py:1088-1097`). The on-disk segments
are already per-(expert, projection, rank): `layer-020.k3.safetensors` holds
12,288 tensors named
`model.layers.20.mlp.experts.{e}.{proj}.rank{r}.{suh,svh,trellis,mcg}`, at
1,192,964 bytes per (expert, projection, rank) at K3. Dropping two of the three
`dsts` entries would be a one-line change and would copy correct bytes.

It would also do **nothing**, and that is the point. Three separate things
force whole-expert granularity:

**(a) There is no slot to copy into.** Tier occupancy is per-expert by
construction. `build_tiered_maps`
(`b12x/moe/_shared/kernels/w4a16/mixed_trellis.py:1092-1122`) requires the two
tier id lists to be a **disjoint partition of `[0, total)`**, and
`global_to_combined` is one int32 per global expert. Reserving `w13[0, slot]`
for an expert's gate necessarily reserves `w13[1, slot]` and `w2[slot]` too.
`swap.py:755-760` fails closed if the partition invariant is violated.

**(b) One descriptor governs both GEMMs.** The fused kernel resolves the tier
once per expert and uses that one value for FC1 *and* FC2
(`mixed_trellis.py:299-400`):

```python
descriptor   = descriptor_map[combined_expert]
tier         = descriptor >> 8
local_expert = descriptor & 0xFF
gemm = self.tier0.fc1 if is_fc1 else self.tier0.fc2   # same descriptor, both GEMMs
```

A gate-only copy into the other tier is dead bytes: the descriptor still names
the old tier, so all three projections keep being fetched from it. No error, no
torn state, no effect. The same single combined slot indexes all four rotation
tables, including the fused `[gate_svh | up_svh | down_suh]` row of width `3I`
(`kernel.py:7207-7266`).

**(c) Gate and up cannot differ in bit-width at all.** Inside FC1 they are two
N-halves of one GEMM over one B pointer, with the plane stride hard-coded as
half the slab and the trellis bit-width a compile-time constant
(`kernel.py:4792-4816`, `4248-4264`):

```python
t256_proj      = (t256_out_n16 >= Int32(t256_half_n16)).to(Int32)  # 0=gate, 1=up
t256_plane_u32 = Int64(cute.size(b_i32_flat)) // Int64(2)
t256_tile_u32  = 8 * self.trellis_bits            # const_expr of this GEMM object
```

This is the hardest blocker and it is unfixable without restructuring the CuTe
FC1 kernel. Kernel recompilation is additionally forbidden inside a swap window
(`exl3.py:1866-1869` raises under stream capture), so any kernel work lands as a
pre-capture variant, never as a swap-time capability.

**The one split that is comparatively cheap is FC1 vs FC2** — `self.tierX.fc1`
and `self.tierX.fc2` are already distinct GEMM objects
(`mixed_trellis.py:346-352`). It needs a second descriptor map, per-GEMM tier
occupancy counts, and splitting the fused `3I` rotation row into separately
slotted pieces.

Estimated change surface for *full* per-projection tiering: ~30 call sites
across 8 gg-vllm files, ~11 in b12x, one CuTe kernel restructure, and a bump to
the persisted `fq-policy/2` schema (`bits_per_expert` `[L,E]` → `[L,E,3]`, 41
references across 6 files). For the FC1/FC2-only split: no kernel restructure,
but still the second descriptor map, the rotation-row split, the schema bump,
and the policy/telemetry rework.

### The measurement and the mechanism agree, which is unusual and worth noting

Gate and up cannot diverge in the kernel. Gate and up also turn out not to
*need* to diverge. Exact DP over a per-expert two-item knapsack (FC1 costs 2
units and upgrades gate+up together; FC2 costs 1 unit and upgrades down), on
every 5th layer:

| allocation | 78 units | 384 units |
|---|---|---|
| full per-projection optimum (`proxy`) | 1.0000 | 1.0000 |
| **FC1/FC2 split only** | **0.9996** | **0.9976** |
| whole-expert uniform | 0.8549 | 0.9250 |
| full per-projection optimum (`nmse`) | 1.0000 | 1.0000 |
| FC1/FC2 split only | 1.0000 | 1.0000 |

**The one split the runtime could plausibly support captures 99.8% of what
unlimited per-tensor freedom would give.** Separating gate from up is worth
0.04%. That entire axis can be closed.

## 5. Bottom line

**Do not pursue online tensor-level re-tiering.** Three independent reasons,
each sufficient on its own:

1. **There is no online signal below the expert.** Mass is per-expert by
   construction; the per-projection choice is a static encode-time property.
   An online loop at slice granularity would be re-deciding, every interval, a
   question whose answer never changes.
2. **The exploitable signal is one constant.** 73% of within-layer slice
   variance is the projection main effect; the per-(expert, projection)
   interaction is 3.5–9.4% and does not transfer across layers. A global
   three-number prior captures 96–98% of the per-projection optimum. The TP
   rank axis is empty (0.0000 of the variance, 1.003 median spread) —
   3,072 units/layer is 768.
3. **The runtime cost is a kernel restructure for a benefit that may be
   zero.** Gate/up cannot diverge inside the fused FC1 GEMM at all, and the
   FC1/FC2 split — which does capture 99.8% of the ceiling — still costs a
   second descriptor map, a rotation-row split, a policy schema bump, and
   ~40 call sites, for a benefit of between **+0.3% and +14%** of addressable
   error depending on a metric choice we cannot currently adjudicate.

**What is worth pursuing, if anything: an offline tier *shape*, not a finer
allocation unit.** The whole per-projection effect is "upgrade `down_proj`
first". That is a property of the *tier definition*, not of the *swap unit* —
and R10 already exploits it offline, where it belongs, with measured held-out
losses instead of a proxy. Closing task #39 as "already solved offline, and
correctly not solved online" is the accurate disposition.

### The smallest experiment that settles the remaining question

The only open question is §2's metric disagreement, and it does **not** need
any runtime change, any GPU-side swap work, or any kernel edit. Two equal-byte
checkpoints, one KLD eval each:

- **A** — 26 experts per layer upgraded on all three projections (our current
  whole-expert K3/K4 policy).
- **B** — same 78 upgrade units per layer, spent by the `proxy`-optimal
  FC1/FC2 split: median **66 `down_proj` tensors + 6 gate/up pairs**.

Both are assembled from the K3/K4 segments we already have, with the existing
assembly pipeline, at byte-identical size. The two metrics make **opposite
predictions**, which is what makes this decisive:

| | predicted gain ratio B/A |
|---|---|
| under `slice_proxy_err` | **1.169** |
| under `slice_nmse` | **0.795** |

A ±17–20% swing in addressable expert-quantization error should be visible as a
KLD difference. If B wins, the actionable result is "make the offline tier
shape down-heavy" — a policy change, not a granularity change, and no swap
engine work at all. If B loses, `proxy_err` is not comparable across
projections, the `nmse` column is the right one, per-projection allocation is
worth +0.3%, and tensor-level fungible quant is closed on measurement rather
than on argument.

Either outcome is a result. Neither requires making the swap unit smaller.

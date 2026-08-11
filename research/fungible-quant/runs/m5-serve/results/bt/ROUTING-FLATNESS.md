# ⚠️ PARTIALLY RETRACTED — read this first

**The instability half of this document was an artefact of a bug in the guard,
not a property of the model.**

`_jaccard` clamped an empty union to 1, so a layer with `n_k4 = 0` — whose
desired set is empty in every snapshot, and which therefore *cannot* churn —
scored 0.0 and was averaged in. GLM-5.2's fitted policy gives **27 of 75
layers** `n_k4 = 0`.

Measured on the same two heatmap samples:

| | |
|---|---|
| mean over all 75 layers (what the guard computed) | **0.562** |
| mean over the 48 layers that can actually swap | **0.879** |
| floor | 0.950 |

So the real interval-to-interval stability is **~0.879, not ~0.61**. That is a
tuning conversation (0.879 vs a 0.950 floor). 0.562 vs 0.950 looked like a dead
end. They are not the same situation, and I drew the wrong conclusion from the
wrong number.

**Still valid** (independent measurements, unaffected by the bug):

- routing flatness: 18,959/19,200 cells active (98.7%), top-26 carry 36.6% of
  mass, 3.6x concentration over uniform;
- the ~24% ceiling on expert-quantization error addressable at this budget;
- everything in `SELECTION-SIGNAL.md` and `R10-ALLOCATOR.md`.

**Retracted or requiring re-measurement:**

- "the desired K4 set churns ~39% between adjacent intervals" — it churns ~12%
  on layers that can hold K4;
- "the phase change produced no visible dip" — measured with the corrupted
  metric, must be redone;
- FLAT-1's "15x the sample buys +0.014" — both numbers were corrupted; the
  delta may survive, the levels do not.

Fixed in `loop.py`: empty-vs-empty is no evidence, not maximal instability, so
those rows are skipped rather than averaged in.

The original text follows unchanged, so the reasoning that led to the wrong
conclusion stays legible.

---

# The routing distribution bounds the idea, not the machinery

**Two independent measurements on a live GLM-5.2, under sustained single-domain
load, say the same thing: this model's expert routing is too flat and too
unstable for a top-K tiering policy to converge.**

That is a limit on the *premise* of expert-granular fungible quant on this
model, not on any of the code built to serve it. It needs to be in the PR.

## Measurement 1 — routing is flat

From the live activation matrix (`/fq/heatmap`, three samples, math domain):

| | |
|---|---|
| active cells | **18,959 / 19,200** — 98.7% of experts see traffic |
| top-26 mass share | **0.366** (26 = the per-layer K4 budget) |
| concentration | 3.6× over uniform |

The K4 budget covers 10.2% of experts and those carry 36.6% of the routing
mass. Real concentration — but **~63% of activations land on a K3 expert no
matter how perfectly the budget is spent.**

## Measurement 2 — the top-K set does not stabilise

The loop holds its swaps when consecutive intervals disagree about which
experts *should* be K4. Over 35 intervals of **the same domain**, at a fixed
concurrency, with no phase change:

```
step 1100  0.612    step 2100  0.592    step 3100  0.616
step 1200  0.601    step 2200  0.610    step 3200  0.616
step 1300  0.599    step 2300  0.612    step 3300  0.611
...
first 0.542   best 0.616   floor 0.950
half-series medians: 0.603 -> 0.611
VERDICT: PLATEAUED
```

It rose from 0.542 to ~0.61 in the first few intervals — a genuine warm-up —
and then sat at 0.61 for twenty-two consecutive intervals. **The desired K4 set
churns by ~39% between adjacent intervals under a stationary workload.**

## They are the same phenomenon

When 98.7% of experts are active and the top 10% carry only 36% of the mass,
the boundary between "26th expert" and "27th expert" is decided by a very small
difference in counts. Sampling noise reshuffles it every interval. Flat
distribution ⇒ unstable top-K. One measurement explains the other.

This also reframes something that looked encouraging earlier. Swaps displacing
experts with `score 0.000e+00` were displacing experts that are cold **in the
decayed window** (decay 0.95, horizon ~640 steps), not experts that are unused.
Over a longer horizon nearly everything is touched. Concentration is a property
of the observation window as much as of the router.

## What this does and does not invalidate

**Not affected.** The loader, the segment format, warm restart, the memory
preflight, the swap engine. A forced re-tier still installs weights correctly
(e1 K3→K4, e0 displaced, cardinality held, `delta_bytes 0`). Those are
mechanism results and they stand.

**Directly affected.** The claim that an *automatic* policy converges on a
better expert set under load. On this model, with this collector window, it
does not converge — it proposes a different set every interval and a stability
guard correctly refuses all of them.

## Do not fix this by lowering the floor

Dropping `jaccard_floor` from 0.95 to 0.55 would produce swaps immediately, a
moving composition table, and a demo. It would also be a policy chasing
sampling noise, re-tiering 64 experts per interval toward a target that is
different next interval — pure churn, paid for in fragment IO and quiesce
windows, with no reason to expect a quality gain.

`swap_evidence.py` was written with this warning in its docstring before any of
this was measured: *real upgrades need real router movement, and the honest way
to get it is to change what the model is asked to do rather than to lower the
guards until something moves.*

## The real options

1. **Longer collector window.** Decay 0.95 / horizon ~640 steps may simply be
   too short for a 256-way router. A longer horizon trades responsiveness for a
   stable set — and stability is the precondition for the policy to mean
   anything. Cheapest thing to try, and it is a config change.
2. **Coarser granularity.** If per-expert top-K is inherently unstable here,
   the unit of re-tiering may be wrong. Layer-level or group-level budgets
   average over more mass and would be correspondingly more stable.
3. **Pick the corpus, not the moment.** Scenario 1 (M5-H) was always the
   stronger demo: derive a policy offline from a *whole corpus*, then show the
   loader serving it. That does not need interval-to-interval stability at all,
   and the four-axis convergence work already showed corpus-level expert sets
   are meaningfully distinct.
4. **Accept it as a finding.** "Expert routing in GLM-5.2 is too flat for
   online top-K re-tiering" is a real, useful, publishable result. It is more
   valuable than a demo that moves because a guard was loosened.

## Recommendation

Try (1) because it is one env var. Then take (3) as the demo. And report the
flatness either way, because anyone building on this idea needs to measure
their model's routing distribution *first* — it decides whether the whole
approach can pay off before a single line of swap machinery is written.

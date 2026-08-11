# BT-6 — the loop re-tiers a live model on its own decision

**GLM-5.2, TP4, serving traffic. The fungible-quant loop observed routing,
decided 64 expert swaps, staged them off-step, agreed across all four ranks,
and installed them into the running model. No restart, no reload, no
assembled checkpoint.**

```
FQ live apply: staging 64 swap(s) off-step        (x4 ranks)
FQ live apply: 64 swap(s) INSTALLED               (x4 ranks)
```

## Verified from independent surfaces, not from the log line

The swap path has twice reported success it did not deliver, so the claim
rests on state read back afterwards:

| check | before | after |
|---|---|---|
| policy sha (all 4 ranks) | `66c15e3430cd504f` | **`bcf86906c9b119f5`** |
| ranks agreed | — | **true** |
| declared K4 cardinality | 2,658 | **2,658** (D1 held) |
| generation at temp 0 | coherent | **coherent** |

And the tier map itself, compared against the boot policy file rather than
against memory:

```
layer  3: 6 experts moved   e0 K4->K3, e1 K3->K4, e51 K4->K3
layer 20: 4 experts moved   e15 K4->K3, e38 K4->K3, e114 K3->K4
layer 49: 2 experts moved   e27 K4->K3, e110 K3->K4
layer 53: 2 experts moved   e0 K4->K3, e61 K3->K4
layer 60: unchanged
```

14 experts across 5 sampled layers, each a paired K3<->K4 trade. The weights
in the running model are not the weights it booted with.

## What had to be fixed to get here, in order

1. **Memory preflight** — a policy 5.6 GiB over budget was only caught after a
   62-minute load. Now arithmetic, before the engine starts.
2. **Swap engine could not see a mixed layer** — `build_swap_engine` required
   `layer_id` on a module identified by `layer_name`, so it found zero of 75
   and answered "the serve is uniform-K".
3. **Documents dropped layer 78** — rebuilt from the decision domain, which
   excludes the MTP layer, producing a policy the engine correctly refused.
4. **`plan_sha` depended on the wall clock** — provenance carries a per-rank
   timestamp, so four ranks straddling a second derived four hashes and the
   cross-check refused a correct change.
5. **Inline apply deadlocked TP** — `stage()` does hundreds of MB of
   synchronous fragment IO inside the step, per rank, so the ranks drifted and
   the shm broadcast starved.
6. **Staging buffer sized to the wrong knob** — engine `max_pairs` defaulted to
   the admin batch limit (32) against a loop cap of 64.
7. **A staged batch was discarded on plan drift** — keyed on the exact plan,
   which changes every interval: 40 staging events, 0 applies.
8. **The router-shift guard scored 27 phantom layers** — layers with
   `n_k4 = 0` have an empty desired set, cannot churn, and were being counted
   as maximally unstable. 0.562 reported where the swappable layers agreed at
   0.879.
9. **The guard's floor was mis-calibrated** — 0.95 sits inside this model's
   normal band. Recalibrated to 0.80 from the measured separation between
   within-corpus (0.88-0.95) and cross-language (0.32).

Seven of those nine produced a log that looked like progress.

## What this does NOT show

- No quality claim. Whether the re-tiered posture is *better* is BT-7, and the
  ceiling analysis says the whole K4 budget addresses ~24% of expert
  quantization error.
- Eager execution only. CUDA graphs are still unproven for live apply — that
  is what the second instance exists to test.
- No unpaired growth. Every swap here is a 1-for-1 trade; growth needs the
  reserve work in GROW-1..3.

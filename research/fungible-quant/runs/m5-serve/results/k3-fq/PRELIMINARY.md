# Preliminary convergence result — 2026-08-11, NOT the designed experiment

First convergence number measured on the real GLM-5.2, from a live serve.
Recorded because it is the first end-to-end run of the whole chain; it is
**not** the experiment the operator designed, and must not be quoted as it.

## Result

| metric | value |
|---|---|
| mean per-layer Jaccard | **0.3603** |
| pooled Jaccard | 0.3603 |
| chance floor (analytic) | 0.2652 |
| chance floor (sampled) | 0.2641 |
| human-human ceiling (3.42 vs 3.40bpw) | 0.6710 |
| **lift over chance** | **1.36x** |
| **fraction of human agreement** | **54%** |

Layers scored: 75/75. Routing observed: 7,088,550 events, 19,173 of 19,200
(layer, expert) cells non-zero.

## Why this is preliminary — three reasons, each of which should move the number

1. **Wrong corpus.** This ran on ~1 minute of a synthetic math+code prompt mix
   from `swap_evidence.py`, not the `reap_recall_calib.jsonl` (MTP78)
   calibration corpus the reference quant was actually built from. The whole
   point of the design is to replay *that* corpus. Convergence measured
   against a different traffic distribution is a lower bound at best.
2. **Ranking by hit count, not gate mass.** The collector aliases `mass` to
   `count` when no topk-weights getter is bound — verified here: the two
   arrays are byte-identical. The policy is specified to score on routing
   *mass* (sum of gate weights), which weights a confident route above a
   marginal one. We are currently ranking by raw hit frequency.
3. **Short observation window.** 18 intervals. The decayed window had barely
   warmed; late intervals see more traffic than early ones.

## What it does establish

Above chance, decisively for a first pass: 0.3603 against a 0.2652 floor, with
the sampled floor (0.2641) confirming the analytic one. Routing hotness alone,
on the wrong corpus and the wrong signal, already recovers **54% of the
agreement two humans reach with each other**. That is signal, not noise.

It also proves the full chain works on the real model: policy -> collector ->
stats dump -> offline scorer, with the composition table confirming the loop
is genuinely running rather than silently degraded.

## Next, in order of expected effect

1. Replay the MTP78 corpus (`harness/load_mtp78_corpus.py`, 12,228 rows).
2. Bind the topk-weights getter so `mass` is real gate mass, and re-score both
   ways to show the difference the signal makes.
3. Longer window.

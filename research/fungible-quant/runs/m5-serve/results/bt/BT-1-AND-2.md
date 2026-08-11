# BT-1 and BT-2 — cold boot and hot restart

**A GLM-5.2 checkpoint that exists nowhere on disk boots, serves, and restarts
from cache without touching the network.**

Both runs: TP4 on 4× RTX PRO 6000 (SM120), GG vLLM r33, `--load-format
progressive`, `policy-demo1-fitted.json` (2,658 experts at K4, 16,798 at K3),
`FQ_FAST=1` (eager, `-O0`, no graph capture — decode throughput here is NOT
comparable to a production run and is not reported).

## Results

| metric | BT-1 cold (attempt 14) | BT-2 warm (attempt 15) |
|---|---|---|
| GiB fetched from Hub | 295.8 | **0.0** |
| segments from cache | 0 | **84** |
| segments local (no fetch) | 40 | 40 |
| layers loaded | 76 | 76 |
| model load | 1616.8 s | **318.8 s** |
| time to serve | 1888 s | **350 s (5.4×)** |
| weights/rank | 79.08 GiB | 79.08 GiB |
| KV cache | 3.67 GiB | 3.67 GiB |
| posture digest | `ef47a340835a33ff` | **`ef47a340835a33ff`** |
| substitutions | 0 | 0 |

**BT-1 PASS.** Served, coherent, prompt-dependent generation at
`temperature=0` from segments plus a policy — no assembled mixed checkpoint
was ever written.

**BT-2 PASS**, on both assertions:

- **Zero bytes fetched.** This is the claim, and it is the only evidence that
  supports it. The 5.4× speedup is *reported and explicitly not asserted on*:
  a restart is faster for page cache, a warmer JIT, or a quieter Hub whether
  or not the segment cache did anything. Only the byte counter separates those.
- **Identical posture.** Same digest means the same experts came back at the
  same K. A restart that returns fast at a *different* posture has lost state.

The fungible loop also initialised on the warm run —
`FQ loop: boot policy 66c15e3430cd504f committed as current`, 75 MoE layers ×
256 experts instrumented, layer 78 correctly excluded from the decision domain
while remaining in the loader's bitrate map.

## The memory story, including a retracted claim

Attempt 11 died at `Available KV cache memory: -3.1 GiB` after a 62-minute
load. I attributed that to two causes. **One was right; one was wrong, and
the instrumentation built to exploit it is what disproved it.**

**Cause 1 — policy over budget. Confirmed.** Each K3→K4 promotion costs
exactly 1,179,648 B/rank. The seeded policy promoted 5,126 experts
(+5.63 GiB) onto a card with ~3.4 GiB spare. The preflight now computes this
before the engine starts: projected 79.06 GiB against **79.08 GiB measured**,
an error of 0.02 GiB, and the check costs 40 seconds instead of 62 minutes.

**Cause 2 — allocator residue. RETRACTED.** I claimed ~3.92 GiB of
progressive staging was stranded in the caching allocator and charged against
KV. Moving the reclaim to the correct hook — after
`process_weights_after_loading`, where the footprint genuinely is 79.17 GiB
rather than the 35.11 GiB visible earlier — produced:

```
FQ reclaim (post process_weights_after_loading):
    reserved 79.39 -> 79.39 GiB, freed 0.00 GiB; weight footprint 79.17 GiB
```

`reserved` exceeds `allocated` by 0.22 GiB. **There is no residue.** The
3.92 GiB gap came from comparing two runs that differed in configuration
(`max_model_len` 32768 vs 8192, graph capture on vs off), not in allocator
behaviour. 5.27 GiB was never a constant of this system.

Measured overhead, at util 0.95 / eager / 8192 context, against the DEVICE
budget the loader sees rather than nvidia-smi's card total:
`90.22 − 79.08 − 3.67 = 7.47 GiB`. That is the preflight's default, and a
known-answer test pins the projection to the 3.67 GiB the engine reported.
(An 8.06 GiB figure appeared here first; it used nvidia-smi's 90.81 GiB and
over-charged by 0.6 GiB.)

The relocated hook still earns its place, for the *other* thing it does: it
measures the true footprint and writes the **dense calibration (11.40 GiB per
rank)**, so future preflights project from measurement instead of falling back
to a header upper bound that over-charges one rank for all four ranks'
non-expert tensors.

Worth stating plainly: had the residue theory been right, it still would not
have rescued that boot (+0.82 GiB KV, under any usable floor). The policy was
always the binding constraint.

## What this does not show

- Decode throughput. Eager mode, deliberately; the M0 baseline of 219 tok/s at
  cc8 stands as the performance figure and was measured with graphs on.
- Live K3→K4 promotion. The loop is running but no swap has been driven yet —
  that is BT-5 and BT-6.
- Quality. BT-7 pairs a degraded arm against a converged one on GSM8K.
- KV lands at 3.67 GiB (73,024 tokens ~ 9 concurrent sequences at 8192 ctx).

  **Corrected figure.** An earlier draft of this file said the envelope
  allowed only ~2,300 promotions. That was wrong twice: it charged the
  projection against nvidia-smi's card total (90.81 GiB) instead of the device
  budget the loader actually sees (90.22 GiB), and it measured against a 4 GiB
  KV floor I had invented rather than derived — a floor that would have
  rejected the very boot that serves fine at 3.67 GiB.

  Against the device budget and a 2 GiB floor, the ceiling is **~4,200
  promotions**. The fitted policy's 2,658 clears it comfortably; the seeded
  5,126 does not. So the envelope is still set by the card rather than by the
  3.42bpw quant's shape, and that still belongs in the PR — but the headroom
  is roughly 1.6x what I first reported, and the fitted policy is not close to
  the limit.

# Where things stand — 2026-08-11 ~19:00 UTC

Written before a context compaction so the next session can resume without
re-deriving anything.

## Live state

Both vLLM instances are DOWN as of 18:55. GPUs 0-3 and 4-7 need a reap check
before any launch (`bash reap-devices.sh 0,1,2,3` — never `pgrep -f <port>`,
see below). Disk 154 G free, below the 180 G flag; the 302 GB shared segment
cache is the reason and is the asset BT-2 proved, so it is watched, not pruned.

Launch either instance with:

    cd runs/m5-serve
    nohup setsid env FQ_TAG=demo1 FQ_DEVICES_ENV=0,1,2,3 FQ_FAST=1 \
      FQ_GPUMEM=0.95 FQ_MAXLEN=8192 VLLM_FQ_LIVE_APPLY=1 \
      bash serve-demo1.sh policy-demo1-fitted.json 8100 \
      > results/demo1/serve-<name>.log 2>&1 < /dev/null & disown

`tee` into a directory that does not exist yet fails silently and you get no
log; `$PWD` in a tmux send-keys resolves to the *sender's* cwd.

## What is proven, with the artifact that proves it

| | |
|---|---|
| BT-1 cold boot from segments | 79.08 GiB, KV 3.67, coherent generation, no assembled checkpoint |
| BT-2 warm restart | **0.0 GiB fetched** vs 295.8 cold, identical posture digest, 5.4x |
| M4 swap engine | forced retier moved e1 K3→K4, e0 displaced, delta_bytes 0 |
| BT-6 live loop | **64 swaps INSTALLED**, policy_sha advanced, 14 experts verified moved vs the boot policy |
| memory preflight | projected 79.06 against 79.08 measured |

## The three retractions, so they are not re-asserted

1. **Allocator residue** — claimed ~3.92 GiB stranded; measured **0.00**, sixteen
   times. The reclaim stays only because it writes the dense calibration.
2. **Routing instability** — claimed the top-K set churns 39%; that was 27
   phantom layers (`n_k4 = 0`) scored as maximally unstable. Real figure ~0.88.
3. **"~2,300 promotion ceiling"** — used nvidia-smi's total instead of the
   device budget, and a KV floor I invented. Real ceiling ~4,200.

## Traps this box has actually sprung

- `pgrep -f 8200` killed a healthy serve on 8100: `--hf-overrides` carries 150+
  sha256 digests and one contains "8200". **Reap by device.**
- `pkill -f vllm` matches nothing — workers exec through the rootfs ld-linux
  shim.
- The serve loads `exl3_fungible` from the **rootfs**, not the source tree.
  `deploy-fq.sh` runs inside serve-demo1.sh; any other launch path must deploy.
- GPU memory release lags the kill by 10-20 s; a boot started too early reaps
  itself.
- An unconditional `export CUDA_MODULE_LOADING=EAGER` silently ate an explicit
  LAZY, so an experiment re-ran its own control and looked like confirmation.
- Verify env with `/proc/<pid>/environ`, not the launch command.

## The one number to check first

`bash runs/health/sweep.sh` now prints, per instance:

    [demo1 :8100 HEALTHY] 76/76 layers | warm: 336 cached, 160 local (0 fetched)
        apply bound on 4 rank(s), 1 install(s), last 64 swaps

`install(s)` is the artifact. Seven distinct failures this session produced a
healthy log, a moving counter, and no installed swap.

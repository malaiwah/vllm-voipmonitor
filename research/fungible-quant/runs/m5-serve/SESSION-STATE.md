# Where things stand — 2026-08-14 ~03:15 UTC

## Submission state

- GG source: `/home/mbelleau/src/gg-vllm`
- Branch: `fq/m1-stats-collector`
- Head: `14b5c5f5f` (`fq: harden live mixed-tier reconfiguration`)
- Rebased onto `origin/dev/gilded-gnosis` at `fa033bd4e`.
- Pushed to `malaiwah/vllm-voipmonitor`.
- Pull request: https://github.com/local-inference-lab/vllm/pull/307
- PR metadata verified: open, non-draft, correct base/head/body, mergeable.
- `pre-run-check` is blocked only by the repository trust gate. A maintainer
  must add `verified`, `ready`, or `ready-run-all-tests`; the author cannot add
  these labels. Request:
  https://github.com/local-inference-lab/vllm/pull/307#issuecomment-5288889824
- Post-rebase CPU checks: 224 passed / 10 deselected, plus 4/4 warmup tests.

## Live state

All vLLM instances are down. GPUs 0-3 were reaped and reported 0 MiB before the
last launch. Do not start another full boot: the second random-seed control
twice reached model loading and was killed by the instance's 640 GiB cgroup
limit (`memory.events`: `oom_kill 3`). JarvisAI credits are nearly exhausted.
The remaining graph/control/quality experiments were deliberately dropped;
use the existing eager, K2/K4 swap, convergence, and quality evidence.

Disk and model assets remain under persistent `/home`. The rootfs runtime is
ephemeral and must be restored on a replacement instance.

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

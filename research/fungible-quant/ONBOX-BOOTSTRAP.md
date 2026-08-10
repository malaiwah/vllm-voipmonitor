# On-box bootstrap (interruptible instance — disk survives, memory does not)

## Iron rules for every session and every job on this machine
1. tmux always: work inside `tmux new -s fq`; long jobs get their own window,
   launched via `nohup`/script with ON-DISK state (crash-resumable, collector-
   campaign style). Nothing important lives only in a process.
2. Commit-and-push cadence: after EVERY completed step (test green, measurement
   done, doc updated) — small commits to this branch. Results >100 MB go to HF
   (dataset repo), referenced by URL+sha256 from the committed report.
   A step that produced no committed artifact DID NOT HAPPEN.
3. Long jobs write progress to `research/fungible-quant/runs/<job>/state.json`
   every few minutes + partial outputs; every job restartable from state.
4. Secrets: fine-grained HF token in ~/.fq_env (chmod 600), never in history,
   chat, or commits.

## Bootstrap prompt (paste into a fresh `claude` inside tmux)
See below — it clones this branch and resumes from the committed state.

## First-session priorities (order matters)
0. Sanity: nvidia-smi (GPU 4 showed 100% util with no processes at rental —
   reset/reboot before any benchmark), disk free, `python3 -c "import torch"`.
1. Pre-M4 checklist (02-swap-engine.md §"Pre-M4"): 4 one-file checks against
   the pinned b12x build + occupancy<capacity check (05 §2). Minutes each.
2. Phase 0f(ii): benchmark quantize_exl3() K3 vs K4 on one expert tensor
   (sizes VLLM_FQ_ENCODE_BUDGET_PCT; 7-lazy-encode.md depends on it).
3. T1 graph-freeze test (03-testing-validation.md) — the collector's
   load-bearing assumption.
4. M0 seed: run poc/poc_slice.py logic at scale → repack brandonmusic K3 into
   attested Progressive Tensors segments; publish to HF incrementally
   (per-layer file + attestation as each completes, resumable).
5. 0c measure campaign (per-expert dKL variance) on the idle quad.
Then follow implementation/04-milestones.md (M1→M2→M3→M4) and
implementation/06-decisions-checklist.md.

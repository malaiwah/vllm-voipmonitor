# loader-v2 — Progressive Loader v2 boot artifacts

- `serve-progressive.sh` — launcher: `--load-format progressive` from
  segments + policy (no assembled checkpoint). Emits hf-overrides via
  `python -m vllm...exl3_fungible.progressive`.
- `preflight_parity.py` — CPU byte-parity check of the progressive stream
  vs the assembled fruit-mixed-042 checkpoint (run before first GPU boot).
- `boot-A.log` / `bench-A.txt` — 042 policy boot.
- `boot-B.log` / `bench-B.txt` — rotated-membership policy boot
  (policy-at-boot flexibility).
- `boot-base.log` / `bench-baseline.txt` — assembled-checkpoint reference
  boot, same session.
- `report.md` — the run report.

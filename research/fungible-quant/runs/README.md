# runs/ — on-box job state (ONBOX-BOOTSTRAP rule 3)

Session 2026-08-10, box: 8x RTX PRO 6000 (SM120), 224 cores, 1.5 TB RAM,
3.0 TB /home (persistent). We run INSIDE a managed container: no nested
runtime, no sudo, namespaces disabled — the GG stack runs from an extracted
rootfs via `gg-env/gg-run.sh` (image loader + shim python; see that script).

## Jobs / evidence directories

| Dir | What | Status 2026-08-10 |
|---|---|---|
| `dl-glm52-orig/` | zai-org/GLM-5.2 @ b4734de4 (1.51 TB) → HF cache | **done** 14:29Z |
| `dl-glm52-k3/` | brandonmusic 3.0bpw K3 @ 9297b9f1 (295 GB) | **done** 14:02Z |
| `dl-gg-images/` | ghcr puller + skopeo r33 pull + rootfs extract | r33 **extracted** to `/home/mbelleau/rootfs/gg-v20-r33`; r12 + vast appliance pulled/pulling |
| `gg-env/` | `gg-run.sh` — run anything in the extracted r33 env | **validated**: torch 2.12 cu132, vllm r33, b12x, exl3 ext, 8x SM120 |
| `drift-check/` | fresh GG/b12x/exl3 vs audited pins | **done**: GG HEAD == audit; b12x +3 dense-only commits |
| `pre-m4-checks/` | 4 adversarial source verdicts (K6 residuals) | **done**: maps PASS, launch PASS, rotations COPY, occupancy PASS (GPU half pending) |
| `encode-bench/` | 0f(ii) quantize_exl3 K3 vs K4 | **done**: 2.5 s/expert, K3≈K4, 71 experts/h @5% |
| `t1-graph-freeze/` | T1 on SIQ-Fruit proxy (GPU 5) | running |
| `m0-seed/` | source verify + repack + bulk publish | verify **green**; 76 layers repacked; upload → `malaiwah/GLM-5.2-EXL3-FQ-segments` (private) in progress |
| `0c-campaign/` | Fruit K3/K4 + dKL measurement | **pivoted** (see PIVOT.md): canonical encoder = calibration_encoder bundle in the K3 repo; adaptation in progress |

## Related artifacts elsewhere

- Segments + assembler tools: `../tools/` (fq_repack, fq_assemble,
  oci_unpack; 9 tests). Real-data round trip **byte-identical**
  (layer 30 sha a5247345).
- M1 collector code: branch `fq/m1-stats-collector` on this repo
  (GG-based tree, base e2666d9a): `exl3_fungible/stats.py` + 7 CPU tests.
- Local big dirs (not in git): `/home/mbelleau/fq-segments/`,
  `/home/mbelleau/fq-0c/`, `/home/mbelleau/rootfs/`, `/home/mbelleau/images/`,
  `/home/mbelleau/src/{gg-vllm,b12x,exllamav3}`.

#!/usr/bin/env python3
"""Generate capture_fruit.py — a minimally-patched copy of the sha-pinned
capture_b300.py from the K3 repo's calibration_encoder bundle, adapted to
the SIQ-Fruit proxy dims and SM120. Patches are exact-string (auditable);
the script fails loudly if the source drifts from the pinned sha.
"""
import hashlib
import sys
from pathlib import Path

BUNDLE = Path(
    "/home/mbelleau/.cache/huggingface/hub/"
    "models--brandonmusic--GLM-5.2-EXL3-TR3-3.0bpw/snapshots/"
    "9297b9f1d53af5c67cffa01e30cc071a1ff7144b/calibration_encoder")
# capture_b300.py is not in the rtx6kpro pin table; pin it ourselves on
# first generation (recorded in the committed report).
PATCHES = [
    ("HIDDEN = 6_144", "HIDDEN = 1_024"),
    ("NUM_LAYERS = 78", "NUM_LAYERS = 13"),
    ("CAPTURE_TP = 8", "CAPTURE_TP = 1"),
    ('worker_extension_cls="capture_b300.CaptureWorkerExtension"',
     'worker_extension_cls="capture_fruit.CaptureWorkerExtension"'),
    ("!= (10, 3):", "!= (12, 0):"),
]


def main(out_path: str) -> None:
    src = (BUNDLE / "capture_b300.py").read_text()
    print(f"source sha256: {hashlib.sha256(src.encode()).hexdigest()}")
    for old, new in PATCHES:
        if src.count(old) != 1:
            raise SystemExit(f"patch anchor not unique/found: {old!r} "
                             f"(count={src.count(old)})")
        src = src.replace(old, new)
    out = Path(out_path)
    out.write_text(src)
    print(f"wrote {out} sha256: "
          f"{hashlib.sha256(src.encode()).hexdigest()}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "capture_fruit.py")

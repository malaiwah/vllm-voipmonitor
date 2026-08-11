#!/usr/bin/env python3
"""Write an lm-eval ``--samples`` index file for a seeded random subsample.

``--limit N`` in lm-eval takes the *first* N documents. GSM8K's test split is
not shuffled, so the first 250 items are not a fair sample of the whole split
and an accuracy measured that way is not comparable to a published number. This
picks N indices with a fixed seed instead: reproducible, unbiased, and the same
items on every run so two configurations stay paired.

The population size is read from the dataset itself rather than hard-coded, so
an index can never point past the end of a split.

    ./subsample.py --task gsm8k_cot_zeroshot --n 250 --seed 1234 --out idx.json
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

# task -> (hf path, hf config, split evaluated by that task)
POPULATIONS = {
    "gsm8k": ("openai/gsm8k", "main", "test"),
    "gsm8k_cot": ("openai/gsm8k", "main", "test"),
    "gsm8k_cot_zeroshot": ("openai/gsm8k", "main", "test"),
    "gsm8k_cot_llama": ("openai/gsm8k", "main", "test"),
    "gpqa_diamond_cot_zeroshot": ("Idavidrein/gpqa", "gpqa_diamond", "train"),
    "gpqa_diamond_zeroshot": ("Idavidrein/gpqa", "gpqa_diamond", "train"),
    "gpqa_diamond_cot_n_shot": ("Idavidrein/gpqa", "gpqa_diamond", "train"),
    "gpqa_diamond_n_shot": ("Idavidrein/gpqa", "gpqa_diamond", "train"),
    "gpqa_diamond_generative_n_shot": ("Idavidrein/gpqa", "gpqa_diamond", "train"),
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", required=True, choices=sorted(POPULATIONS))
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--total", type=int, default=0,
                    help="override population size instead of loading the dataset")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    total = args.total
    if not total:
        from datasets import load_dataset
        path, config, split = POPULATIONS[args.task]
        total = len(load_dataset(path, config)[split])

    if args.n >= total:
        print(f"n={args.n} >= population {total}; run the full set instead "
              f"(drop --samples)", file=sys.stderr)
        return 2

    idx = sorted(random.Random(args.seed).sample(range(total), args.n))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({args.task: idx}))
    print(f"{args.task}: {args.n}/{total} items, seed={args.seed} -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

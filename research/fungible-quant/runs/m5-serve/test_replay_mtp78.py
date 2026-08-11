"""Regression tests for the MTP78 corpus replay driver.

The demo's whole premise is that we drive the serve with the *exact bytes*
the reference quant was calibrated on. A driver that quietly sends something
else still produces a full, plausible convergence number — it just measures
the wrong thing. These tests pin the one property that makes the number
meaningful: prompt text in == prompt text out.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import replay_mtp78 as rp  # noqa: E402

AXES = ("axis1_general", "axis2_legal", "axis3_code_agentic",
        "axis4_reasoning_termination")


@pytest.fixture()
def fake_corpus(tmp_path, monkeypatch):
    """A corpus whose rows are long, multi-line, and axis-tagged — i.e. it
    exercises every property the old --show scraper destroyed."""
    rows = []
    for i in range(8):
        rows.append({
            "axis": AXES[i % len(AXES)],
            "source": f"src:{i}",
            "text": f"row {i} line one\nline two of row {i}\n" + "x" * 400,
            "meta": {"calib_tokens": 100 + i},
        })
    p = tmp_path / "corpus.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    monkeypatch.setenv("FQ_MTP78_CORPUS", str(p))
    return rows


def test_prompts_are_the_raw_corpus_text(fake_corpus):
    """Not truncated, not prefixed with [line_no] axis source, not re-wrapped.

    The previous implementation shelled out to the loader CLI's --show mode
    and split its stdout on newlines. --show is a human display: it truncates
    to 160 chars and prepends metadata, so every replayed prompt was a stub
    with a header glued on.
    """
    prompts, _ = rp.load_corpus(None, None)
    assert prompts == [r["text"] for r in fake_corpus]


def test_multiline_rows_stay_one_prompt(fake_corpus):
    prompts, _ = rp.load_corpus(None, None)
    assert len(prompts) == len(fake_corpus)
    assert all("\n" in p for p in prompts)


def test_summary_lines_are_not_replayed_as_prompts(fake_corpus):
    """The loader CLI also prints a summary block to stdout. Scraping stdout
    sent 'total chars: 4,561' to the model as if it were calibration data."""
    prompts, _ = rp.load_corpus(None, None)
    assert not any(p.startswith(("corpus ", "sha256 ", "bytes ", "ITEM COUNT",
                                 "total chars"))
                   for p in prompts)


def test_limit_counts_prompts_not_output_lines(fake_corpus):
    prompts, meta = rp.load_corpus(None, 3)
    assert len(prompts) == 3
    assert meta["items_yielded"] == 3


def test_axis_filter_selects_only_that_axis(fake_corpus):
    prompts, _ = rp.load_corpus("axis3_code_agentic", None)
    expected = [r["text"] for r in fake_corpus
                if r["axis"] == "axis3_code_agentic"]
    assert prompts == expected


def test_metadata_carries_the_hash_for_provenance(fake_corpus):
    """main() refuses to replay on a sha mismatch, so the field must exist
    and must be the hash of the file actually read."""
    _, meta = rp.load_corpus(None, None)
    assert meta["sha256_ok"] is False   # synthetic corpus, pinned hash differs
    assert len(meta["sha256"]) == 64
    assert meta["corpus"] == "reap_recall_calib.jsonl"


def test_empty_corpus_refuses_to_replay(tmp_path, monkeypatch):
    p = tmp_path / "empty.jsonl"
    p.write_text("")
    monkeypatch.setenv("FQ_MTP78_CORPUS", str(p))
    with pytest.raises(SystemExit, match="empty corpus"):
        rp.load_corpus(None, None)


def _real_corpus_available() -> bool:
    sys.path.insert(0, str(HERE / "harness"))
    import load_mtp78_corpus as c
    try:
        c.corpus_path()
        return True
    except Exception:  # noqa: BLE001
        return False


@pytest.mark.skipif(not _real_corpus_available(),
                    reason="reap_recall_calib.jsonl not on this box")
def test_real_corpus_axis_counts_are_exact():
    """3,057 rows per axis. The scraped-stdout version reported 3,063 for the
    code axis — six summary lines counted as prompts — which is how the bug
    was visible in the first place."""
    prompts, meta = rp.load_corpus("axis3_code_agentic", None)
    assert len(prompts) == 3057
    assert meta["sha256_ok"] is True
    assert max(len(p) for p in prompts) > 160   # nothing was truncated

#!/usr/bin/env python3
"""Tests for the battle-test log parser.

The parser exists to make BT-2's claim falsifiable, so the tests concentrate
on the ways a WRONG parser would report a pass: counting per-rank lines four
times, taking the last download counter instead of the peak, or letting a
wall-clock speedup stand in for evidence that the cache was used.
"""
from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

import bt_metrics as BT

COLD = textwrap.dedent("""\
    (APIServer pid=1) INFO 08-11 15:09:31 [api.py:1] starting
    (Worker_TP0 pid=2) INFO 08-11 15:10:00 [progressive.py:1] FQ progressive L3: prefetched layer-003.k3.safetensors (100 B) from hf:repo
    (Worker_TP0 pid=2) INFO 08-11 15:10:01 [progressive.py:1] FQ progressive layer 3: tiers=((3,220),(4,36)) bits_digest=aaaa1111 tensors=768
    (Worker_TP1 pid=3) INFO 08-11 15:10:01 [progressive.py:1] FQ progressive layer 3: tiers=((3,220),(4,36)) bits_digest=aaaa1111 tensors=768
    (Worker_TP2 pid=4) INFO 08-11 15:10:01 [progressive.py:1] FQ progressive layer 3: tiers=((3,220),(4,36)) bits_digest=aaaa1111 tensors=768
    (Worker_TP3 pid=5) INFO 08-11 15:10:01 [progressive.py:1] FQ progressive layer 3: tiers=((3,220),(4,36)) bits_digest=aaaa1111 tensors=768
    (Worker_TP0 pid=2) INFO 08-11 15:10:02 [progressive.py:1] FQ progressive layer 4: tiers=((3,256),) bits_digest=bbbb2222 tensors=768
    (Worker_TP0 pid=2) INFO 08-11 15:11:00 [fragments.py:1] FQ downloads: 5 in flight, 244.6 GiB delivered, 188 MiB/s avg (recent 140 MiB/s)
    (Worker_TP1 pid=3) INFO 08-11 15:11:01 [fragments.py:1] FQ downloads: 4 in flight, 12.0 GiB delivered, 20 MiB/s avg (recent 5 MiB/s)
    (Worker_TP0 pid=2) INFO 08-11 15:20:00 [gpu.py:1] Model loading took 79.06 GiB memory and 600.0 seconds
    (Worker_TP0 pid=2) INFO 08-11 15:20:05 [pl.py:1] FQ post-load reclaim: reserved 12.00 -> 8.00 GiB (freed 4.00 GiB of allocator residue; allocated 79.06 GiB is the real weight footprint)
    (Worker_TP0 pid=2) INFO 08-11 15:21:00 [gpu.py:1] Available KV cache memory: 6.48 GiB
    (APIServer pid=1) INFO 08-11 15:22:31 [api.py:9] Application startup complete
    """)

WARM = COLD.replace(
    "FQ downloads: 5 in flight, 244.6 GiB delivered, 188 MiB/s avg",
    "FQ downloads: 0 in flight, 0.0 GiB delivered, 0 MiB/s avg",
).replace(
    "FQ downloads: 4 in flight, 12.0 GiB delivered, 20 MiB/s avg",
    "FQ downloads: 0 in flight, 0.0 GiB delivered, 0 MiB/s avg",
).replace("15:22:31", "15:12:31").replace(
    "FQ progressive L3: prefetched layer-003.k3.safetensors (100 B) from hf:repo",
    "FQ progressive L3: local /seg/layer-003.k3.safetensors (no fetch)")


def _w(tmp_path: Path, name: str, body: str) -> Path:
    p = tmp_path / name
    p.write_text(body)
    return p


def test_layers_counted_once_not_once_per_rank(tmp_path):
    """Four ranks logging layer 3 is ONE layer. Summing gives 248/76."""
    m = BT.extract(_w(tmp_path, "c.log", COLD))
    assert m["layers_loaded"] == 2


def test_download_counter_takes_the_peak_rank(tmp_path):
    """Ranks report independent totals and the log order is arbitrary; taking
    the last line seen would report 12.0 GiB for a 244.6 GiB boot."""
    m = BT.extract(_w(tmp_path, "c.log", COLD))
    assert m["gib_fetched"] == pytest.approx(244.6)


def test_extracts_the_memory_story(tmp_path):
    m = BT.extract(_w(tmp_path, "c.log", COLD))
    assert m["model_gib"] == pytest.approx(79.06)
    assert m["kv_cache_gib"] == pytest.approx(6.48)
    assert m["allocator_reclaimed_gib"] == pytest.approx(4.0)
    assert m["served"] is True
    assert m["time_to_serve_s"] == pytest.approx(780)  # 15:09:31 -> 15:22:31


def test_local_and_remote_segments_are_distinguished(tmp_path):
    cold = BT.extract(_w(tmp_path, "c.log", COLD))
    warm = BT.extract(_w(tmp_path, "w.log", WARM))
    assert cold["segments_prefetched_remote"] == 1
    assert cold["segments_local_no_fetch"] == 0
    assert warm["segments_prefetched_remote"] == 0
    assert warm["segments_local_no_fetch"] == 1


def test_warm_restart_passes_on_zero_bytes(tmp_path):
    ok, lines = BT.compare(BT.extract(_w(tmp_path, "c.log", COLD)),
                           BT.extract(_w(tmp_path, "w.log", WARM)))
    assert ok
    assert any("PASS: warm restart fetched 0.0 GiB" in x for x in lines)
    assert any("PASS: posture identical" in x for x in lines)


def test_a_fast_restart_that_refetched_still_FAILS(tmp_path):
    """The whole point. A restart 10x faster that pulled 244 GiB proves the
    network was quick, not that the cache worked."""
    fast_refetch = COLD.replace("15:22:31", "15:10:31")
    ok, lines = BT.compare(BT.extract(_w(tmp_path, "c.log", COLD)),
                           BT.extract(_w(tmp_path, "w.log", fast_refetch)))
    assert not ok
    assert any("FAIL: warm restart fetched 244.6 GiB" in x for x in lines)
    # ...and the speedup is still reported, just not treated as evidence.
    assert any("NOT asserted on" in x for x in lines)


def test_posture_drift_fails_even_with_zero_bytes(tmp_path):
    """Coming back at different tiers is lost state, however cheap the boot."""
    drifted = WARM.replace("bits_digest=aaaa1111", "bits_digest=cccc3333")
    ok, lines = BT.compare(BT.extract(_w(tmp_path, "c.log", COLD)),
                           BT.extract(_w(tmp_path, "w.log", drifted)))
    assert not ok
    assert any("posture changed" in x for x in lines)


def test_a_restart_that_never_served_fails(tmp_path):
    never = WARM.replace("Application startup complete", "still loading")
    ok, lines = BT.compare(BT.extract(_w(tmp_path, "c.log", COLD)),
                           BT.extract(_w(tmp_path, "w.log", never)))
    assert not ok
    assert any("never reached a served state" in x for x in lines)


def test_substitutions_are_surfaced_for_BT4(tmp_path):
    degraded = COLD.replace(
        "bits_digest=aaaa1111 tensors=768",
        "bits_digest=aaaa1111 tensors=768 substituted=e19:K4->K3", 1)
    m = BT.extract(_w(tmp_path, "d.log", degraded))
    assert m["substitutions"] == ["e19:K4->K3"]


def test_cache_hits_are_counted_as_their_own_origin(tmp_path):
    """A warm boot resolves segments as 'cached' — neither local nor
    prefetched. Without a counter for it the run shows zero of everything,
    which is indistinguishable from a cache that did nothing."""
    warm_cached = COLD.replace(
        "FQ progressive L3: prefetched layer-003.k3.safetensors (100 B) from hf:repo",
        "FQ progressive L3: cached layer-003.k3.safetensors")
    m = BT.extract(_w(tmp_path, "wc.log", warm_cached))
    assert m["segments_from_cache"] == 1
    assert m["segments_prefetched_remote"] == 0
    assert m["segments_local_no_fetch"] == 0


def test_served_is_true_without_a_timestamp_on_the_ready_line(tmp_path):
    """uvicorn prints 'INFO:     Application startup complete.' with no
    timestamp. Keying `served` off a parsed timestamp reported False for a
    boot that demonstrably served — the single most important field."""
    uvicorn = COLD.replace(
        "(APIServer pid=1) INFO 08-11 15:22:31 [api.py:9] Application startup complete",
        "INFO:     Application startup complete.")
    m = BT.extract(_w(tmp_path, "u.log", uvicorn))
    assert m["served"] is True
    # ...and the timing still resolves, from the last timestamped line.
    assert m["time_to_serve_s"] == pytest.approx(689)  # 15:09:31 -> 15:21:00


def test_time_to_serve_stays_none_when_the_boot_never_finished(tmp_path):
    """The fallback must not manufacture a duration for a run that died."""
    died = COLD.replace("Application startup complete", "Traceback")
    assert BT.extract(_w(tmp_path, "d.log", died))["time_to_serve_s"] is None


def test_stream_summary_does_not_invent_a_substitution(tmp_path):
    """'substituted=0, encode_queued=0' in the completion summary is NOT a
    substitution; matching it reported one on a clean boot."""
    clean = COLD + (
        "(Worker_TP0 pid=2) INFO 08-11 15:19:00 [progressive.py:1] FQ "
        "progressive stream complete: policy=x k_values=(3, 4) "
        "fragments(local=2240, substituted=0, encode_queued=0) wall=1.0s\n")
    assert BT.extract(_w(tmp_path, "s.log", clean))["substitutions"] == []


def test_missing_fields_do_not_crash_the_parser(tmp_path):
    """A boot killed mid-load has no KV line, no digest, no ready marker."""
    m = BT.extract(_w(tmp_path, "p.log", "INFO 08-11 15:00:00 [x:1] starting\n"))
    assert m["served"] is False
    assert m["kv_cache_gib"] is None
    assert m["tier_digest"] is None
    assert m["gib_fetched"] == 0.0

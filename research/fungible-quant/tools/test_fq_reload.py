"""Tests for fq_reload: policy permutation, checkpoint policy loading,
safetensors shard reading, tier partition parity with exl3."""
import json
import struct
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent))
import fq_reload  # noqa: E402

E = 16


def _policy(bits_by_layer):
    return {"schema": "fq-policy/2",
            "bits_per_expert": {str(k): list(v)
                                for k, v in bits_by_layer.items()}}


def _benefit(layers, values):
    return {L: list(values) for L in layers}


class TestPermutePolicy:
    def test_cardinality_preserved(self):
        bits = [4] * 6 + [3] * 10
        pol = _policy({3: bits})
        # benefit ascending with expert id: bottom K4 = lowest ids among K4
        ben = _benefit([3], [float(e) for e in range(E)])
        out, log = fq_reload.permute_policy(pol, ben, swaps=2)
        nb = out["bits_per_expert"]["3"]
        assert sorted(nb) == sorted(bits)
        assert nb != bits
        # demoted: two lowest-benefit K4 experts = ids 0,1
        assert log["3"]["demoted_to_k3"] == [0, 1]
        # promoted: two highest-benefit K3 experts = ids 15,14
        assert log["3"]["promoted_to_k4"] == [15, 14]
        assert nb[0] == 3 and nb[1] == 3 and nb[15] == 4 and nb[14] == 4

    def test_deterministic_ties_break_by_id(self):
        bits = [4] * 4 + [3] * 12
        pol = _policy({7: bits})
        ben = _benefit([7], [1.0] * E)
        out1, log1 = fq_reload.permute_policy(pol, ben, swaps=2)
        out2, log2 = fq_reload.permute_policy(pol, ben, swaps=2)
        assert out1["bits_per_expert"] == out2["bits_per_expert"]
        assert log1 == log2

    def test_not_enough_experts_raises(self):
        pol = _policy({3: [4] * 1 + [3] * (E - 1)})
        ben = _benefit([3], [float(e) for e in range(E)])
        with pytest.raises(ValueError):
            fq_reload.permute_policy(pol, ben, swaps=2)

    def test_swap_sets_disjoint(self):
        bits = [4 if e % 3 == 0 else 3 for e in range(E)]
        pol = _policy({5: bits})
        ben = _benefit([5], [float((e * 7) % E) for e in range(E)])
        _, log = fq_reload.permute_policy(pol, ben, swaps=2)
        d = set(log["5"]["demoted_to_k3"])
        p = set(log["5"]["promoted_to_k4"])
        assert not d & p
        assert all(bits[e] == 4 for e in d)
        assert all(bits[e] == 3 for e in p)


class TestTiersOf:
    def test_matches_exl3_partition(self):
        bits = (3, 4, 3, 4, 4, 3)
        tiers = fq_reload.tiers_of(bits)
        assert list(tiers) == [3, 4]  # sorted bits order
        assert tiers[3] == (0, 2, 5)  # ascending expert ids
        assert tiers[4] == (1, 3, 4)

    def test_single_tier(self):
        assert fq_reload.tiers_of((3, 3)) == {3: (0, 1)}


def _write_st(path: Path, tensors: dict[str, torch.Tensor]):
    hdr, off = {}, 0
    blobs = []
    for name, t in tensors.items():
        b = t.numpy().tobytes()
        dt = {torch.int16: "I16", torch.float16: "F16",
              torch.int32: "I32"}[t.dtype]
        hdr[name] = {"dtype": dt, "shape": list(t.shape),
                     "data_offsets": [off, off + len(b)]}
        off += len(b)
        blobs.append(b)
    hj = json.dumps(hdr).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hj)))
        f.write(hj)
        for b in blobs:
            f.write(b)


class TestShardReader:
    def test_roundtrip(self, tmp_path):
        t1 = torch.arange(24, dtype=torch.int16).reshape(2, 3, 4)
        t2 = torch.randn(5, dtype=torch.float32).to(torch.float16)
        _write_st(tmp_path / "s.safetensors", {"a": t1, "b": t2})
        r = fq_reload.ShardReader(tmp_path / "s.safetensors")
        assert torch.equal(r.tensor("a"), t1)
        assert torch.equal(r.tensor("b"), t2)
        # returned tensors are own copies (mutating source is safe)
        a = r.tensor("a")
        a += 1
        assert torch.equal(r.tensor("a"), t1)
        r.close()


class TestLoadCheckpointPolicy:
    def test_reads_reference(self, tmp_path):
        cfg = {"hybrid_tr3_tail": {
            "bits": "mixed", "k_values": [3, 4],
            "bits_per_expert": "tier_bitmap.json:bits_per_expert"}}
        (tmp_path / "config.json").write_text(json.dumps(cfg))
        bitmap = {"3": {"bits_per_expert": [3, 4, 3]},
                  "4": {"bits_per_expert": [4, 3, 3]},
                  "13": {"tail_tr3": [1, 2, 3]}}  # MTP-style, no field
        (tmp_path / "tier_bitmap.json").write_text(json.dumps(bitmap))
        pol = fq_reload.load_checkpoint_policy(tmp_path)
        assert pol == {3: (3, 4, 3), 4: (4, 3, 3)}

    def test_rejects_uniform(self, tmp_path):
        (tmp_path / "config.json").write_text(
            json.dumps({"hybrid_tr3_tail": {"bits": 3.0}}))
        with pytest.raises(ValueError):
            fq_reload.load_checkpoint_policy(tmp_path)


class TestLoadBenefit:
    def test_benefit_formula(self, tmp_path):
        for k, mse in ((3, [0.03, 0.02]), (4, [0.01, 0.01])):
            wd = tmp_path / f"work-k{k}-tr3"
            wd.mkdir()
            (wd / "layer-003.done.json").write_text(json.dumps({
                "layer": 3, "expert_rel_rt_mse": mse,
                "expert_routed_count": [30, 10]}))
        ben = fq_reload.load_benefit(tmp_path)
        assert ben[3] == pytest.approx([0.02 * 0.75, 0.01 * 0.25])

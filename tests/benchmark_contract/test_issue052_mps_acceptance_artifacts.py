"""Fail-closed contracts for the measured ISSUE-052 A100 artifacts."""

import json
from pathlib import Path

import pytest

ROOT = Path("benchmarks/development")
pytestmark = [
    pytest.mark.benchmark_contract,
    pytest.mark.release_gate,
    pytest.mark.skipif(
        not ROOT.exists(), reason="legacy ISSUE-052 evidence is not present"
    ),
]


def load(name):
    path = ROOT / name
    if not path.is_file():
        pytest.skip("legacy ISSUE-052 development evidence is not stored in source")
    return json.loads(path.read_text())


def test_matched_crossover_is_measured_but_does_not_claim_failed_speedup():
    payload = load("issue052_mps_crossover.json")
    assert payload["evidence_source"] == "measured_runtime"
    assert payload["workload_identity_complete"] is False
    assert payload["matched_workload"] is False
    assert payload["warmup"] >= 5
    assert payload["repetitions"] >= 20
    assert {item["world_size"] for item in payload["measurements"]} == {1, 2, 4, 8}
    assert payload["performance_gate"]["passed"] is False
    assert payload["performance_gate"]["reason"] == "workload_identity_incomplete"
    assert payload["scalability_claim_allowed"] is False
    assert payload["release_gate_allowed"] is False
    for source in payload["source_artifacts"]:
        path = Path(source["path"])
        assert (
            __import__("hashlib").sha256(path.read_bytes()).hexdigest()
            == source["sha256"]
        )


def test_capacity_baseline_is_real_oom_but_probe_cannot_promote_capacity():
    local = load("issue052_mps_capacity_1gpu.json")
    sharded = load("issue052_mps_capacity_8gpu.json")
    assert local["single_gpu_capacity_failure"] is True
    assert local["rank_records"][0]["status"] == "cuda_oom"
    assert sharded["sharded_completion"] is True
    assert len(sharded["rank_records"]) == 8
    assert all(item["useful_tensor_work"] for item in sharded["rank_records"])
    assert sharded["full_mps_materialization"] is False
    assert sharded["general_mps_training_loop"] is False
    assert sharded["capacity_gate"]["passed"] is False
    assert sharded["scalability_claim_allowed"] is False

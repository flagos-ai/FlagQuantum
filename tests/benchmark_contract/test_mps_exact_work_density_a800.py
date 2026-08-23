import json
from pathlib import Path

import pytest

pytestmark = [pytest.mark.benchmark_contract, pytest.mark.release_gate]
ROOT = Path("benchmarks/results/local")
NAME = "mps_training_128q_l3_b64_adam_{world_size}gpu_exact_prefetch_probe_20260805.json"


def _artifact(world_size: int) -> dict:
    return json.loads((ROOT / NAME.format(world_size=world_size)).read_text())


def test_exact_work_density_matrix_proves_speedup_without_release_promotion():
    artifacts = {world_size: _artifact(world_size) for world_size in (1, 2, 4, 8, 16)}
    single = artifacts[1]
    assert {item["workload_sha256"] for item in artifacts.values()} == {
        single["workload_sha256"]
    }
    assert single["workload"]["gradient_policy"] == "exact"
    assert single["workload"]["prefetch_layer_halos"] is True
    conservative_floor = {2: 1.5, 4: 2.1, 8: 2.5, 16: 2.3}
    for world_size, sharded in artifacts.items():
        assert sharded["world_size"] == world_size
        assert max(
            abs(actual - expected)
            for actual, expected in zip(sharded["losses"], single["losses"])
        ) <= 2e-6
        assert sharded["work_density"]["latency_stress_workload"] is False
        assert sharded["scalability_claim_allowed"] is False
        assert sharded["release_gate_allowed"] is False
        if world_size > 1:
            conservative_speedup = (
                single["confidence_interval_95_seconds"][0]
                / sharded["confidence_interval_95_seconds"][1]
            )
            assert conservative_speedup >= conservative_floor[world_size]
            assert sharded["distribution_semantics"] == "sharded_across_ranks"
    assert artifacts[16]["node_count"] == 2
    assert artifacts[8]["mean_seconds"] < artifacts[16]["mean_seconds"]
    assert artifacts[8]["p50_seconds"] < artifacts[16]["p50_seconds"]

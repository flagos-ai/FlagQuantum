import json
from pathlib import Path

import pytest

pytestmark = [pytest.mark.benchmark_contract, pytest.mark.release_gate]
ROOT = Path("benchmarks/results/local")


def test_chi32_alone_does_not_justify_cross_node_over_parallelization():
    name = "mps_training_128q_l4_b64_adam_{world_size}gpu_exact_bond_probe_20260805.json"
    eight = json.loads((ROOT / name.format(world_size=8)).read_text())
    sixteen = json.loads((ROOT / name.format(world_size=16)).read_text())
    assert eight["workload_sha256"] == sixteen["workload_sha256"]
    assert eight["workload"]["gradient_policy"] == "exact"
    assert eight["work_density"]["maximum_realized_bond"] == 32
    assert sixteen["work_density"]["maximum_realized_bond"] == 32
    assert max(
        abs(actual - expected)
        for actual, expected in zip(sixteen["losses"], eight["losses"])
    ) <= 2e-6
    assert eight["mean_seconds"] < sixteen["mean_seconds"]
    assert eight["p50_seconds"] < sixteen["p50_seconds"]
    assert sixteen["scalability_claim_allowed"] is False
    assert sixteen["release_gate_allowed"] is False

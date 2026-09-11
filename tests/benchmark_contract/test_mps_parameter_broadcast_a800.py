import json
from pathlib import Path

import pytest

pytestmark = [pytest.mark.benchmark_contract, pytest.mark.release_gate]


@pytest.mark.parametrize(
    ("filename", "world_size", "node_count"),
    (
        ("mps_parameter_broadcast_8xa800_20260805.json", 8, 1),
        ("mps_parameter_broadcast_16xa800_2node_20260805.json", 16, 2),
    ),
)
def test_a800_parameter_broadcast_bucket_evidence_is_complete(
    filename, world_size, node_count
):
    path = Path("benchmarks/results/local") / filename
    payload = json.loads(path.read_text())
    assert payload["schema"] == "flagquantum.mps_parameter_broadcast_ab.v1"
    assert payload["backend"] == "nccl"
    assert payload["device_name"] == "NVIDIA A800-SXM4-80GB"
    assert payload["world_size"] == world_size
    assert payload["local_world_size"] == 8
    assert payload["node_count"] == node_count
    assert payload["parameter_count"] == 1000
    assert payload["legacy_collective_count"] == 1000
    assert payload["bucketed_collective_count"] == 1
    assert payload["bucketed_collective_count"] < payload["legacy_collective_count"]
    assert payload["speedup"] >= 1.05
    assert payload["exact_parameter_parity"] is True
    assert payload["distribution_semantics"] == "requires_runtime_summary"
    assert payload["intended_distribution_semantics"] == "sharded_across_ranks"
    assert "per-rank memory evidence is not attached" in payload["scalability_blockers"]
    assert payload["parameter_owner_policy"] == "round_robin_unique_owner"
    assert payload["communication_protocol"] == "bounded_dtype_padded_all_gather_single"
    assert payload["scalability_claim_allowed"] is False
    assert payload["scalability_blockers"]
    assert payload["release_gate_allowed"] is False

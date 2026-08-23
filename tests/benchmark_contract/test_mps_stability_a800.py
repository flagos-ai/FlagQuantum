import hashlib
import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.benchmark_contract
ROOT = Path(__file__).resolve().parents[2]
WORKLOAD = ROOT / "benchmarks/mps_stability.py"


@pytest.mark.parametrize("optimizer", ("sgd", "adam"))
def test_mps_100_step_a800_soak_is_stable_and_restart_exact(optimizer):
    artifact = (
        ROOT
        / f"benchmarks/results/local/mps_stability_{optimizer}_100step_8xa800_20260805.json"
    )
    if not artifact.exists():
        pytest.skip(f"{optimizer} A800 stability evidence is not present")
    payload = json.loads(artifact.read_text())
    assert payload["schema"] == "flagquantum.mps_stability_run.v1"
    assert payload["world_size"] == 8
    assert payload["steps"] == 100
    assert payload["optimizer"] == optimizer
    assert payload["memory_stable"] is True
    assert payload["restart_equivalent"] is True
    assert payload["all_ranks_useful"] is True
    assert payload["workload_sha256"] == hashlib.sha256(
        WORKLOAD.read_bytes()
    ).hexdigest()
    records = payload["rank_records"]
    assert {record["rank"] for record in records} == set(range(8))
    assert all(len(record["memory_timeline"]) == 100 for record in records)
    assert max(record["memory_growth_bytes"] for record in records) == 0
    assert max(record["restart_parameter_error"] for record in records) == 0.0
    assert max(record["restart_loss_error"] for record in records) == 0.0

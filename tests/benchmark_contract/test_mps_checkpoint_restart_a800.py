import json
from pathlib import Path

import pytest

pytestmark = [pytest.mark.benchmark_contract, pytest.mark.release_gate]
ROOT = Path("benchmarks/results/local")


@pytest.mark.parametrize("optimizer", ("sgd", "adam"))
def test_8xa800_checkpoint_restart_development_evidence_is_complete(optimizer):
    path = ROOT / f"mps_checkpoint_restart_8xa800_{optimizer}_10cycle_20260805.json"
    payload = json.loads(path.read_text())
    assert payload["schema"] == "flagquantum.mps_checkpoint_restart_cycles.v1"
    assert payload["backend"] == "nccl"
    assert payload["device_name"] == "NVIDIA A800-SXM4-80GB"
    assert payload["world_size"] == 8
    assert payload["cycles"] == 10
    assert payload["optimizer"] == optimizer
    assert payload["distribution_semantics"] == "sharded_across_ranks"
    assert payload["loss_max_abs_error"] <= 1e-10
    assert payload["parameter_max_abs_error"] <= 1e-10
    assert payload["stale_generation_isolation"] is True
    assert len(payload["rank_summaries"]) == 8
    assert all(item["rank_useful_work"] for item in payload["rank_summaries"])
    assert all(
        item["checkpoint_storage_semantics"] == "shared_filesystem_verified"
        for item in payload["rank_summaries"]
    )
    assert payload["release_gate_allowed"] is False
    assert payload["blockers"] == ["accelerator_restart_development_evidence_only"]


@pytest.mark.parametrize("optimizer", ("sgd", "adam"))
def test_2node_16xa800_checkpoint_restart_evidence_is_complete(optimizer):
    path = (
        ROOT / f"mps_checkpoint_restart_16xa800_2node_{optimizer}_10cycle_20260805.json"
    )
    payload = json.loads(path.read_text())
    assert payload["schema"] == "flagquantum.mps_checkpoint_restart_cycles.v1"
    assert payload["backend"] == "nccl"
    assert payload["device_name"] == "NVIDIA A800-SXM4-80GB"
    assert payload["world_size"] == 16
    assert payload["n_wires"] == 16
    assert payload["cycles"] == 10
    assert payload["optimizer"] == optimizer
    assert payload["distribution_semantics"] == "sharded_across_ranks"
    assert payload["loss_max_abs_error"] <= 1e-10
    assert payload["parameter_max_abs_error"] <= 1e-10
    assert payload["stale_generation_isolation"] is True
    assert len(payload["rank_summaries"]) == 16
    assert {item["node_count"] for item in payload["rank_summaries"]} == {2}
    assert {item["local_world_size"] for item in payload["rank_summaries"]} == {8}
    assert all(item["rank_useful_work"] for item in payload["rank_summaries"])
    assert all(
        item["checkpoint_storage_semantics"] == "shared_filesystem_verified"
        for item in payload["rank_summaries"]
    )
    assert payload["release_gate_allowed"] is False
    assert payload["blockers"] == ["accelerator_restart_development_evidence_only"]

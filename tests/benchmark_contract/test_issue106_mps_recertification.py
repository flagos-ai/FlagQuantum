import json
from pathlib import Path

ARTIFACT = Path("benchmarks/development/issue106_mps_recertification.json")


def test_issue106_complete_matrix_passes_all_recertification_gates():
    payload = json.loads(ARTIFACT.read_text())
    assert payload["schema"] == "flagquantum.issue106.mps_recertification.v1"
    assert payload["predeclared_matrix_complete"]
    assert payload["baseline_audit"]["passed"]
    assert len(payload["points"]) == 12
    assert all(
        point["baseline_end_to_end"]["sample_count"] == 20
        for point in payload["points"]
    )
    assert all(
        point["optimized_end_to_end"]["sample_count"] == 20
        for point in payload["points"]
    )
    assert payload["acceptance_passed"]
    assert payload["blockers"] == []
    assert all(gate["passed"] for gate in payload["gates"].values())
    assert payload["gates"]["performance"]["promoted_points"] == 4
    assert payload["gates"]["peak_memory"]["baseline_available"]
    assert payload["performance_claim_allowed"]
    assert not payload["scalability_claim_allowed"]
    assert not payload["release_gate_allowed"]


def test_issue106_planner_refresh_keeps_regions_explicit():
    payload = json.loads(ARTIFACT.read_text())
    assert payload["planner_inputs_refreshed"]
    decisions = payload["planner_decisions"]
    optimized = [
        item for item in decisions if item["decision"] == "site_sharded_optimized"
    ]
    local = [item for item in decisions if item["decision"] == "local_fast_path"]
    capacity = [item for item in decisions if item["decision"] == "capacity_only"]
    assert len(optimized) == 4
    assert local
    assert len(capacity) == 1
    assert all(item["speedup_ci_low"] > 1.0 for item in optimized)
    assert capacity[0]["case"] == "issue092_capacity_workload"


def test_issue106_local_fast_path_and_peak_memory_do_not_regress():
    payload = json.loads(ARTIFACT.read_text())
    local = [point for point in payload["points"] if point["world_size"] == 1]
    assert all(point["speedup_mean"] >= 1 / 1.05 for point in local)
    assert max(point["peak_memory_ratio"] for point in payload["points"]) <= 1.05

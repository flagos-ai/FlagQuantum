import json
from pathlib import Path

from benchmarks.audit_results import audit_paths

REPORT = Path(
    "benchmarks/results/smoke/release_candidates/" "mps_crossover_sprint1/summary.json"
)


def test_fixed_problem_crossover_report_is_auditable_non_release_evidence():
    payload = json.loads(REPORT.read_text(encoding="utf-8"))
    audit = audit_paths([REPORT])

    assert payload["schema"] == "flagquantum.mps_crossover_sprint1_report.v1"
    assert payload["workload"]["scaling_mode"] == "fixed_problem_strong_scaling"
    assert payload["workload"]["n_wires"] == 64
    assert [point["world_size"] for point in payload["points"]] == [4, 8]
    assert all(point["source_tree_dirty"] is False for point in payload["points"])
    assert len({point["source_commit"] for point in payload["points"]}) == 1
    assert all(point["all_losses_finite"] for point in payload["points"])
    assert all(point["phase_reconciliation_passed"] for point in payload["points"])
    assert all(point["largest_realized_bond"] == 64 for point in payload["points"])
    assert payload["four_to_eight_speedup"] < 1.0
    assert payload["diagnosis"]["rank_imbalance_primary"] is False
    assert payload["diagnosis"]["boundary_growth_observed"] is True
    assert payload["diagnosis"]["primary_optimization_target"] == (
        "boundary_transport_and_partition_synchronization"
    )
    assert set(payload["scalability_blockers"]) == {
        "eight_gpu_slower_than_four_gpu",
        "nccl_visible_transport_breakdown_not_captured",
    }
    assert payload["non_release_evidence"] is True
    assert payload["scalability_claim_allowed"] is False
    assert payload["release_gate_allowed"] is False
    assert audit["invalid_count"] == 0
    assert audit["claimable_count"] == 0

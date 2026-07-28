import json
from pathlib import Path

from benchmarks.audit_results import audit_paths

REPORT = Path(
    "benchmarks/results/smoke/release_candidates/"
    "mps_communication_aware_ownership/summary.json"
)


def test_mps_ownership_report_is_auditable_non_release_evidence():
    payload = json.loads(REPORT.read_text(encoding="utf-8"))
    audit = audit_paths([REPORT])

    assert payload["schema"] == "flagquantum.mps_ownership_ab_report.v1"
    assert payload["workload"]["scaling_mode"] == "fixed_problem_strong_scaling"
    assert len(payload["points"]) == 4
    assert all(point["source_tree_dirty"] is False for point in payload["points"])
    assert len({point["source_commit"] for point in payload["points"]}) == 1
    assert payload["four_gpu_policy_speedup"] > 1
    assert payload["eight_gpu_policy_speedup"] > 1
    assert payload["eight_gpu_boundary_message_reduction"] > 0
    assert payload["eight_gpu_boundary_byte_reduction"] > 0
    assert payload["aware_four_to_eight_speedup"] < 1
    assert payload["eight_gpu_compile_setup_ratio"] > 1
    assert payload["passed"] is True
    assert payload["scalability_claim_allowed"] is False
    assert payload["release_gate_allowed"] is False
    assert audit["invalid_count"] == 0
    assert audit["claimable_count"] == 0

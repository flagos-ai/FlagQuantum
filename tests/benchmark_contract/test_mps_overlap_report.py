import json
from pathlib import Path

from benchmarks.audit_results import audit_paths

REPORT = Path(
    "benchmarks/results/smoke/release_candidates/mps_halo_overlap/summary.json"
)


def test_mps_halo_overlap_report_is_auditable_non_release_evidence():
    payload = json.loads(REPORT.read_text(encoding="utf-8"))
    audit = audit_paths([REPORT])

    assert payload["schema"] == "flagquantum.mps_halo_overlap_report.v1"
    assert payload["workload"]["scaling_mode"] == "fixed_problem_strong_scaling"
    assert len(payload["points"]) == 4
    assert all(point["source_tree_dirty"] is False for point in payload["points"])
    assert len({point["source_commit"] for point in payload["points"]}) == 1
    assert payload["four_gpu_overlap_speedup"] > 1.0
    assert payload["eight_gpu_overlap_speedup"] > 1.0
    assert payload["overlap_enabled_four_to_eight_speedup"] < 1.0
    assert payload["passed"] is True
    assert payload["scalability_claim_allowed"] is False
    assert payload["release_gate_allowed"] is False
    assert audit["invalid_count"] == 0
    assert audit["claimable_count"] == 0

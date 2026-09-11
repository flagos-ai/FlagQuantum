import json
from pathlib import Path

from benchmarks.audit_results import audit_paths

REPORT = Path("benchmarks/results/smoke/release_candidates/mps_svd_policy/summary.json")


def test_mps_svd_policy_report_preserves_accuracy_modes():
    payload = json.loads(REPORT.read_text(encoding="utf-8"))
    audit = audit_paths([REPORT])

    assert payload["schema"] == "flagquantum.mps_svd_policy_report.v1"
    assert [item["bond"] for item in payload["selections"]] == [128, 256, 512]
    assert all(item["exact_driver"] == "gesvd" for item in payload["selections"])
    assert all(item["approximate_driver"] == "gesvda" for item in payload["selections"])
    assert all(
        item["exact_residual"] <= payload["exact_residual_budget"]
        for item in payload["selections"]
    )
    assert all(
        item["approximate_residual"] <= payload["approximate_residual_budget"]
        for item in payload["selections"]
    )
    assert min(item["approximate_speedup"] for item in payload["selections"]) > 5
    assert payload["source_tree_dirty"] is False
    assert payload["passed"] is True
    assert payload["scalability_claim_allowed"] is False
    assert payload["release_gate_allowed"] is False
    assert audit["invalid_count"] == 0
    assert audit["claimable_count"] == 0

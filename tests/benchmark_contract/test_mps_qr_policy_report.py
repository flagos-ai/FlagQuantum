import json
from pathlib import Path

from benchmarks.audit_results import audit_paths

REPORT = Path("benchmarks/results/smoke/release_candidates/mps_qr_policy/summary.json")


def test_mps_qr_policy_report_preserves_no_truncation_accuracy():
    payload = json.loads(REPORT.read_text(encoding="utf-8"))
    audit = audit_paths([REPORT])

    assert payload["schema"] == "flagquantum.mps_qr_policy_report.v1"
    assert [item["bond"] for item in payload["records"]] == [128, 256, 512]
    assert payload["selected_policy"]["mode"] == "reduced"
    assert all(item["correctness_passed"] for item in payload["records"])
    assert (
        max(item["relative_reconstruction_residual"] for item in payload["records"])
        <= 2e-6
    )
    assert (
        max(item["normalized_orthogonality_error"] for item in payload["records"])
        <= 2e-6
    )
    assert max(item["transfer_overhead_fraction"] for item in payload["records"]) < 0.1
    assert payload["source_tree_dirty"] is False
    assert payload["passed"] is True
    assert payload["scalability_claim_allowed"] is False
    assert payload["release_gate_allowed"] is False
    assert audit["invalid_count"] == 0
    assert audit["claimable_count"] == 0

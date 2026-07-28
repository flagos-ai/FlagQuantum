import json
from pathlib import Path

from benchmarks.audit_results import audit_paths

REPORT = Path(
    "benchmarks/results/smoke/release_candidates/"
    "mps_gradient_buckets/summary.json"
)


def test_mps_gradient_bucket_report_is_auditable_non_release_evidence():
    payload = json.loads(REPORT.read_text(encoding="utf-8"))
    audit = audit_paths([REPORT])

    assert payload["schema"] == "flagquantum.mps_gradient_bucket_report.v1"
    assert payload["world_size"] == 8
    assert payload["parameter_count"] >= 1000
    assert payload["collective_count_reduction"] >= 0.9
    assert payload["bucket_collective_count"] <= payload["owner_bucket_count"]
    assert payload["nonowner_materialized_gradients"] == 0
    assert max(
        payload["loss_error"],
        payload["max_gradient_error"],
        payload["max_sgd_update_error"],
        payload["max_adam_update_error"],
    ) <= 2e-6
    assert payload["optimizer_resume_state_entries"] == payload["parameter_count"]
    assert payload["passed"] is True
    assert payload["scalability_claim_allowed"] is False
    assert payload["release_gate_allowed"] is False
    assert audit["invalid_count"] == 0
    assert audit["claimable_count"] == 0

from __future__ import annotations

from benchmarks.internal.evidence.issue108_mps_factorization_workspace import (
    audit_factorization_workspace,
)


def _payload(*, selected_bytes: int = 100, available: int = 200):
    ranks = []
    for rank in range(8):
        record = {
            "selected_chunk_size": 2,
            "requested_chunk_size": 8,
            "available_working_bytes": available,
            "selected_working_set_bytes": selected_bytes,
            "shape": ((1, 1024, 2, 1024), (1, 1024, 2, 1024), (1, 4, 4)),
        }
        ranks.append(
            {
                "rank": rank,
                "factorization_records": [record],
                "factorization_summary": {
                    "decision_count": 1,
                    "allocator_retry_count": 0,
                    "allocator_oom_count": 0,
                    "minimum_observed_free_bytes": 3 * (1 << 30),
                    "minimum_headroom_bytes": 2 * (1 << 30),
                    "maximum_selected_working_set_bytes": selected_bytes,
                    "selected_chunk_histogram": {"2": 1},
                },
            }
        )
    return {
        "status": "passed",
        "audit": {"passed": True},
        "workload": {"max_bond": 1024, "depths": (1, 2, 4, 8, 12, 20)},
        "environment": {"world_size": 8},
        "rank_records": ranks,
    }


def test_issue108_workspace_audit_accepts_bounded_eight_rank_evidence() -> None:
    audit = audit_factorization_workspace(_payload())
    assert audit["passed"] is True
    assert audit["blockers"] == ()


def test_issue108_workspace_audit_rejects_budget_retry_and_missing_rank() -> None:
    payload = _payload(selected_bytes=300, available=200)
    payload["rank_records"][0]["factorization_summary"]["allocator_retry_count"] = 1
    payload["rank_records"].pop()
    audit = audit_factorization_workspace(payload)
    assert audit["passed"] is False
    assert "rank_record_count_mismatch" in audit["blockers"]
    assert "rank_0_allocator_retry" in audit["blockers"]
    assert "rank_0_decision_0_budget_exceeded" in audit["blockers"]


def test_issue108_workspace_audit_rejects_before_bond_saturation_depth() -> None:
    payload = _payload()
    payload["workload"]["depths"] = (1, 2, 4)
    audit = audit_factorization_workspace(payload)
    assert audit["passed"] is False
    assert "saturation_depth_missing" in audit["blockers"]

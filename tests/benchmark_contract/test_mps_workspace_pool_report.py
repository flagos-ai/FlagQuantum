import json
from pathlib import Path

from benchmarks.audit_results import audit_paths

REPORT = Path(
    "benchmarks/results/smoke/release_candidates/" "mps_workspace_pool/summary.json"
)


def test_mps_workspace_pool_report_requires_real_rank_reuse():
    payload = json.loads(REPORT.read_text(encoding="utf-8"))
    audit = audit_paths([REPORT])

    assert payload["schema"] == "flagquantum.mps_workspace_pool_report.v1"
    assert payload["world_size"] == 8
    assert payload["workload"]["purpose"] == "stable_shape_workspace_reuse"
    assert all(record["reuse_count"] > 0 for record in payload["rank_records"])
    assert payload["total_reuse_count"] > 0
    assert payload["allocator_retry_count"] == 0
    assert payload["allocator_oom_count"] == 0
    assert payload["memory_plateau_audit"]["passed"] is True
    assert payload["source_tree_dirty"] is False
    assert payload["passed"] is True
    assert payload["scalability_claim_allowed"] is False
    assert payload["release_gate_allowed"] is False
    assert audit["invalid_count"] == 0
    assert audit["claimable_count"] == 0

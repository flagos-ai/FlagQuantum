import json
from pathlib import Path

from benchmarks.audit_results import audit_paths

REPORT = Path(
    "benchmarks/results/smoke/release_candidates/" "mps_boundary_transport/summary.json"
)


def test_mps_boundary_transport_report_is_non_release_and_auditable():
    payload = json.loads(REPORT.read_text(encoding="utf-8"))
    audit = audit_paths([REPORT])

    assert payload["schema"] == "flagquantum.mps_boundary_transport_report.v1"
    assert payload["backend"] == "nccl"
    assert payload["world_size"] == 8
    assert payload["source_tree_dirty"] is False
    assert payload["passed"] is True
    assert payload["message_count_reduction"] >= 0.5
    assert payload["logical_byte_growth"] <= 0.05
    assert payload["p2p_wait_reduction"] > 0
    assert min(payload["packed_buffer_pool_hits_by_rank"]) > 0
    assert payload["shape_mismatch_fault"] == {
        "bounded_cleanup": True,
        "shape_mismatch_passed": True,
    }
    assert payload["scalability_claim_allowed"] is False
    assert payload["release_gate_allowed"] is False
    assert audit["invalid_count"] == 0
    assert audit["claimable_count"] == 0

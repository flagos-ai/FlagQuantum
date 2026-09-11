import json
from pathlib import Path

from benchmarks.audit_results import audit_paths

REPORT = Path(
    "benchmarks/results/smoke/release_candidates/" "mps_sprint2_closeout/summary.json"
)


def test_mps_sprint2_closeout_fails_performance_gate_without_overclaiming():
    payload = json.loads(REPORT.read_text(encoding="utf-8"))
    audit = audit_paths([REPORT])

    assert payload["schema"] == "flagquantum.mps_sprint2_closeout.v1"
    assert [point["world_size"] for point in payload["points"]] == [4, 8]
    assert all(point["source_tree_dirty"] is False for point in payload["points"])
    assert len({point["source_commit"] for point in payload["points"]}) == 1
    assert payload["correctness_gate_passed"] is True
    assert payload["four_to_eight_speedup"] < payload["target_speedup"]
    assert payload["performance_gate_passed"] is False
    assert payload["scalability_claim_allowed"] is False
    assert payload["release_gate_allowed"] is False
    assert audit["invalid_count"] == 0
    assert audit["claimable_count"] == 0

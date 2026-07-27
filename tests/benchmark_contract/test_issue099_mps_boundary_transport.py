import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = ROOT / "benchmarks/development/issue099_mps_boundary_transport_matrix.json"
pytestmark = [
    pytest.mark.benchmark_contract,
    pytest.mark.skipif(
        not ARTIFACT.exists(), reason="legacy ISSUE-099 evidence is not present"
    ),
]


def test_issue099_measured_transport_matrix_passes_without_promotion():
    payload = json.loads(ARTIFACT.read_text())
    assert payload["passed"]
    assert payload["world_sizes"] == [2, 8]
    assert payload["minimum_message_count_reduction"] >= 0.40
    assert payload["maximum_logical_byte_growth"] <= 0.05
    assert payload["minimum_p2p_wait_reduction"] > 0.0
    assert payload["minimum_overlap_work_seconds"] > 0.0
    assert payload["trace_count"] == 10
    assert not payload["performance_claim_allowed"]
    assert not payload["scalability_claim_allowed"]

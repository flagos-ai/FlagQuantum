import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = ROOT / "benchmarks/development/issue098_fused_z_zz_scan_matrix.json"
pytestmark = [
    pytest.mark.benchmark_contract,
    pytest.mark.skipif(
        not ARTIFACT.exists(), reason="legacy ISSUE-098 evidence is not present"
    ),
]


def test_issue098_measured_matrix_is_complete_and_non_promotional():
    payload = json.loads(ARTIFACT.read_text())
    assert payload["passed"]
    assert payload["world_sizes"] == [1, 2, 4, 8]
    assert payload["scan_pairs_per_probe_microbatch"] == 1
    assert payload["minimum_launch_reduction"] >= 0.35
    assert payload["maximum_value_error"] <= 1e-5
    assert payload["maximum_gradient_error"] <= 2e-4
    assert not payload["performance_claim_allowed"]
    assert not payload["scalability_claim_allowed"]

import json
from pathlib import Path

import pytest

from flagquantum.testing import (
    MPSStabilityCertificationError,
    require_mps_stability,
)

ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = ROOT / "benchmarks/development/issue093_mps_stability_matrix.json"
pytestmark = [
    pytest.mark.benchmark_contract,
    pytest.mark.skipif(
        not ARTIFACT.exists(), reason="legacy ISSUE-093 evidence is not present"
    ),
]


def test_issue093_legacy_matrix_is_rejected_until_remeasured():
    payload = json.loads(ARTIFACT.read_text())
    with pytest.raises(MPSStabilityCertificationError):
        require_mps_stability(payload)

import json
from pathlib import Path

import pytest

from flagquantum.testing import (
    MPSCertificationError,
    require_mps_numerical_certification,
)

ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = ROOT / "benchmarks/development/distributed_mps_correctness_matrix.json"
pytestmark = [
    pytest.mark.benchmark_contract,
    pytest.mark.skipif(
        not ARTIFACT.exists(), reason="legacy ISSUE-091 evidence is not present"
    ),
]


def test_issue091_legacy_matrix_is_rejected_until_remeasured_with_v2_evidence():
    payload = json.loads(ARTIFACT.read_text())
    with pytest.raises(MPSCertificationError):
        require_mps_numerical_certification(payload)

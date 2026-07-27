import json
from pathlib import Path

import pytest

from flagquantum.testing import (
    MPSCapacityCertificationError,
    require_general_mps_capacity,
)

ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = ROOT / "benchmarks/development/issue092_general_mps_capacity.json"
pytestmark = [
    pytest.mark.benchmark_contract,
    pytest.mark.skipif(
        not ARTIFACT.exists(), reason="legacy ISSUE-092 evidence is not present"
    ),
]


def test_issue092_legacy_artifact_is_rejected_until_remeasured():
    payload = json.loads(ARTIFACT.read_text())
    with pytest.raises(MPSCapacityCertificationError):
        require_general_mps_capacity(payload)

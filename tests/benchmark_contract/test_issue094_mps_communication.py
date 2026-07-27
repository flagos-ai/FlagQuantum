import copy
import json
from pathlib import Path

import pytest

from flagquantum.testing import (
    MPSCommunicationCertificationError,
    require_mps_communication,
)

ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = ROOT / "benchmarks/development/issue094_mps_communication_matrix.json"
pytestmark = [
    pytest.mark.benchmark_contract,
    pytest.mark.skipif(
        not ARTIFACT.exists(), reason="legacy ISSUE-094 evidence is not present"
    ),
]


def test_issue094_measured_communication_matrix_passes_contract():
    payload = json.loads(ARTIFACT.read_text())
    require_mps_communication(payload)
    assert payload["clean_batched_microbenchmark"]["legacy_to_batched_ratio"] > 1


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("runs", 2, "unbatched_warning_count"), 1),
        (("runs", 1, "boundary_trace", 0, "communication_sequence"), 5),
        (("issue091_max_gradient_error",), 1.0),
        (("sequence_mismatch_fault", "bounded_cleanup"), False),
    ],
)
def test_issue094_contract_rejects_transport_fault(path, value):
    invalid = copy.deepcopy(json.loads(ARTIFACT.read_text()))
    target = invalid
    for part in path[:-1]:
        target = target[part]
    target[path[-1]] = value
    with pytest.raises(MPSCommunicationCertificationError):
        require_mps_communication(invalid)


def test_issue094_contract_rejects_reordered_boundary_schedule():
    invalid = copy.deepcopy(json.loads(ARTIFACT.read_text()))
    invalid["runs"][2]["boundary_trace"][0:2] = reversed(
        invalid["runs"][2]["boundary_trace"][0:2]
    )
    with pytest.raises(MPSCommunicationCertificationError):
        require_mps_communication(invalid)


def test_issue094_contract_rejects_incomplete_detailed_report():
    invalid = copy.deepcopy(json.loads(ARTIFACT.read_text()))
    invalid["runs"][0]["trace_by_rank_step"] = [{"rank": 0, "step": 0}]
    with pytest.raises(MPSCommunicationCertificationError):
        require_mps_communication(invalid)

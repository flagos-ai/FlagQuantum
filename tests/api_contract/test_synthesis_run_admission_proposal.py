"""Historical proposal status; production behavior has its own admission suite."""

import inspect
import json
from pathlib import Path

import pytest

import flagquantum as fq

pytestmark = pytest.mark.integration
ROOT = Path(__file__).resolve().parents[2]
CONTRACT = json.loads(
    (ROOT / "contracts/synthesis-program-run-admission-proposal.json").read_text()
)


def test_historical_proposal_stays_unapproved_while_implementation_has_authorization() -> (
    None
):
    assert CONTRACT["status"] == "proposed"
    assert CONTRACT["implementation_authorized"] is False
    assert CONTRACT["production_implemented"] is False
    assert CONTRACT["api_owner_approval"] is None
    assert not CONTRACT["human_trial_records"]
    admission = json.loads(
        (ROOT / "contracts/synthesis-program-run-v1-candidate.json").read_text()
    )
    assert admission["implementation_authorized"] is True
    assert admission["release_authorized"] is False
    assert not admission["human_trial_records"]
    assert str(inspect.signature(fq.run, eval_str=False)) == admission["run_signature"]
    assert admission["run_signature"] == CONTRACT["proposed_run_signature"]

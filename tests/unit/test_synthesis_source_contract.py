"""Only the explicitly reviewed source API delta may be activated."""

import copy
import json
from pathlib import Path

import pytest

from tools.public_api_snapshot import _validate_authorized_synthesis_qmod

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[2]


def contract():
    return json.loads((ROOT / "contracts/synthesis-qmod-v1-candidate.json").read_text())


def test_source_implementation_has_separate_authorization_from_historical_proposal():
    assert _validate_authorized_synthesis_qmod(contract()) == []
    proposal = json.loads(
        (ROOT / "contracts/synthesis-qmod-admission-proposal.json").read_text()
    )
    assert proposal["implementation_authorized"] is False
    assert proposal["api_owner_approval"] is None
    assert (
        contract()["authorization_source"]["proposal_merge"]
        == "c994934998eb3a6dbe983e008c8ad7fe8875bfca"
    )


@pytest.mark.parametrize(
    "change",
    [
        "signature",
        "memory",
        "report",
        "source_version",
        "error",
        "approval",
        "release",
        "trials",
        "types",
        "executor",
        "serializer",
        "proposal_hash",
        "flag_type",
        "unknown",
    ],
)
def test_source_contract_rejects_unreviewed_expansion(change):
    altered = copy.deepcopy(contract())
    if change == "signature":
        altered["synthesize_signature"] += "extra"
    elif change == "memory":
        altered["source_memory_default"] = None
    elif change == "report":
        altered["common_report_fields"].append("arithmetic_resources")
    elif change == "source_version":
        altered["source_version"] = "latest"
    elif change == "error":
        altered["source_error_type"] = "ValueError"
    elif change == "approval":
        altered["implementation_authorized"] = False
    elif change == "release":
        altered["release_authorized"] = True
    elif change == "trials":
        altered["human_trial_records"] = ["agent"]
    elif change == "types":
        altered["new_public_types"] = ["Source"]
    elif change == "executor":
        altered["new_executor"] = True
    elif change == "serializer":
        altered["serialization_change"] = True
    elif change == "proposal_hash":
        altered["proposal_contract_sha256"] = "unreviewed"
    elif change == "flag_type":
        altered["release_authorized"] = 0
    else:
        altered["extra"] = True
    assert _validate_authorized_synthesis_qmod(altered)

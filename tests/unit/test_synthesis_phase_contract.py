"""Only the explicitly authorized predicate phase addition may be activated."""

import copy
import json
from pathlib import Path

import pytest

from tools.public_api_snapshot import _validate_authorized_synthesis_phase

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[2]


def contract():
    return json.loads(
        (ROOT / "contracts/synthesis-phase-v1-candidate.json").read_text()
    )


def test_phase_implementation_has_separate_explicit_authorization():
    assert _validate_authorized_synthesis_phase(contract()) == []
    historical = json.loads(
        (ROOT / "contracts/synthesis-phase-admission-proposal.json").read_text()
    )
    assert historical["public_implementation_authorized"] is False
    assert historical["api_owner_approval"] is None
    assert contract()["authorization_source"]["request"] == "do"
    assert (
        contract()["authorization_source"]["proposal_merge"]
        == "1afca2cbc9128f070b6637f75b2818e0f6b2ae61"
    )


@pytest.mark.parametrize(
    "change",
    [
        "signature",
        "scope",
        "source",
        "report",
        "memory",
        "authorization",
        "release",
        "trials",
        "types",
        "executor",
        "serializer",
        "proposal_hash",
        "flag_type",
        "exports",
        "unknown",
    ],
)
def test_phase_contract_rejects_unreviewed_expansion(change):
    altered = copy.deepcopy(contract())
    if change == "signature":
        altered["synthesize_signature"] += "extra"
    elif change == "scope":
        altered["phase_scope"]["topology"] = "line"
    elif change == "source":
        altered["native_source_modes"].append("phase")
    elif change == "report":
        altered["phase_report_excludes"] = []
    elif change == "memory":
        altered["memory_behavior"] = "unbounded"
    elif change == "authorization":
        altered["implementation_authorized"] = False
    elif change == "release":
        altered["release_authorized"] = True
    elif change == "trials":
        altered["human_trial_records"] = ["agent"]
    elif change == "types":
        altered["new_public_types"] = ["PhaseProgram"]
    elif change == "executor":
        altered["new_executor"] = True
    elif change == "serializer":
        altered["serialization_change"] = True
    elif change == "proposal_hash":
        altered["proposal_contract_sha256"] = "unreviewed"
    elif change == "flag_type":
        altered["release_authorized"] = 0
    elif change == "exports":
        altered["namespace_exports"].append("phase")
    else:
        altered["extra"] = True
    assert _validate_authorized_synthesis_phase(altered)

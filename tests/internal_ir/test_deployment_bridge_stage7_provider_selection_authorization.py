from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
AUTHORIZATION = (
    ROOT / "contracts/deployment-bridge-stage7-provider-selection-authorization.json"
)


def _load() -> dict[str, object]:
    return json.loads(AUTHORIZATION.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_selection_authorization_binds_the_reviewed_stage7_proposal() -> None:
    authorization = _load()

    assert authorization["status"] == "selection_intent_recorded_target_not_qualified"
    assert authorization["approval_command"] == (
        "approve DEPLOYMENT-BRIDGE-STAGE7-PROVIDER-SELECTION "
        "provider=Quafu sandbox=Quafu"
    )
    for key in ("proposal", "proposal_review"):
        artifact = authorization[key]
        assert _sha256(ROOT / artifact["path"]) == artifact["sha256"]


def test_owner_labels_are_recorded_without_inventing_a_backend() -> None:
    selection = _load()["owner_supplied_selection"]

    assert selection["provider_input"] == "Quafu"
    assert selection["sandbox_input"] == "Quafu"
    assert selection["canonical_provider_namespace"] == "quafu"
    assert selection["sandbox_target_identity"] is None
    assert selection["provider_label_recorded"] is True
    assert selection["sandbox_label_recorded"] is True
    assert selection["concrete_backend_identity_supplied"] is False


def test_unverified_sandbox_and_billing_claims_fail_closed() -> None:
    authorization = _load()
    qualification = authorization["qualification"]
    decisions = authorization["decisions"]

    assert qualification["provider_identity_supported_by_authoritative_source"] is True
    assert qualification["named_quafu_sandbox_found_in_authoritative_source"] is False
    assert qualification["sandbox_only_attestation_verified"] is False
    assert qualification["non_billable_attestation_verified"] is False
    assert qualification["exact_backend_identity_verified"] is False
    allowed = {
        "owner_provider_selection_intent_accepted",
        "provider_label_recorded",
        "sandbox_label_recorded",
    }
    for name, value in decisions.items():
        assert value is (name in allowed)
    assert authorization["blockers"]


def test_inbound_adapter_does_not_qualify_outbound_transport() -> None:
    direction = _load()["architecture_direction"]

    assert direction["directions_are_equivalent"] is False
    assert direction["existing_adapter_qualifies_outbound_live_sandbox"] is False
    assert direction["external_adapter_may_contain_flagquantum_core_source"] is False

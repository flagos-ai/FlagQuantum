from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
AUTHORIZATION = ROOT / "contracts/deployment-bridge-stage6-entry-authorization.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_stage6_authorization_binds_review_and_proposal() -> None:
    authorization = json.loads(AUTHORIZATION.read_text(encoding="utf-8"))

    assert authorization["status"] == "approved"
    assert authorization["approval_command"] == "approve DEPLOYMENT-BRIDGE-STAGE6-ENTRY"
    for field in ("review_candidate", "proposal"):
        artifact = authorization[field]
        assert _sha256(ROOT / artifact["path"]) == artifact["sha256"]


def test_stage6_authorization_keeps_live_and_public_surfaces_closed() -> None:
    authorization = json.loads(AUTHORIZATION.read_text(encoding="utf-8"))
    decisions = authorization["decisions"]
    scope = authorization["allowed_scope"]

    allowed = {
        "stage6_private_connector_contracts_authorized",
        "stage6_scripted_transport_and_anonymous_fixtures_authorized",
        "stage6_independent_performance_baseline_authorized",
    }
    assert all(decisions[name] is True for name in allowed)
    assert all(
        value is False for name, value in decisions.items() if name not in allowed
    )
    assert scope["explicit_offline_invocation_only"] is True
    assert scope["finite_immutable_scripted_transport_only"] is True
    assert scope["credential_reference_is_non_resolvable"] is True
    assert scope["maximum_certification"] == "offline_scripted_evidence"
    assert (
        scope[
            "existing_observation_rehearsal_readiness_deployment_provider_runtime_and_shadow_sources_mutable"
        ]
        is False
    )

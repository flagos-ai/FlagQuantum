from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
AUTHORIZATION = ROOT / "contracts/deployment-bridge-stage4-entry-authorization.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_stage4_authorization_binds_review_and_proposal() -> None:
    authorization = json.loads(AUTHORIZATION.read_text(encoding="utf-8"))

    assert authorization["status"] == "approved"
    assert authorization["approval_command"] == "approve DEPLOYMENT-BRIDGE-STAGE4-ENTRY"
    for field in ("review_candidate", "proposal"):
        artifact = authorization[field]
        assert _sha256(ROOT / artifact["path"]) == artifact["sha256"]


def test_stage4_authorization_keeps_every_real_action_surface_closed() -> None:
    authorization = json.loads(AUTHORIZATION.read_text(encoding="utf-8"))
    decisions = authorization["decisions"]
    scope = authorization["allowed_scope"]

    allowed = {
        "stage4_private_offline_rehearsal_authorized",
        "stage4_anonymous_synthetic_fixtures_authorized",
        "stage4_independent_performance_baseline_authorized",
    }
    assert all(decisions[name] is True for name in allowed)
    assert all(
        value is False for name, value in decisions.items() if name not in allowed
    )
    assert scope["explicit_offline_invocation_only"] is True
    assert scope["simulated_enum_actions_only"] is True
    assert scope["maximum_outcome"] == "offline_rehearsal_passed"
    assert (
        scope[
            "existing_readiness_deployment_provider_runtime_and_shadow_sources_mutable"
        ]
        is False
    )

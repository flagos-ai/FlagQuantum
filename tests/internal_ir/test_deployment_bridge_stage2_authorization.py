from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
AUTHORIZATION = ROOT / "contracts/deployment-bridge-stage2-entry-authorization.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_stage2_authorization_binds_review_and_proposal() -> None:
    authorization = json.loads(AUTHORIZATION.read_text(encoding="utf-8"))

    assert authorization["status"] == "approved"
    assert authorization["approval_command"] == "approve DEPLOYMENT-BRIDGE-STAGE2-ENTRY"
    for field in ("review_candidate", "proposal"):
        artifact = authorization[field]
        assert _sha256(ROOT / artifact["path"]) == artifact["sha256"]


def test_stage2_authorization_keeps_every_activation_surface_closed() -> None:
    authorization = json.loads(AUTHORIZATION.read_text(encoding="utf-8"))
    decisions = authorization["decisions"]
    scope = authorization["allowed_scope"]

    allowed = {
        "stage2_private_implementation_authorized",
        "stage2_ephemeral_target_ir_and_artifact_authorized",
        "stage2_canonical_encoder_refactor_authorized",
        "stage2_performance_baseline_authorized",
    }
    assert all(decisions[name] is True for name in allowed)
    assert all(
        value is False for name, value in decisions.items() if name not in allowed
    )
    assert scope["explicit_offline_invocation_only"] is True
    assert scope["ephemeral_candidate_objects_only"] is True
    assert scope["existing_deployment_provider_and_runtime_sources_mutable"] is False

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
AUTHORIZATION = ROOT / "contracts/deployment-bridge-stage1-entry-authorization.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_stage1_authorization_binds_review_and_proposal() -> None:
    authorization = json.loads(AUTHORIZATION.read_text(encoding="utf-8"))

    assert authorization["status"] == "approved"
    assert authorization["approval_command"] == "approve DEPLOYMENT-BRIDGE-STAGE1-ENTRY"
    for field in ("review_candidate", "proposal"):
        artifact = authorization[field]
        assert _sha256(ROOT / artifact["path"]) == artifact["sha256"]


def test_stage1_authorization_keeps_every_activation_surface_closed() -> None:
    authorization = json.loads(AUTHORIZATION.read_text(encoding="utf-8"))
    decisions = authorization["decisions"]
    scope = authorization["allowed_scope"]

    assert decisions["stage1_implementation_authorized"] is True
    assert decisions["stage1_performance_baseline_authorized"] is True
    for name, value in decisions.items():
        if name not in {
            "stage1_implementation_authorized",
            "stage1_performance_baseline_authorized",
        }:
            assert value is False
    assert scope["module"] == "flagquantum/_compiler/deployment_compatibility.py"
    assert scope["explicit_offline_invocation_only"] is True
    assert scope["existing_deployment_and_provider_sources_mutable"] is False

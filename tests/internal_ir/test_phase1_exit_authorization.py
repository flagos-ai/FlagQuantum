from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
AUTHORIZATION = ROOT / "contracts/ir-phase1-exit-authorization.json"


def _authorization() -> dict[str, object]:
    return json.loads(AUTHORIZATION.read_text(encoding="utf-8"))


def test_phase1_exit_authorization_binds_review_candidate() -> None:
    authorization = _authorization()
    candidate = ROOT / authorization["review_candidate"]["path"]

    assert authorization["status"] == "approved"
    assert authorization["approval_command"] == "approve IR-PHASE1-EXIT"
    assert hashlib.sha256(candidate.read_bytes()).hexdigest() == (
        authorization["review_candidate"]["sha256"]
    )
    assert all(authorization["formal_signoffs"].values())


def test_phase1_completion_does_not_authorize_phase2_or_public_paths() -> None:
    authorization = _authorization()
    decisions = authorization["decisions"]

    assert decisions["phase1_technical_evidence_accepted"] is True
    assert decisions["phase1_complete"] is True
    assert decisions["phase2_authorized"] is False
    assert decisions["public_ir_api_authorized"] is False
    assert decisions["public_pass_api_authorized"] is False
    assert decisions["default_path_change_authorized"] is False
    assert decisions["production_path_change_authorized"] is False
    assert decisions["provider_codegen_authorized"] is False
    assert decisions["optimizer_migration_authorized"] is False
    assert authorization["accepted_scope"]["public_exports"] == []
    assert authorization["accepted_scope"]["default_path_enabled"] is False

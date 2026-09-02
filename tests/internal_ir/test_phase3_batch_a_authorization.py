from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
AUTHORIZATION = ROOT / "contracts/ir-phase3-entry-batch-a-authorization.json"


def _authorization() -> dict[str, object]:
    return json.loads(AUTHORIZATION.read_text(encoding="utf-8"))


def test_phase3_batch_a_authorization_binds_reviewed_candidate() -> None:
    record = _authorization()
    candidate = ROOT / record["review_candidate"]["path"]

    assert record["status"] == "approved"
    assert record["approval_command"] == "approve IR-PHASE3-ENTRY-BATCH-A"
    assert (
        hashlib.sha256(candidate.read_bytes()).hexdigest()
        == record["review_candidate"]["sha256"]
    )


def test_phase3_batch_a_authorization_keeps_later_surfaces_closed() -> None:
    record = _authorization()
    decisions = record["decisions"]
    scope = record["allowed_scope"]

    assert decisions["phase3_entry_authorized"] is True
    assert decisions["phase3_batch_a_authorized"] is True
    assert decisions["batch_a_performance_baseline_authorized"] is True
    assert decisions["batch_a_performance_budget_approved"] is False
    assert decisions["batch_b_through_h_authorized"] is False
    assert decisions["target_ir_or_artifact_implementation_authorized"] is False
    assert decisions["runtime_adapter_or_conformance_authorized"] is False
    assert decisions["shadow_or_default_integration_authorized"] is False
    assert decisions["provider_or_remote_submission_authorized"] is False
    assert decisions["real_backend_identity_authorized"] is False
    assert decisions["public_api_change_authorized"] is False
    assert decisions["legacy_retirement_authorized"] is False
    assert decisions["phase3_exit_authorized"] is False
    assert scope["public_exports"] == []
    assert scope["default_path_enabled"] is False
    assert scope["provider_fields_allowed"] is False

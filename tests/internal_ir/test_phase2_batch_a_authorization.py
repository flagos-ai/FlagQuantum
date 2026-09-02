from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
AUTHORIZATION = ROOT / "contracts/ir-phase2-entry-batch-a-authorization.json"


def _authorization() -> dict[str, object]:
    return json.loads(AUTHORIZATION.read_text(encoding="utf-8"))


def test_phase2_batch_a_authorization_binds_reviewed_candidate() -> None:
    record = _authorization()
    candidate = ROOT / record["review_candidate"]["path"]

    assert record["status"] == "approved"
    assert record["approval_command"] == "approve IR-PHASE2-ENTRY-BATCH-A"
    assert (
        hashlib.sha256(candidate.read_bytes()).hexdigest()
        == record["review_candidate"]["sha256"]
    )


def test_phase2_batch_a_authorization_keeps_later_surfaces_closed() -> None:
    record = _authorization()
    decisions = record["decisions"]
    scope = record["allowed_scope"]

    assert decisions["phase2_entry_authorized"] is True
    assert decisions["phase2_batch_a_authorized"] is True
    assert decisions["phase2_performance_budget_approved"] is True
    assert decisions["phase2_later_batches_authorized_to_start"] is False
    assert decisions["public_api_change_authorized"] is False
    assert decisions["default_path_change_authorized"] is False
    assert decisions["legacy_compiler_retirement_authorized"] is False
    assert decisions["provider_codegen_authorized"] is False
    assert scope["public_exports"] == []
    assert scope["default_path_enabled"] is False

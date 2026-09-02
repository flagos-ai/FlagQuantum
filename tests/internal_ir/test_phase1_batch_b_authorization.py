from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
AUTHORIZATION = ROOT / "contracts/ir-phase1-batch-b-authorization.json"


def test_batch_b_authorization_binds_batch_a_review() -> None:
    record = json.loads(AUTHORIZATION.read_text(encoding="utf-8"))
    candidate = ROOT / record["review_candidate"]["path"]

    assert record["status"] == "approved"
    assert record["approval_command"] == "approve IR-PHASE1-BATCH-A-EXIT-BATCH-B"
    assert (
        hashlib.sha256(candidate.read_bytes()).hexdigest()
        == record["review_candidate"]["sha256"]
    )


def test_batch_b_authorization_keeps_later_scopes_closed() -> None:
    record = json.loads(AUTHORIZATION.read_text(encoding="utf-8"))
    decisions = record["decisions"]

    assert decisions["batch_a_owner_accepted"] is True
    assert decisions["batch_b_authorized"] is True
    assert decisions["batch_c_authorized"] is False
    assert decisions["phase1_complete"] is False
    assert decisions["phase2_authorized"] is False
    assert decisions["public_api_change_authorized"] is False
    assert decisions["default_runtime_change_authorized"] is False
    assert record["allowed_scope"]["public_exports"] == []

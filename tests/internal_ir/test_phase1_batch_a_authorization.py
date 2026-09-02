from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
AUTHORIZATION = ROOT / "contracts/ir-phase1-batch-a-authorization.json"


def test_batch_a_authorization_binds_reviewed_candidate() -> None:
    record = json.loads(AUTHORIZATION.read_text(encoding="utf-8"))
    candidate = ROOT / record["review_candidate"]["path"]

    assert record["status"] == "approved"
    assert record["approval_command"] == "approve IR-PHASE0-EXIT-PHASE1-BATCH-A"
    assert (
        hashlib.sha256(candidate.read_bytes()).hexdigest()
        == record["review_candidate"]["sha256"]
    )


def test_batch_a_authorization_keeps_public_and_later_phases_closed() -> None:
    record = json.loads(AUTHORIZATION.read_text(encoding="utf-8"))
    decisions = record["decisions"]

    assert decisions["phase0_exited"] is True
    assert decisions["phase1_batch_a_authorized"] is True
    assert decisions["phase1_later_batches_authorized_to_start"] is False
    assert decisions["phase2_authorized"] is False
    assert decisions["public_api_change_authorized"] is False
    assert decisions["default_runtime_change_authorized"] is False
    assert record["allowed_scope"]["public_exports"] == []
    assert record["allowed_scope"]["default_path_enabled"] is False

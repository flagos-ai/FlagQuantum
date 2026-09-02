from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
AUTHORIZATION = ROOT / "contracts/ir-phase1-batch-d-authorization.json"


def test_batch_d_authorization_binds_batch_c_review() -> None:
    record = json.loads(AUTHORIZATION.read_text(encoding="utf-8"))
    candidate = ROOT / record["review_candidate"]["path"]

    assert record["status"] == "approved"
    assert record["approval_command"] == "approve IR-PHASE1-BATCH-C-EXIT-BATCH-D"
    assert (
        hashlib.sha256(candidate.read_bytes()).hexdigest()
        == record["review_candidate"]["sha256"]
    )


def test_batch_d_authorization_keeps_lossy_provider_and_public_paths_closed() -> None:
    record = json.loads(AUTHORIZATION.read_text(encoding="utf-8"))
    decisions = record["decisions"]

    assert decisions["batch_c_owner_accepted"] is True
    assert decisions["batch_d_authorized"] is True
    assert decisions["batch_e_authorized"] is False
    assert decisions["phase1_complete"] is False
    assert decisions["phase2_authorized"] is False
    assert decisions["provider_codegen_authorized"] is False
    assert record["allowed_scope"]["lossy_mode"] is False
    assert record["allowed_scope"]["public_exports"] == []
    assert record["allowed_scope"]["default_path_enabled"] is False

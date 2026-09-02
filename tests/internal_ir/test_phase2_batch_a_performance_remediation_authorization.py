from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
AUTHORIZATION = (
    ROOT / "contracts/ir-phase2-batch-a-performance-remediation-authorization.json"
)


def test_remediation_authorization_binds_candidate_and_blocker() -> None:
    record = json.loads(AUTHORIZATION.read_text(encoding="utf-8"))

    assert record["status"] == "approved"
    assert record["approval_command"] == (
        "approve IR-PHASE2-BATCH-A-PERFORMANCE-REMEDIATION"
    )
    for field in ("review_candidate", "blocker"):
        artifact = ROOT / record[field]["path"]
        assert hashlib.sha256(artifact.read_bytes()).hexdigest() == (
            record[field]["sha256"]
        )


def test_remediation_authorization_keeps_exit_and_budget_approval_closed() -> None:
    record = json.loads(AUTHORIZATION.read_text(encoding="utf-8"))
    decisions = record["decisions"]

    assert decisions["remediation_implementation_authorized"] is True
    assert decisions["identity_cache_authorized"] is True
    assert decisions["entry_exit_verification_authorized"] is True
    assert decisions["fused_private_pipeline_authorized"] is True
    assert decisions["successor_budget_candidate_authorized"] is True
    assert decisions["original_budget_change_authorized"] is False
    assert decisions["successor_budget_approved"] is False
    assert decisions["batch_a_exit_authorized"] is False
    assert decisions["batch_b_start_authorized"] is False
    assert decisions["public_api_change_authorized"] is False
    assert decisions["default_path_change_authorized"] is False

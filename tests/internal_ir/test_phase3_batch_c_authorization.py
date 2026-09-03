from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
AUTHORIZATION = ROOT / "contracts/ir-phase3-batch-b-exit-batch-c-authorization.json"


def _authorization() -> dict[str, object]:
    return json.loads(AUTHORIZATION.read_text(encoding="utf-8"))


def test_phase3_batch_c_authorization_binds_reviewed_candidate() -> None:
    record = _authorization()
    candidate = ROOT / record["review_candidate"]["path"]

    assert record["status"] == "approved"
    assert record["approval_command"] == "approve IR-PHASE3-BATCH-B-EXIT-BATCH-C"
    assert (
        hashlib.sha256(candidate.read_bytes()).hexdigest()
        == record["review_candidate"]["sha256"]
    )


def test_phase3_batch_c_authorization_keeps_later_surfaces_closed() -> None:
    decisions = _authorization()["decisions"]

    assert decisions["batch_b_exit_authorized"] is True
    assert decisions["batch_c_entry_authorized"] is True
    assert decisions["batch_c_performance_baseline_authorized"] is True
    assert decisions["batch_c_performance_budget_approved"] is False
    assert decisions["batch_d_through_h_authorized"] is False
    assert decisions["new_emitter_authorized"] is False
    assert decisions["runtime_adapter_or_execution_authorized"] is False
    assert decisions["conformance_or_shadow_authorized"] is False
    assert decisions["provider_or_remote_submission_authorized"] is False
    assert decisions["real_backend_identity_authorized"] is False
    assert decisions["public_api_change_authorized"] is False
    assert decisions["default_path_change_authorized"] is False
    assert decisions["legacy_retirement_authorized"] is False
    assert decisions["phase3_exit_authorized"] is False

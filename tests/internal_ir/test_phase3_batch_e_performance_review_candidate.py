from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
CANDIDATE = (
    ROOT / "contracts/ir-phase3-batch-e-performance-budget-review-candidate.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _candidate() -> dict[str, object]:
    return json.loads(CANDIDATE.read_text(encoding="utf-8"))


def test_batch_e_performance_candidate_binds_reviewed_evidence() -> None:
    candidate = _candidate()

    assert candidate["status"] == "ready_for_owner_approval"
    assert candidate["approval_command"] == (
        "approve IR-PHASE3-BATCH-E-PERFORMANCE-BUDGET"
    )
    for relative_path, expected_hash in candidate["reviewed_artifacts"].items():
        assert _sha256(ROOT / relative_path) == expected_hash


def test_batch_e_performance_candidate_keeps_later_work_closed() -> None:
    decisions = _candidate()["proposed_decisions"]

    assert decisions["batch_e_performance_budget_approved"] is True
    assert decisions["batch_e_machine_gate_authorized"] is True
    assert decisions["batch_e_exit_authorized"] is False
    assert decisions["batch_f_shadow_harness_authorized"] is False
    assert decisions["provider_sdk_or_remote_submission_authorized"] is False
    assert decisions["real_backend_identity_authorized"] is False
    assert decisions["shadow_or_default_integration_authorized"] is False
    assert decisions["public_api_change_authorized"] is False
    assert decisions["default_path_change_authorized"] is False
    assert decisions["legacy_retirement_authorized"] is False

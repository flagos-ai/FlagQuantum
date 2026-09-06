from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
CANDIDATE = (
    ROOT / "contracts/ir-phase3-batch-d-performance-budget-review-candidate.json"
)
CANDIDATE_SHA256 = "df55dce89e898b76660ee8d7d3379ef99a7d525099a4251c087f6a310508fea1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _candidate() -> dict[str, object]:
    return json.loads(CANDIDATE.read_text(encoding="utf-8"))


def test_batch_d_performance_candidate_preserves_review_record() -> None:
    candidate = _candidate()

    assert _sha256(CANDIDATE) == CANDIDATE_SHA256
    assert candidate["status"] == "ready_for_owner_approval"
    assert candidate["approval_command"] == (
        "approve IR-PHASE3-BATCH-D-PERFORMANCE-BUDGET"
    )
    authorization = candidate["authorization"]
    assert _sha256(ROOT / authorization["path"]) == authorization["sha256"]


def test_batch_d_performance_candidate_keeps_later_work_closed() -> None:
    decisions = _candidate()["proposed_decisions"]

    assert decisions["batch_d_performance_budget_approved"] is True
    assert decisions["batch_d_machine_gate_authorized"] is True
    assert decisions["batch_d_exit_authorized"] is False
    assert decisions["batch_e_provider_conformance_authorized"] is False
    assert decisions["provider_sdk_or_remote_submission_authorized"] is False
    assert decisions["credentials_authorized"] is False
    assert decisions["real_backend_identity_authorized"] is False
    assert decisions["public_api_change_authorized"] is False
    assert decisions["default_path_change_authorized"] is False
    assert decisions["legacy_retirement_authorized"] is False

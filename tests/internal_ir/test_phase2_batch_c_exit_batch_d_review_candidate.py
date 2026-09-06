from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
CANDIDATE = ROOT / "contracts/ir-phase2-batch-c-exit-batch-d-review-candidate.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_batch_c_exit_candidate_binds_approved_evidence() -> None:
    candidate = json.loads(CANDIDATE.read_text(encoding="utf-8"))

    assert candidate["status"] == "ready_for_owner_approval"
    for field in ("authorization", "performance_validation"):
        artifact = candidate[field]
        assert _sha256(ROOT / artifact["path"]) == artifact["sha256"]
    assert all(candidate["evidence"].values())


def test_batch_d_candidate_is_private_and_later_gates_stay_closed() -> None:
    candidate = json.loads(CANDIDATE.read_text(encoding="utf-8"))
    decisions = candidate["proposed_decisions"]

    assert decisions["batch_c_exit_approved"] is True
    assert decisions["batch_d_private_emitters_authorized"] is True
    assert decisions["batch_d_performance_budget_approved"] is False
    assert decisions["provider_sdk_or_remote_submission_authorized"] is False
    assert decisions["credentials_or_backend_ids_authorized"] is False
    assert decisions["public_api_change_authorized"] is False
    assert decisions["default_path_change_authorized"] is False
    assert decisions["legacy_retirement_authorized"] is False
    assert decisions["phase2_exit_authorized"] is False
    assert candidate["performance_risk"]["tracked"] is True
    assert candidate["performance_risk"]["automatic_budget_relaxation_allowed"] is (
        False
    )
    assert candidate["approval_command"] == ("approve IR-PHASE2-BATCH-C-EXIT-BATCH-D")

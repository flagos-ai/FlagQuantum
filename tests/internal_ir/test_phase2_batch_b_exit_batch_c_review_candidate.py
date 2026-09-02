from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
CANDIDATE = ROOT / "contracts/ir-phase2-batch-b-exit-batch-c-review-candidate.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_batch_b_exit_candidate_binds_approved_evidence_and_artifacts() -> None:
    candidate = json.loads(CANDIDATE.read_text(encoding="utf-8"))

    assert candidate["status"] == "ready_for_owner_approval"
    for field in ("authorization", "performance_validation"):
        artifact = candidate[field]
        assert _sha256(ROOT / artifact["path"]) == artifact["sha256"]
    for relative_path, expected_hash in candidate["reviewed_artifacts"].items():
        assert _sha256(ROOT / relative_path) == expected_hash
    assert all(candidate["evidence"].values())


def test_batch_c_candidate_is_private_and_later_gates_stay_closed() -> None:
    candidate = json.loads(CANDIDATE.read_text(encoding="utf-8"))
    decisions = candidate["proposed_decisions"]

    assert decisions["batch_b_exit_approved"] is True
    assert decisions["batch_c_private_placement_routing_authorized"] is True
    assert decisions["batch_c_performance_budget_approved"] is False
    assert decisions["emitter_or_provider_codegen_authorized"] is False
    assert decisions["remote_submission_authorized"] is False
    assert decisions["public_api_change_authorized"] is False
    assert decisions["default_path_change_authorized"] is False
    assert decisions["legacy_retirement_authorized"] is False
    assert decisions["phase2_exit_authorized"] is False
    assert candidate["approval_command"] == ("approve IR-PHASE2-BATCH-B-EXIT-BATCH-C")

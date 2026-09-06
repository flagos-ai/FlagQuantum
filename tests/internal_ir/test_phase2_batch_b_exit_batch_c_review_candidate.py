from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
CANDIDATE = ROOT / "contracts/ir-phase2-batch-b-exit-batch-c-review-candidate.json"
SUCCESSOR = ROOT / "contracts/ir-phase2-batch-c-authorized-artifact-successor.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_batch_b_exit_candidate_binds_approved_evidence() -> None:
    candidate = json.loads(CANDIDATE.read_text(encoding="utf-8"))

    assert candidate["status"] == "ready_for_owner_approval"
    for field in ("authorization", "performance_validation"):
        artifact = candidate[field]
        assert _sha256(ROOT / artifact["path"]) == artifact["sha256"]
    assert all(candidate["evidence"].values())


def test_batch_c_successor_is_authorized_without_public_or_default_change() -> None:
    successor = json.loads(SUCCESSOR.read_text(encoding="utf-8"))
    authorization = successor["authorization"]

    assert successor["status"] == "authorized_artifact_successor"
    assert _sha256(ROOT / authorization["path"]) == authorization["sha256"]
    assert successor["predecessor_review_semantics_changed"] is False
    assert successor["public_or_default_path_changed"] is False


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

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
CANDIDATE = (
    ROOT / "contracts/ir-phase2-batch-b-performance-budget-review-candidate.json"
)
SUCCESSOR = (
    ROOT / "contracts/ir-phase2-batch-b-performance-budget-artifact-successor.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_batch_b_performance_review_binds_authorization_and_artifacts() -> None:
    candidate = json.loads(CANDIDATE.read_text(encoding="utf-8"))
    successor = json.loads(SUCCESSOR.read_text(encoding="utf-8"))
    successor_candidate = successor["candidates"][str(CANDIDATE.relative_to(ROOT))]
    authorization = candidate["authorization"]

    assert candidate["status"] == "ready_for_owner_approval"
    assert _sha256(ROOT / authorization["path"]) == authorization["sha256"]
    for relative_path, expected_hash in candidate["reviewed_artifacts"].items():
        actual_hash = _sha256(ROOT / relative_path)
        if actual_hash == expected_hash:
            continue
        transition = successor_candidate["artifact_transitions"][relative_path]
        assert transition["predecessor_sha256"] == expected_hash
        assert transition["successor_sha256"] == actual_hash


def test_performance_gate_successor_is_authorized_and_scope_closed() -> None:
    successor = json.loads(SUCCESSOR.read_text(encoding="utf-8"))
    authorization = successor["authorization"]

    assert successor["status"] == "authorized_artifact_successor"
    assert _sha256(ROOT / authorization["path"]) == authorization["sha256"]
    assert successor["predecessor_review_semantics_changed"] is False
    assert successor["public_or_default_path_changed"] is False


def test_batch_b_budget_review_cannot_authorize_exit_or_routing() -> None:
    candidate = json.loads(CANDIDATE.read_text(encoding="utf-8"))
    decisions = candidate["proposed_decisions"]

    assert decisions["batch_b_performance_budget_approved"] is True
    assert decisions["batch_b_machine_gate_authorized"] is True
    assert decisions["batch_b_exit_authorized"] is False
    assert decisions["batch_c_routing_authorized"] is False
    assert decisions["public_api_change_authorized"] is False
    assert decisions["default_path_change_authorized"] is False
    assert decisions["legacy_retirement_authorized"] is False
    assert candidate["approval_command"] == (
        "approve IR-PHASE2-BATCH-B-PERFORMANCE-BUDGET"
    )

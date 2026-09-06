from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
CANDIDATE = ROOT / "contracts/ir-phase3-batch-g-exit-batch-h-review-candidate.json"
SUCCESSOR = ROOT / "contracts/ir-phase3-runtime-bindings-successor-authorization.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _candidate() -> dict[str, object]:
    return json.loads(CANDIDATE.read_text(encoding="utf-8"))


def _assert_artifact_hash(relative_path: str, predecessor_hash: str) -> None:
    actual_hash = _sha256(ROOT / relative_path)
    if actual_hash == predecessor_hash:
        return
    successor = json.loads(SUCCESSOR.read_text(encoding="utf-8"))
    transition = successor["artifact_transitions"].get(relative_path)
    assert transition is not None, f"unauthorized artifact successor: {relative_path}"
    assert transition == {
        "predecessor_sha256": predecessor_hash,
        "successor_sha256": actual_hash,
    }


def test_batch_g_exit_candidate_binds_authorization_and_artifacts() -> None:
    candidate = _candidate()

    assert candidate["status"] == "ready_for_owner_approval"
    authorization = candidate["authorization"]
    assert _sha256(ROOT / authorization["path"]) == authorization["sha256"]
    for relative_path, expected_hash in candidate["reviewed_artifacts"].items():
        _assert_artifact_hash(relative_path, expected_hash)


def test_batch_g_exit_evidence_is_complete_and_non_activating() -> None:
    evidence = _candidate()["batch_g_exit_evidence"]

    assert evidence["authorized_scope_implemented"] is True
    assert evidence["current_boundary_inventory_exact"] is True
    assert evidence["ownership_and_identity_mapping_complete"] is True
    assert evidence["compatibility_matrix_complete"] is True
    assert evidence["migration_and_rollback_gated"] is True
    assert evidence["canary_readiness_prerequisites_complete"] is True
    assert evidence["runtime_or_canary_activated"] is False
    assert evidence["default_path_zero_impact"] is True
    assert evidence["public_api_changed"] is False
    assert evidence["known_batch_g_exit_blockers"] == []


def test_batch_h_scope_is_aggregate_review_only() -> None:
    candidate = _candidate()
    scope = candidate["batch_h_scope_if_approved"]
    decisions = candidate["proposed_decisions"]

    assert scope["feature"] == "aggregate Phase 3 exit review"
    assert scope["batches_a_through_g_evidence_required"] is True
    assert scope["three_target_family_boundary_evidence_required"] is True
    assert scope["all_approved_performance_gates_required"] is True
    assert scope["runtime_implementation_authorized"] is False
    assert scope["canary_activation_authorized"] is False
    assert scope["phase4_entry_authorized"] is False
    assert decisions["batch_g_exit_authorized"] is True
    assert decisions["batch_h_entry_authorized"] is True
    assert decisions["phase3_exit_authorized"] is False
    assert candidate["approval_command"] == ("approve IR-PHASE3-BATCH-G-EXIT-BATCH-H")

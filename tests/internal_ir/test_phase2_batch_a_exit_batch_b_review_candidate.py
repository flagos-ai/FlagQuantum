from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
CANDIDATE = ROOT / "contracts/ir-phase2-batch-a-exit-batch-b-review-candidate.json"
SUCCESSOR = ROOT / "contracts/ir-phase2-batch-a-machine-gate-test-successor.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_batch_a_exit_candidate_binds_authorizations_and_artifacts() -> None:
    candidate = json.loads(CANDIDATE.read_text(encoding="utf-8"))
    successor = json.loads(SUCCESSOR.read_text(encoding="utf-8"))
    transitions = successor["candidates"][str(CANDIDATE.relative_to(ROOT))][
        "artifact_transitions"
    ]

    assert candidate["status"] == "ready_for_owner_approval"
    for authorization in candidate["authorization_chain"].values():
        assert _sha256(ROOT / authorization["path"]) == authorization["sha256"]
    for relative_path, expected_hash in candidate["reviewed_artifacts"].items():
        actual_hash = _sha256(ROOT / relative_path)
        if actual_hash == expected_hash:
            continue
        transition = transitions[relative_path]
        assert transition["predecessor_sha256"] == expected_hash
        assert transition["successor_sha256"] == actual_hash


def test_batch_a_gate_successor_preserves_budget_and_workload() -> None:
    successor = json.loads(SUCCESSOR.read_text(encoding="utf-8"))
    authorization = successor["authorization"]

    assert successor["status"] == "authorized_artifact_successor"
    assert _sha256(ROOT / authorization["path"]) == authorization["sha256"]
    assert successor["budget_or_workload_changed"] is False
    assert successor["public_or_default_path_changed"] is False


def test_batch_a_exit_evidence_is_complete_and_has_no_known_blocker() -> None:
    candidate = json.loads(CANDIDATE.read_text(encoding="utf-8"))
    evidence = candidate["batch_a_exit_evidence"]

    assert evidence["authorized_scope_implemented"] is True
    assert evidence["structural_parity"] is True
    assert evidence["state_and_expectation_parity"] is True
    assert evidence["trainable_gradient_parity"] is True
    assert evidence["failure_semantics_fail_closed"] is True
    assert evidence["identity_determinism"] is True
    assert evidence["successor_performance_budget_passed"] is True
    assert evidence["default_path_zero_impact"] is True
    assert evidence["public_api_changed"] is False
    assert evidence["legacy_compiler_retained"] is True
    assert evidence["known_blockers"] == []


def test_batch_b_candidate_scope_is_private_and_narrow() -> None:
    candidate = json.loads(CANDIDATE.read_text(encoding="utf-8"))
    scope = candidate["batch_b_scope_if_approved"]

    assert scope["feature"] == "private target gate-set decomposition"
    assert scope["routing_authorized"] is False
    assert scope["emitter_authorized"] is False
    assert scope["provider_codegen_authorized"] is False
    assert scope["public_api_change_authorized"] is False
    assert scope["default_path_change_authorized"] is False
    assert scope["legacy_retirement_authorized"] is False
    assert scope["batch_c_through_f_authorized"] is False
    assert candidate["approval_command"] == ("approve IR-PHASE2-BATCH-A-EXIT-BATCH-B")

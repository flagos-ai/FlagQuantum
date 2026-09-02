from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
CANDIDATE = ROOT / "contracts/ir-phase1-batch-e-review-candidate.json"
SUCCESSOR = ROOT / "contracts/ir-phase2-batch-a-authorized-artifact-successor.json"
REMEDIATION_SUCCESSOR = (
    ROOT / "contracts/ir-phase2-batch-a-performance-remediation-artifact-successor.json"
)


def _candidate() -> dict[str, object]:
    return json.loads(CANDIDATE.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _assert_artifact_binding(relative_path: str, expected_hash: str) -> None:
    actual_hash = _sha256(ROOT / relative_path)
    if actual_hash == expected_hash:
        return
    remediation = json.loads(REMEDIATION_SUCCESSOR.read_text(encoding="utf-8"))
    remediation_candidate = remediation["candidates"].get(
        str(CANDIDATE.relative_to(ROOT))
    )
    transition = (
        remediation_candidate["artifact_transitions"].get(relative_path)
        if remediation_candidate
        else None
    )
    if transition is not None:
        assert transition["predecessor_sha256"] == expected_hash
        assert transition["successor_sha256"] == actual_hash
        return
    successor = json.loads(SUCCESSOR.read_text(encoding="utf-8"))
    candidate = successor["candidates"][str(CANDIDATE.relative_to(ROOT))]
    transition = candidate["artifact_transitions"][relative_path]
    assert candidate["sha256"] == _sha256(CANDIDATE)
    assert transition["predecessor_sha256"] == expected_hash
    assert transition["successor_sha256"] == actual_hash
    assert successor["predecessor_review_semantics_changed"] is False
    assert successor["public_or_default_path_changed"] is False


def test_batch_e_candidate_binds_authorization_and_artifacts() -> None:
    candidate = _candidate()
    authorization = candidate["authorization"]

    assert _sha256(ROOT / authorization["path"]) == authorization["sha256"]
    for section in ("implementation_artifacts", "test_artifacts"):
        for relative_path, expected_hash in candidate[section].items():
            _assert_artifact_binding(relative_path, expected_hash)


def test_batch_e_candidate_records_private_semantics_preserving_scope() -> None:
    candidate = _candidate()
    scope = candidate["scope"]
    boundary = candidate["capability_boundary"]

    assert candidate["status"] == "ready_for_batch_e_review"
    assert scope["semantic_transformations"] is False
    assert scope["optimizer_migration"] is False
    assert scope["differential_bridge"] is False
    assert scope["public_exports_added"] == []
    assert scope["default_path_enabled"] is False
    assert boundary["can_reject_phase1_pass_contract_violations"] is True
    assert boundary["can_execute_semantic_optimization_passes"] is False


def test_batch_e_candidate_pauses_before_batch_f_and_phase2() -> None:
    candidate = _candidate()
    decision = candidate["exit_decision"]

    assert decision["batch_e_technical_complete"] is True
    assert decision["batch_e_owner_accepted"] is False
    assert decision["batch_f_authorized"] is False
    assert decision["phase1_complete"] is False
    assert decision["phase2_authorized"] is False
    assert candidate["next_review_command"] == (
        "approve IR-PHASE1-BATCH-E-EXIT-BATCH-F"
    )

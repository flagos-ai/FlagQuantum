from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
CANDIDATE = ROOT / "contracts/ir-phase1-exit-review-candidate.json"
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


def test_phase1_exit_candidate_binds_authorizations_and_artifacts() -> None:
    candidate = _candidate()

    for authorization in candidate["authorization"].values():
        assert _sha256(ROOT / authorization["path"]) == authorization["sha256"]
    for section in ("implementation_artifacts", "test_artifacts"):
        for relative_path, expected_hash in candidate[section].items():
            _assert_artifact_binding(relative_path, expected_hash)


def test_phase1_exit_candidate_binds_public_contract_corpus_and_budget() -> None:
    candidate = _candidate()
    public = candidate["public_contract"]
    corpus = candidate["corpus"]
    performance = candidate["performance"]

    assert _sha256(ROOT / public["stable_api_manifest"]["path"]) == (
        public["stable_api_manifest"]["sha256"]
    )
    assert _sha256(ROOT / public["baseline"]["path"]) == (public["baseline"]["sha256"])
    assert _sha256(ROOT / corpus["path"]) == corpus["sha256"]
    assert _sha256(ROOT / performance["budget_path"]) == (performance["budget_sha256"])
    assert _sha256(ROOT / performance["successor_attestation"]["path"]) == (
        performance["successor_attestation"]["sha256"]
    )
    assert public["root_exports_added"] == []
    assert public["default_run_plan_path_changed"] is False


def test_phase1_exit_candidate_records_complete_technical_evidence() -> None:
    candidate = _candidate()

    assert candidate["status"] == "ready_for_owner_review"
    assert len(candidate["implemented_work_packages"]) == 9
    assert candidate["corpus"]["canonical_opcodes_covered"] == 35
    assert candidate["differential"]["global_or_environment_switch"] is False
    assert candidate["differential"]["autograd_binding_identity_preserved"] is True
    assert candidate["performance"]["all_gate_counts_passed"] is True
    assert candidate["known_blockers"] == []
    assert (
        candidate["capability_boundary"]["phase1_static_profile_technically_complete"]
        is True
    )


def test_phase1_exit_candidate_pauses_before_final_acceptance_and_phase2() -> None:
    candidate = _candidate()
    decision = candidate["formal_exit_decision"]

    assert decision["technical_evidence_complete"] is True
    assert decision["api_owner_accepted"] is False
    assert decision["compiler_owner_accepted"] is False
    assert decision["runtime_owner_accepted"] is False
    assert decision["training_owner_accepted"] is False
    assert decision["phase1_complete"] is False
    assert decision["phase2_authorized"] is False
    assert candidate["next_review_command"] == "approve IR-PHASE1-EXIT"
    assert candidate["capability_boundary"]["phase2_features"] is False

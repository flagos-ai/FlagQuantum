from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
CANDIDATE = ROOT / "contracts/ir-phase2-entry-batch-a-review-candidate.json"
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


def test_phase2_candidate_binds_every_reviewed_artifact() -> None:
    candidate = _candidate()

    assert candidate["status"] == "ready_for_owner_approval"
    for relative_path, expected_hash in candidate["reviewed_artifacts"].items():
        _assert_artifact_binding(relative_path, expected_hash)


def test_phase2_candidate_is_narrow_and_cannot_authorize_itself() -> None:
    candidate = _candidate()
    scope = candidate["batch_a_scope"]

    assert scope["allowed_root"] == "flagquantum/_compiler"
    assert scope["public_exports"] == []
    assert scope["default_path_enabled"] is False
    assert scope["legacy_compiler_retained"] is True
    assert scope["later_batches_authorized"] is False
    assert all(value is False for value in candidate["authorization"].values())
    assert all(value is False for value in candidate["formal_signoffs"].values())


def test_phase2_candidate_requires_exact_approval_semantics() -> None:
    candidate = _candidate()
    text = candidate["approval_text"]

    assert candidate["approval_command"] == "approve IR-PHASE2-ENTRY-BATCH-A"
    assert "performance budget" in text
    assert "Batch A private static canonicalization only" in text
    assert "does not change Stable Core" in text
    assert "does not authorize Batch B through F" in text
    assert "legacy compiler retirement" in text


def test_phase2_budget_is_pending_and_excludes_routing_emitters() -> None:
    candidate = _candidate()
    budget = candidate["performance_budget"]
    payload = json.loads((ROOT / budget["path"]).read_text(encoding="utf-8"))

    assert budget["status"] == "pending_owner_approval"
    assert budget["public_sla"] is False
    assert budget["routing_or_emitter_budget"] is False
    assert payload["approval"]["approved"] is False
    assert "topology placement and routing" in payload["separate_baselines_required"]
    assert "QCIS emission" in payload["separate_baselines_required"]

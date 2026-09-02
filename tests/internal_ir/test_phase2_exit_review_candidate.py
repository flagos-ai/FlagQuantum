from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
CANDIDATE = ROOT / "contracts/ir-phase2-exit-review-candidate.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_phase2_exit_candidate_binds_entry_final_public_and_implementation() -> None:
    candidate = json.loads(CANDIDATE.read_text(encoding="utf-8"))

    assert candidate["status"] == "ready_for_owner_review"
    for field in ("entry_authorization", "final_authorization"):
        artifact = candidate[field]
        assert _sha256(ROOT / artifact["path"]) == artifact["sha256"]
    for artifact in candidate["public_contract"].values():
        if isinstance(artifact, dict) and "path" in artifact:
            assert _sha256(ROOT / artifact["path"]) == artifact["sha256"]
    for artifact in candidate["performance_validations"].values():
        assert _sha256(ROOT / artifact["path"]) == artifact["sha256"]
    for relative_path, expected_hash in candidate["implementation_artifacts"].items():
        assert _sha256(ROOT / relative_path) == expected_hash


def test_phase2_exit_candidate_is_technical_private_completion_only() -> None:
    candidate = json.loads(CANDIDATE.read_text(encoding="utf-8"))
    decision = candidate["formal_exit_decision"]
    scope = candidate["accepted_scope_if_approved"]

    assert decision["technical_evidence_complete"] is True
    assert decision["phase2_complete"] is False
    assert decision["phase3_authorized"] is False
    assert all(
        decision[name] is False
        for name in (
            "api_owner_accepted",
            "compiler_owner_accepted",
            "runtime_owner_accepted",
            "training_owner_accepted",
        )
    )
    assert scope["implementation_visibility"] == "private internal"
    assert scope["public_exports"] == []
    assert scope["default_path_enabled"] is False
    assert scope["provider_or_remote_execution_enabled"] is False
    assert scope["legacy_retired"] is False
    assert candidate["next_review_command"] == "approve IR-PHASE2-EXIT"

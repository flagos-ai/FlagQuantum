from __future__ import annotations

import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
CANDIDATE = ROOT / "contracts/ir-phase0-exit-phase1-review-candidate.json"


def _candidate() -> dict[str, object]:
    return json.loads(CANDIDATE.read_text(encoding="utf-8"))


def test_review_candidate_cannot_authorize_itself() -> None:
    candidate = _candidate()

    assert candidate["status"] == "ready_for_owner_approval"
    assert candidate["technical_review"]["completed"] is True
    assert all(value is False for value in candidate["formal_signoffs"].values())
    assert all(value is False for value in candidate["authorization"].values())
    assert candidate["remaining_required_actions"] == [
        "compiler owner formal signoff",
        "runtime owner formal signoff",
        "training owner formal signoff",
        "API owner explicit Phase 0 exit approval",
        "API owner explicit Phase 1 implementation authorization",
    ]


def test_review_candidate_limits_phase1_to_internal_batch_a() -> None:
    candidate = _candidate()
    scope = candidate["phase1_scope"]

    assert scope["allowed_root"] == "flagquantum/_compiler"
    assert scope["public_exports"] == []
    assert scope["default_path_enabled"] is False
    assert scope["first_batch"].startswith("A:")
    assert candidate["authorization"]["phase2_authorized"] is False
    assert candidate["authorization"]["public_api_change_authorized"] is False


def test_review_candidate_preserves_exact_approval_semantics() -> None:
    candidate = _candidate()
    text = candidate["approval_text"]

    assert candidate["approval_command"] == "approve IR-PHASE0-EXIT-PHASE1-BATCH-A"
    assert "accept the compiler, runtime, and training technical review" in text
    assert "Phase 0 exit" in text
    assert "P1-001 through P1-009" in text
    assert "does not change Stable Core" in text
    assert "authorize Phase 2" in text
    assert "batch A only" in text

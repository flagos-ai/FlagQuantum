from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
CANDIDATE = ROOT / "contracts/ir-phase3-batch-e-exit-batch-f-review-candidate.json"
CANDIDATE_SHA256 = "1bb9d05d938ba95548f54212ca4abc18af4f7bc296f2fe1391bfac04b5040079"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _candidate() -> dict[str, object]:
    return json.loads(CANDIDATE.read_text(encoding="utf-8"))


def test_batch_e_exit_candidate_preserves_record_and_authorizations() -> None:
    candidate = _candidate()

    assert _sha256(CANDIDATE) == CANDIDATE_SHA256
    assert candidate["status"] == "ready_for_owner_approval"
    for authorization in candidate["authorization_chain"].values():
        assert _sha256(ROOT / authorization["path"]) == authorization["sha256"]


def test_batch_e_exit_evidence_is_complete_and_has_no_known_blocker() -> None:
    evidence = _candidate()["batch_e_exit_evidence"]

    assert evidence["authorized_scope_implemented"] is True
    assert evidence["identical_three_family_conformance"] is True
    assert evidence["complete_identity_continuity"] is True
    assert evidence["compatibility_and_privacy_fail_closed"] is True
    assert evidence["anonymous_deterministic_fixtures"] is True
    assert evidence["approved_performance_budget_passed"] is True
    assert evidence["default_path_zero_impact"] is True
    assert evidence["public_api_changed"] is False
    assert evidence["known_blockers"] == []


def test_batch_f_candidate_scope_is_explicit_private_and_narrow() -> None:
    candidate = _candidate()
    scope = candidate["batch_f_scope_if_approved"]
    decisions = candidate["proposed_decisions"]

    assert scope["feature"] == "private explicit opt-in shadow comparison harness"
    assert scope["legacy_authoritative"] is True
    assert scope["kill_switch_required"] is True
    assert scope["privacy_safe_evidence_required"] is True
    assert scope["default_or_environment_activation_authorized"] is False
    assert scope["background_or_production_shadow_authorized"] is False
    assert scope["telemetry_export_authorized"] is False
    assert scope["public_api_change_authorized"] is False
    assert scope["batch_g_through_h_authorized"] is False
    assert decisions["batch_e_exit_authorized"] is True
    assert decisions["batch_f_entry_authorized"] is True
    assert decisions["batch_g_through_h_authorized"] is False
    assert candidate["approval_command"] == ("approve IR-PHASE3-BATCH-E-EXIT-BATCH-F")

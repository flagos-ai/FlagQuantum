from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
AUTHORIZATION = ROOT / "contracts/ir-phase2-exit-authorization.json"


def _authorization() -> dict[str, object]:
    return json.loads(AUTHORIZATION.read_text(encoding="utf-8"))


def test_phase2_exit_authorization_binds_review_candidate() -> None:
    authorization = _authorization()
    candidate = ROOT / authorization["review_candidate"]["path"]

    assert authorization["status"] == "approved"
    assert authorization["approval_command"] == "approve IR-PHASE2-EXIT"
    assert hashlib.sha256(candidate.read_bytes()).hexdigest() == (
        authorization["review_candidate"]["sha256"]
    )
    assert all(authorization["formal_signoffs"].values())


def test_phase2_completion_does_not_authorize_integration_or_public_paths() -> None:
    authorization = _authorization()
    decisions = authorization["decisions"]
    scope = authorization["accepted_scope"]

    assert decisions["phase2_technical_evidence_accepted"] is True
    assert decisions["phase2_complete"] is True
    for name, value in decisions.items():
        if name not in {"phase2_technical_evidence_accepted", "phase2_complete"}:
            assert value is False
    assert scope["implementation_visibility"] == "private internal"
    assert scope["public_exports"] == []
    assert scope["default_path_enabled"] is False
    assert scope["provider_or_remote_execution_enabled"] is False
    assert scope["legacy_retired"] is False
    assert scope["rollback_requires_user_migration"] is False


def test_phase2_exit_records_performance_jitter_without_relaxation() -> None:
    observations = _authorization()["accepted_observations"]

    assert observations["batch_f_five_sample_host_jitter_recorded"] is True
    assert observations["approved_budget_relaxed"] is False
    assert observations["public_performance_claim_authorized"] is False

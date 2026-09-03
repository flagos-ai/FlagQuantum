from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
AUTHORIZATION = ROOT / "contracts/ir-phase3-exit-authorization.json"


def _authorization() -> dict[str, object]:
    return json.loads(AUTHORIZATION.read_text(encoding="utf-8"))


def test_phase3_exit_authorization_binds_review_candidate() -> None:
    authorization = _authorization()
    candidate = ROOT / authorization["review_candidate"]["path"]

    assert authorization["status"] == "approved"
    assert authorization["approval_command"] == "approve IR-PHASE3-EXIT"
    assert hashlib.sha256(candidate.read_bytes()).hexdigest() == (
        authorization["review_candidate"]["sha256"]
    )
    assert all(authorization["formal_signoffs"].values())


def test_phase3_completion_does_not_authorize_activation_or_phase4() -> None:
    authorization = _authorization()
    decisions = authorization["decisions"]
    scope = authorization["accepted_scope"]

    assert decisions["phase3_technical_evidence_accepted"] is True
    assert decisions["phase3_complete"] is True
    for name, value in decisions.items():
        if name not in {"phase3_technical_evidence_accepted", "phase3_complete"}:
            assert value is False
    assert scope["implementation_visibility"] == "private internal"
    assert scope["public_exports"] == []
    assert scope["default_path_enabled"] is False
    assert scope["provider_or_remote_execution_enabled"] is False
    assert scope["canary_enabled"] is False
    assert scope["deployment_migrated"] is False
    assert scope["legacy_retired"] is False
    assert scope["rollback_requires_user_migration"] is False


def test_phase3_exit_records_synthetic_boundary_and_proposal_limits() -> None:
    observations = _authorization()["accepted_observations"]

    assert observations["local_runtime_plan_conformance"] is True
    assert observations["synthetic_qasm_conformance"] is True
    assert observations["synthetic_non_qasm_conformance"] is True
    assert observations["real_provider_or_qpu_claimed"] is False
    assert observations["batch_a_through_f_private_performance_budgets_passed"] is True
    assert observations["public_performance_sla_authorized"] is False
    assert observations["phase1_pytest_plugin_instrumentation_jitter_recorded"] is True
    assert observations["approved_performance_budgets_relaxed"] is False
    assert observations["batch_g_is_proposal_only"] is True
    assert observations["deployment_compatibility_bridge_implemented"] is False

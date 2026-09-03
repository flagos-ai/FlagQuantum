from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
CANDIDATE = (
    ROOT / "contracts/ir-phase3-batch-f-performance-budget-review-candidate.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _candidate() -> dict[str, object]:
    return json.loads(CANDIDATE.read_text(encoding="utf-8"))


def test_batch_f_performance_candidate_binds_reviewed_evidence() -> None:
    candidate = _candidate()

    assert candidate["status"] == "ready_for_owner_approval"
    assert candidate["approval_command"] == (
        "approve IR-PHASE3-BATCH-F-PERFORMANCE-BUDGET"
    )
    for relative_path, expected_hash in candidate["reviewed_artifacts"].items():
        assert _sha256(ROOT / relative_path) == expected_hash


def test_batch_f_performance_candidate_keeps_later_work_closed() -> None:
    decisions = _candidate()["proposed_decisions"]

    assert decisions["batch_f_performance_budget_approved"] is True
    assert decisions["batch_f_machine_gate_authorized"] is True
    assert decisions["batch_f_exit_authorized"] is False
    assert decisions["batch_g_deployment_compatibility_authorized"] is False
    assert decisions["production_or_default_shadow_authorized"] is False
    assert decisions["telemetry_export_authorized"] is False
    assert decisions["provider_sdk_or_network_authorized"] is False
    assert decisions["public_api_change_authorized"] is False
    assert decisions["default_path_change_authorized"] is False
    assert decisions["legacy_retirement_authorized"] is False
    assert decisions["phase3_exit_authorized"] is False

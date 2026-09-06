from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
CANDIDATE = (
    ROOT / "contracts/ir-phase2-batch-c-performance-budget-review-candidate.json"
)
SUCCESSOR = (
    ROOT / "contracts/ir-phase2-batch-c-performance-budget-artifact-successor.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_batch_c_performance_review_binds_authorization() -> None:
    candidate = json.loads(CANDIDATE.read_text(encoding="utf-8"))
    authorization = candidate["authorization"]

    assert candidate["status"] == "ready_for_owner_approval"
    assert _sha256(ROOT / authorization["path"]) == authorization["sha256"]


def test_batch_c_gate_successor_is_authorized_and_scope_closed() -> None:
    successor = json.loads(SUCCESSOR.read_text(encoding="utf-8"))
    authorization = successor["authorization"]

    assert successor["status"] == "authorized_artifact_successor"
    assert _sha256(ROOT / authorization["path"]) == authorization["sha256"]
    assert successor["predecessor_review_semantics_changed"] is False
    assert successor["public_or_default_path_changed"] is False


def test_batch_c_budget_review_keeps_exit_and_later_scope_closed() -> None:
    candidate = json.loads(CANDIDATE.read_text(encoding="utf-8"))
    decisions = candidate["proposed_decisions"]

    assert decisions["batch_c_performance_budget_approved"] is True
    assert decisions["batch_c_machine_gate_authorized"] is True
    for name, value in decisions.items():
        if name not in {
            "batch_c_performance_budget_approved",
            "batch_c_machine_gate_authorized",
        }:
            assert value is False
    assert candidate["approval_command"] == (
        "approve IR-PHASE2-BATCH-C-PERFORMANCE-BUDGET"
    )

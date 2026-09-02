from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
AUTHORIZATION = ROOT / "contracts/ir-phase2-batch-b-exit-batch-c-authorization.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_batch_c_authorization_binds_exit_candidate() -> None:
    authorization = json.loads(AUTHORIZATION.read_text(encoding="utf-8"))
    candidate = authorization["review_candidate"]

    assert authorization["status"] == "approved"
    assert authorization["approval_command"] == (
        "approve IR-PHASE2-BATCH-B-EXIT-BATCH-C"
    )
    assert _sha256(ROOT / candidate["path"]) == candidate["sha256"]


def test_batch_c_authorization_keeps_later_gates_closed() -> None:
    decisions = json.loads(AUTHORIZATION.read_text(encoding="utf-8"))["decisions"]

    assert decisions["batch_b_exit_approved"] is True
    assert decisions["batch_c_private_placement_routing_authorized"] is True
    assert decisions["batch_c_performance_baseline_authorized"] is True
    assert decisions["batch_c_performance_budget_approved"] is False
    assert decisions["emitter_or_provider_codegen_authorized"] is False
    assert decisions["remote_submission_authorized"] is False
    assert decisions["public_api_change_authorized"] is False
    assert decisions["default_path_change_authorized"] is False
    assert decisions["legacy_retirement_authorized"] is False
    assert decisions["phase2_exit_authorized"] is False

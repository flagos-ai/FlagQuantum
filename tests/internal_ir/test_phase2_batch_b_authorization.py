from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
AUTHORIZATION = ROOT / "contracts/ir-phase2-batch-a-exit-batch-b-authorization.json"


def test_batch_b_authorization_binds_exact_exit_review() -> None:
    authorization = json.loads(AUTHORIZATION.read_text(encoding="utf-8"))
    review = authorization["review_candidate"]

    assert authorization["status"] == "approved"
    assert authorization["approval_command"] == (
        "approve IR-PHASE2-BATCH-A-EXIT-BATCH-B"
    )
    assert hashlib.sha256((ROOT / review["path"]).read_bytes()).hexdigest() == (
        review["sha256"]
    )


def test_batch_b_authorization_is_private_and_stops_before_batch_c() -> None:
    authorization = json.loads(AUTHORIZATION.read_text(encoding="utf-8"))
    decisions = authorization["decisions"]

    assert decisions["batch_a_exit_authorized"] is True
    assert decisions["batch_b_entry_authorized"] is True
    assert decisions["private_target_gate_set_decomposition_authorized"] is True
    assert decisions["batch_c_through_f_authorized"] is False
    assert decisions["routing_authorized"] is False
    assert decisions["emitter_authorized"] is False
    assert decisions["provider_codegen_authorized"] is False
    assert decisions["public_api_change_authorized"] is False
    assert decisions["default_path_change_authorized"] is False
    assert decisions["legacy_retirement_authorized"] is False

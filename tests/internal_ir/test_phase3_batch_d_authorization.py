from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
AUTHORIZATION = ROOT / "contracts/ir-phase3-batch-c-exit-batch-d-authorization.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_batch_d_authorization_binds_approved_exit_candidate() -> None:
    authorization = json.loads(AUTHORIZATION.read_text(encoding="utf-8"))

    assert authorization["status"] == "approved"
    assert authorization["approval_command"] == (
        "approve IR-PHASE3-BATCH-C-EXIT-BATCH-D"
    )
    candidate = authorization["review_candidate"]
    assert _sha256(ROOT / candidate["path"]) == candidate["sha256"]


def test_batch_d_authorization_is_private_offline_and_narrow() -> None:
    authorization = json.loads(AUTHORIZATION.read_text(encoding="utf-8"))
    decisions = authorization["decisions"]
    scope = authorization["allowed_scope"]

    assert decisions["batch_c_exit_authorized"] is True
    assert decisions["batch_d_entry_authorized"] is True
    assert decisions["batch_d_performance_budget_approved"] is False
    assert decisions["batch_e_through_h_authorized"] is False
    assert decisions["provider_sdk_or_remote_submission_authorized"] is False
    assert decisions["credentials_authorized"] is False
    assert decisions["real_backend_identity_in_anonymous_evidence_authorized"] is False
    assert decisions["new_emitter_authorized"] is False
    assert decisions["public_api_change_authorized"] is False
    assert decisions["default_path_change_authorized"] is False
    assert decisions["legacy_retirement_authorized"] is False
    assert decisions["phase3_exit_authorized"] is False
    assert scope["local_sync_adapter_allowed"] is True
    assert scope["offline_async_mock_allowed"] is True
    assert scope["real_provider_calls_allowed"] is False
    assert scope["credentials_allowed"] is False
    assert scope["public_exports"] == []

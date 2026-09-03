from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
AUTHORIZATION = ROOT / "contracts/ir-phase3-batch-f-exit-batch-g-authorization.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_batch_g_authorization_binds_owner_review() -> None:
    authorization = json.loads(AUTHORIZATION.read_text(encoding="utf-8"))

    assert authorization["status"] == "approved"
    assert authorization["approval_command"] == (
        "approve IR-PHASE3-BATCH-F-EXIT-BATCH-G"
    )
    candidate = authorization["review_candidate"]
    assert _sha256(ROOT / candidate["path"]) == candidate["sha256"]


def test_batch_g_authorization_is_proposal_only() -> None:
    authorization = json.loads(AUTHORIZATION.read_text(encoding="utf-8"))
    decisions = authorization["decisions"]
    scope = authorization["allowed_scope"]

    assert decisions["batch_f_exit_authorized"] is True
    assert decisions["batch_g_entry_authorized"] is True
    assert decisions["batch_h_authorized"] is False
    assert decisions["runtime_implementation_authorized"] is False
    assert decisions["canary_activation_authorized"] is False
    assert decisions["production_or_default_shadow_authorized"] is False
    assert decisions["provider_sdk_or_network_authorized"] is False
    assert decisions["public_api_change_authorized"] is False
    assert decisions["default_path_change_authorized"] is False
    assert decisions["legacy_retirement_authorized"] is False
    assert decisions["phase3_exit_authorized"] is False
    assert scope["offline_contract_tests_only"] is True
    assert set(scope["roots"]) == {
        "docs/development",
        "contracts",
        "tests/internal_ir",
    }

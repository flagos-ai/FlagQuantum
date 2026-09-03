from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
AUTHORIZATION = ROOT / "contracts/ir-phase3-batch-e-exit-batch-f-authorization.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_batch_f_authorization_binds_approved_exit_candidate() -> None:
    authorization = json.loads(AUTHORIZATION.read_text(encoding="utf-8"))

    assert authorization["status"] == "approved"
    assert authorization["approval_command"] == (
        "approve IR-PHASE3-BATCH-E-EXIT-BATCH-F"
    )
    candidate = authorization["review_candidate"]
    assert _sha256(ROOT / candidate["path"]) == candidate["sha256"]


def test_batch_f_authorization_is_explicit_private_and_narrow() -> None:
    authorization = json.loads(AUTHORIZATION.read_text(encoding="utf-8"))
    decisions = authorization["decisions"]
    scope = authorization["allowed_scope"]

    assert decisions["batch_e_exit_authorized"] is True
    assert decisions["batch_f_entry_authorized"] is True
    assert decisions["batch_f_performance_budget_approved"] is False
    assert decisions["batch_g_through_h_authorized"] is False
    assert decisions["production_shadow_authorized"] is False
    assert decisions["telemetry_export_authorized"] is False
    assert decisions["provider_sdk_or_network_authorized"] is False
    assert decisions["public_api_change_authorized"] is False
    assert decisions["default_path_change_authorized"] is False
    assert decisions["legacy_retirement_authorized"] is False
    assert decisions["phase3_exit_authorized"] is False
    assert scope["offline_explicit_invocation_only"] is True
    assert scope["legacy_authoritative"] is True
    assert scope["kill_switch_required"] is True
    assert scope["bounded_overhead_required"] is True
    assert scope["privacy_safe_local_evidence_only"] is True
    assert scope["environment_or_import_hook_activation_allowed"] is False
    assert scope["background_or_production_activation_allowed"] is False
    assert scope["telemetry_export_allowed"] is False
    assert scope["public_exports"] == []

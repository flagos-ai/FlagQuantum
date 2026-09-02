from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
CANDIDATE = ROOT / "contracts/ir-phase1-batch-d-review-candidate.json"


def _candidate() -> dict[str, object]:
    return json.loads(CANDIDATE.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_batch_d_candidate_binds_authorization_and_artifacts() -> None:
    candidate = _candidate()
    authorization = candidate["authorization"]

    assert _sha256(ROOT / authorization["path"]) == authorization["sha256"]
    for section in ("implementation_artifacts", "test_artifacts"):
        for relative_path, expected_hash in candidate[section].items():
            assert _sha256(ROOT / relative_path) == expected_hash


def test_batch_d_candidate_records_exact_restricted_scope() -> None:
    candidate = _candidate()
    scope = candidate["scope"]

    assert candidate["status"] == "ready_for_batch_d_review"
    assert scope["supported_fixture_round_trip_exact"] is True
    assert scope["lossy_mode"] is False
    assert scope["best_effort_mode"] is False
    assert scope["provider_codegen"] is False
    assert scope["public_exports_added"] == []
    assert scope["default_path_enabled"] is False


def test_batch_d_candidate_does_not_overclaim_export_capability() -> None:
    candidate = _candidate()
    capability = candidate["capability_boundary"]
    decision = candidate["exit_decision"]

    assert capability["can_restore_sealed_supported_source_exactly"] is True
    assert capability["can_export_arbitrarily_transformed_quantum_ir"] is False
    assert capability["can_generate_provider_payload"] is False
    assert decision["batch_d_owner_accepted"] is False
    assert decision["batch_e_authorized"] is False
    assert decision["phase1_complete"] is False
    assert decision["phase2_authorized"] is False
    assert candidate["next_review_command"] == (
        "approve IR-PHASE1-BATCH-D-EXIT-BATCH-E"
    )

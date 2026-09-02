from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
CANDIDATE = ROOT / "contracts/quafu-provider-p0-review-candidate.json"


def _candidate() -> dict[str, object]:
    return json.loads(CANDIDATE.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_quafu_p0_candidate_binds_reviewed_baseline() -> None:
    candidate = _candidate()

    assert candidate["status"] == "ready_for_owner_approval"
    for relative_path, expected_hash in candidate["reviewed_artifacts"].items():
        assert _sha256(ROOT / relative_path) == expected_hash


def test_quafu_p0_candidate_separates_mock_and_real_evidence() -> None:
    candidate = _candidate()
    evidence = candidate["current_evidence"]

    assert evidence["mock_submit_status_cancel_result"] is True
    assert evidence["receipt_to_result_identity"] is True
    assert evidence["real_backend_execution"] is False
    assert evidence["five_state_normalization"] is False
    assert evidence["idempotent_submission"] is False


def test_quafu_p0_candidate_cannot_authorize_public_or_external_actions() -> None:
    candidate = _candidate()
    authorization = candidate["authorization"]
    scope = candidate["p0_scope"]

    assert all(value is False for value in authorization.values())
    assert scope["public_api_changes"] == []
    assert scope["real_paid_submission_authorized"] is False
    assert scope["adapter_repository_write_authorized"] is False


def test_quafu_p0_candidate_requires_exact_approval_semantics() -> None:
    candidate = _candidate()
    text = candidate["approval_text"]

    assert candidate["approval_command"] == "approve QUAFU-PROVIDER-P0"
    assert "core, mocked, fail-closed implementation" in text
    assert "does not add or change stable public fields" in text
    assert "real paid task submission" in text
    assert "adapter repository writes" in text

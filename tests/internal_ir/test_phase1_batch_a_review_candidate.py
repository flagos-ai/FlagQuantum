from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
CANDIDATE = ROOT / "contracts/ir-phase1-batch-a-review-candidate.json"


def _candidate() -> dict[str, object]:
    return json.loads(CANDIDATE.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_batch_a_candidate_binds_authorization() -> None:
    candidate = _candidate()
    authorization = candidate["authorization"]

    assert _sha256(ROOT / authorization["path"]) == authorization["sha256"]


def test_batch_a_candidate_records_completed_scope() -> None:
    candidate = _candidate()

    assert candidate["status"] == "ready_for_batch_a_review"
    assert candidate["scope"]["canonical_opcode_schemas"] == 35
    assert candidate["scope"]["public_exports_added"] == []
    assert candidate["scope"]["default_path_enabled"] is False
    assert candidate["exit_decision"]["batch_a_technical_complete"] is True


def test_batch_a_candidate_cannot_authorize_next_batch() -> None:
    candidate = _candidate()
    decision = candidate["exit_decision"]

    assert decision["batch_a_owner_accepted"] is False
    assert decision["batch_b_authorized"] is False
    assert decision["phase1_complete"] is False
    assert decision["phase2_authorized"] is False
    assert candidate["next_review_command"] == (
        "approve IR-PHASE1-BATCH-A-EXIT-BATCH-B"
    )
    assert (
        "only Batch B verifier and structured diagnostics"
        in candidate["next_review_semantics"]
    )

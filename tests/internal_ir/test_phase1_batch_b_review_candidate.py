from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
CANDIDATE = ROOT / "contracts/ir-phase1-batch-b-review-candidate.json"


def _candidate() -> dict[str, object]:
    return json.loads(CANDIDATE.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_batch_b_candidate_binds_authorization_and_artifacts() -> None:
    candidate = _candidate()
    authorization = candidate["authorization"]

    assert _sha256(ROOT / authorization["path"]) == authorization["sha256"]
    for section in ("implementation_artifacts", "test_artifacts"):
        for relative_path, expected_hash in candidate[section].items():
            assert _sha256(ROOT / relative_path) == expected_hash


def test_batch_b_candidate_records_fail_closed_scope() -> None:
    candidate = _candidate()

    assert candidate["status"] == "ready_for_batch_b_review"
    assert candidate["scope"]["diagnostic_codes"] == 14
    assert candidate["scope"]["prints_diagnostics"] is False
    assert candidate["scope"]["public_exports_added"] == []
    assert candidate["scope"]["default_path_enabled"] is False
    assert candidate["exit_decision"]["batch_b_technical_complete"] is True


def test_batch_b_candidate_keeps_importer_and_later_phases_closed() -> None:
    candidate = _candidate()
    decision = candidate["exit_decision"]

    assert decision["batch_b_owner_accepted"] is False
    assert decision["batch_c_authorized"] is False
    assert decision["phase1_complete"] is False
    assert decision["phase2_authorized"] is False
    assert candidate["next_review_command"] == (
        "approve IR-PHASE1-BATCH-B-EXIT-BATCH-C"
    )
    assert (
        "No exporter, public API, default runtime" in candidate["next_review_semantics"]
    )

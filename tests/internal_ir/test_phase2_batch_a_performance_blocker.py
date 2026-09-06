from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
BLOCKER = ROOT / "contracts/ir-phase2-batch-a-performance-blocker.json"
CANDIDATE = (
    ROOT / "contracts/ir-phase2-batch-a-performance-remediation-review-candidate.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_performance_blocker_preserves_original_budget_and_stops_exit() -> None:
    blocker = json.loads(BLOCKER.read_text(encoding="utf-8"))
    budget = ROOT / blocker["budget"]["path"]

    assert blocker["status"] == "blocked"
    assert blocker["blocker_id"] == "IR2A-PERF-001"
    assert blocker["budget"]["modified"] is False
    assert _sha256(budget) == blocker["budget"]["sha256"]
    assert blocker["evidence"]["latency_passed"] is False
    assert blocker["evidence"]["memory_passed"] is False
    assert blocker["decisions"]["batch_a_exit_allowed"] is False
    assert blocker["decisions"]["batch_b_start_allowed"] is False
    assert blocker["decisions"]["budget_auto_update_allowed"] is False


def test_remediation_candidate_binds_blocker() -> None:
    candidate = json.loads(CANDIDATE.read_text(encoding="utf-8"))

    assert candidate["status"] == "ready_for_owner_approval"
    assert _sha256(ROOT / candidate["blocker"]["path"]) == (
        candidate["blocker"]["sha256"]
    )


def test_remediation_candidate_cannot_approve_budget_or_later_work() -> None:
    candidate = json.loads(CANDIDATE.read_text(encoding="utf-8"))
    authorization = candidate["authorization"]

    assert all(value is False for value in authorization.values())
    assert candidate["approval_command"] == (
        "approve IR-PHASE2-BATCH-A-PERFORMANCE-REMEDIATION"
    )
    assert "does not modify or supersede the original budget" in (
        candidate["approval_text"]
    )
    assert "approve any successor budget" in candidate["approval_text"]
    assert "authorize Batch A exit or Batch B" in candidate["approval_text"]

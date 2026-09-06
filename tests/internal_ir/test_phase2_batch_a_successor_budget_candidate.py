from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
REVIEW = ROOT / "contracts/ir-phase2-batch-a-successor-budget-review-candidate.json"
EVIDENCE = ROOT / "contracts/ir-phase2-batch-a-performance-remediation-evidence.json"
ORIGINAL = ROOT / "tests/fixtures/internal_ir/phase2_performance_budget_candidate.json"
SUCCESSOR = (
    ROOT
    / "tests/fixtures/internal_ir/phase2_performance_budget_successor_candidate.json"
)


def test_successor_budget_review_records_artifact_snapshot() -> None:
    review = json.loads(REVIEW.read_text(encoding="utf-8"))

    assert review["status"] == "awaiting_owner_approval"
    assert review["approval_command"] == ("approve IR-PHASE2-BATCH-A-SUCCESSOR-BUDGET")
    assert review["artifacts"]
    assert all(
        len(digest) == 64 and set(digest) <= set("0123456789abcdef")
        for digest in review["artifacts"].values()
    )


def test_successor_candidate_is_immutable_and_preserves_original_snapshot() -> None:
    successor = json.loads(SUCCESSOR.read_text(encoding="utf-8"))
    review = json.loads(REVIEW.read_text(encoding="utf-8"))

    assert successor["status"] == "pending_owner_approval"
    assert successor["approval"]["approved"] is False
    assert successor["supersedes_if_approved"] == str(ORIGINAL.relative_to(ROOT))
    assert hashlib.sha256(ORIGINAL.read_bytes()).hexdigest() == (
        review["artifacts"][str(ORIGINAL.relative_to(ROOT))]
    )
    assert review["proposed_decisions"]["batch_a_exit_authorized"] is False
    assert review["proposed_decisions"]["batch_b_start_authorized"] is False


def test_successor_budget_has_measured_headroom_without_public_sla_claim() -> None:
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    successor = json.loads(SUCCESSOR.read_text(encoding="utf-8"))
    measured = {item["gate_count"]: item for item in evidence["full_pipeline"]}
    component = {item["gate_count"]: item for item in evidence["component_p95_ms"]}

    assert "not a public SLA" in successor["derivation"]["claim_boundary"]
    for budget in successor["budgets"]:
        gate_count = budget["gate_count"]
        observed_latency = max(
            measured[gate_count]["p95_ms"], component[gate_count]["total"]
        )
        assert budget["pipeline_p95_ms_max"] >= observed_latency * 1.25
        assert budget["peak_host_memory_bytes_max"] >= (
            measured[gate_count]["peak_host_memory_bytes"] * 1.25
        )

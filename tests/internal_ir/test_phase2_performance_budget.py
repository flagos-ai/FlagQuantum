from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from benchmarks.internal.ir_phase2_batch_a_gate import measure_case

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
BUDGET = (
    ROOT
    / "tests/fixtures/internal_ir/phase2_performance_budget_successor_candidate.json"
)
ORIGINAL_BUDGET = (
    ROOT / "tests/fixtures/internal_ir/phase2_performance_budget_candidate.json"
)
AUTHORIZATION = ROOT / "contracts/ir-phase2-batch-a-successor-budget-authorization.json"
CANDIDATE = ROOT / "contracts/ir-phase2-batch-a-successor-budget-review-candidate.json"


def test_phase2_successor_budget_is_approved_only_through_bound_authorization() -> None:
    budget = json.loads(BUDGET.read_text(encoding="utf-8"))
    authorization = json.loads(AUTHORIZATION.read_text(encoding="utf-8"))

    assert budget["status"] == "pending_owner_approval"
    assert authorization["decisions"]["successor_budget_approved"] is True
    assert authorization["approved_budget"]["path"] == str(BUDGET.relative_to(ROOT))
    assert hashlib.sha256(BUDGET.read_bytes()).hexdigest() == (
        authorization["approved_budget"]["sha256"]
    )
    assert (
        hashlib.sha256(CANDIDATE.read_bytes()).hexdigest()
        == authorization["review_candidate"]["sha256"]
    )


def test_approved_phase2_batch_a_successor_budget_is_machine_enforced() -> None:
    budget = json.loads(BUDGET.read_text(encoding="utf-8"))
    case_budget = next(item for item in budget["budgets"] if item["gate_count"] == 1000)

    case = measure_case(case_budget, iterations=3, warmup=1)

    assert case["latency_passed"] is True
    assert case["memory_passed"] is True
    assert case["deterministic_identity"] is True


def test_phase2_performance_gate_never_rewrites_budget() -> None:
    original_before = ORIGINAL_BUDGET.read_bytes()
    before = BUDGET.read_bytes()
    budget = json.loads(before)
    case_budget = next(item for item in budget["budgets"] if item["gate_count"] == 10)

    measure_case(case_budget, iterations=3, warmup=1)

    assert BUDGET.read_bytes() == before
    assert ORIGINAL_BUDGET.read_bytes() == original_before

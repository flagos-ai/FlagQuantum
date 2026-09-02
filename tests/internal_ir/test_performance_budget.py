from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.internal.ir_phase1_gate import evaluate

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
BUDGET = ROOT / "tests/fixtures/internal_ir/phase1_performance_budget_candidate.json"


def test_approved_import_verify_budget_is_machine_enforced() -> None:
    result = evaluate(BUDGET, iterations=5, warmup=2)

    assert result["status"] == "passed"
    assert result["growth_passed"] is True
    assert [case["gate_count"] for case in result["cases"]] == [10, 100, 1000, 10000]
    assert all(case["latency_passed"] for case in result["cases"])
    assert all(case["memory_passed"] for case in result["cases"])
    assert all(case["deterministic_identity"] for case in result["cases"])


def test_performance_gate_uses_approved_budget_without_rewriting_it() -> None:
    before = BUDGET.read_bytes()
    budget = json.loads(before)

    evaluate(BUDGET, iterations=3, warmup=1)

    assert BUDGET.read_bytes() == before
    assert budget["status"] == "approved_internal_phase1_gate"
    assert budget["approval"]["approved"] is True

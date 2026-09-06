from __future__ import annotations

import json
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.benchmark_contract]

ROOT = Path(__file__).resolve().parents[2]
BASELINE = ROOT / "tests/fixtures/internal_ir/phase2_batch_c_performance_baseline.json"
BUDGET = ROOT / "tests/fixtures/internal_ir/phase2_batch_c_performance_budget.json"


def test_batch_c_baseline_and_budget_are_internal() -> None:
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    budget = json.loads(BUDGET.read_text(encoding="utf-8"))

    assert baseline["status"] == "measured_baseline"
    assert budget["status"] == "active_private_regression_budget"
    assert "not a public SLA" in baseline["claim_scope"]
    assert "not a public SLA" in budget["derivation"]["claim_boundary"]
    assert baseline["method"]["pipeline_scope"] == (
        "isolated Batch C placement/routing pass"
    )


def test_batch_c_budget_has_at_least_twenty_five_percent_headroom() -> None:
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    budget = json.loads(BUDGET.read_text(encoding="utf-8"))
    measured = {item["gate_count"]: item for item in baseline["cases"]}

    for candidate in budget["budgets"]:
        observation = measured[candidate["gate_count"]]
        assert candidate["pipeline_p95_ms_max"] >= observation["p95_ms"] * 1.25
        assert candidate["peak_host_memory_bytes_max"] >= (
            observation["peak_host_memory_bytes"] * 1.25
        )
        assert observation["deterministic_identity"] is True

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.benchmark_contract]

ROOT = Path(__file__).resolve().parents[2]
BASELINE = ROOT / "tests/fixtures/internal_ir/phase2_batch_e_performance_baseline.json"
BUDGET = (
    ROOT
    / "tests"
    / "fixtures"
    / "internal_ir"
    / "phase2_batch_e_performance_budget.json"
)


def test_batch_e_budget_has_headroom_over_observed_envelope() -> None:
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    budget_record = json.loads(BUDGET.read_text(encoding="utf-8"))
    budgets = {item["gate_count"]: item for item in budget_record["budgets"]}

    assert baseline["status"] == "measured_baseline"
    assert budget_record["status"] == "active_private_regression_budget"
    for observed in baseline["observed_envelope"]:
        budget = budgets[observed["gate_count"]]
        for metric in (
            "cold_miss_p95_ms_max",
            "cache_hit_p95_ms_max",
            "identity_bypass_p95_ms_max",
            "cold_peak_host_memory_bytes_max",
        ):
            assert budget[metric] >= observed[metric] * 1.25
        assert budget["cache_hit_speedup_min"] <= (
            observed["observed_hit_speedup_min"] / 1.25
        )

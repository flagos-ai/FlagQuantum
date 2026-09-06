from __future__ import annotations

import json
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.benchmark_contract]

ROOT = Path(__file__).resolve().parents[2]
BASELINE = ROOT / "tests/fixtures/internal_ir/phase2_batch_f_performance_baseline.json"
BUDGET = ROOT / "tests/fixtures/internal_ir/phase2_batch_f_performance_budget.json"


def test_batch_f_baseline_is_observation_not_sla() -> None:
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))

    assert baseline["status"] == "measured_baseline"
    assert "not a public SLA" in baseline["claim_scope"]
    assert [case["gate_count"] for case in baseline["cases"]] == [10, 100, 1000, 10000]
    assert all(case["deterministic_identity"] for case in baseline["cases"])
    assert all(case["deterministic_text_size"] for case in baseline["cases"])


def test_batch_f_budget_is_above_observed_baseline() -> None:
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    budget = json.loads(BUDGET.read_text(encoding="utf-8"))
    measured = {case["gate_count"]: case for case in baseline["cases"]}

    assert budget["status"] == "active_private_regression_budget"
    for limit in budget["budgets"]:
        observed = measured[limit["gate_count"]]
        assert limit["cold_end_to_end_p95_ms_max"] > observed["cold_end_to_end_p95_ms"]
        assert (
            limit["cached_end_to_end_p95_ms_max"] > observed["cached_end_to_end_p95_ms"]
        )
        assert (
            limit["cold_peak_host_memory_bytes_max"]
            > observed["cold_peak_host_memory_bytes"]
        )

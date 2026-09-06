from __future__ import annotations

import json
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.benchmark_contract]

ROOT = Path(__file__).resolve().parents[2]
BASELINE = ROOT / "tests/fixtures/internal_ir/phase2_batch_d_performance_baseline.json"
BUDGET = (
    ROOT
    / "tests"
    / "fixtures"
    / "internal_ir"
    / "phase2_batch_d_performance_budget.json"
)


def test_batch_d_budget_has_headroom_over_observed_envelope() -> None:
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    budget_record = json.loads(BUDGET.read_text(encoding="utf-8"))
    budgets = {item["gate_count"]: item for item in budget_record["budgets"]}

    assert baseline["status"] == "measured_baseline"
    assert budget_record["status"] == "active_private_regression_budget"
    for observed in baseline["observed_maxima"]:
        budget = budgets[observed["gate_count"]]
        for emitter in ("openqasm2", "openqasm3", "qcis_v1"):
            assert budget[f"{emitter}_p95_ms_max"] >= (
                observed[f"{emitter}_p95_ms"] * 1.25
            )
            assert budget[f"{emitter}_peak_host_memory_bytes_max"] >= (
                observed[f"{emitter}_peak_host_memory_bytes"] * 1.25
            )

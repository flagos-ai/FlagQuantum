from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from benchmarks.internal import ir_phase2_batch_b_gate as gate

pytestmark = [pytest.mark.unit, pytest.mark.benchmark_contract]

ROOT = Path(__file__).resolve().parents[2]
BUDGET = ROOT / "tests/fixtures/internal_ir/phase2_batch_b_performance_budget.json"


def test_batch_b_machine_gate_accepts_budget_boundary_and_rejects_regression(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    budgets = json.loads(BUDGET.read_text(encoding="utf-8"))["budgets"]
    budget_by_size = {item["gate_count"]: item for item in budgets}
    regression = {"enabled": False}

    def observation(gate_count: int, *, iterations: int, warmup: int) -> dict[str, Any]:
        del iterations, warmup
        budget = budget_by_size[gate_count]
        p95 = budget["pipeline_p95_ms_max"]
        if regression["enabled"] and gate_count == 1000:
            p95 += 0.001
        return {
            "gate_count": gate_count,
            "emitted_gate_count": gate_count,
            "expansion_ratio": 1.0,
            "timings_ms": {"p50": p95, "p95": p95, "max": p95},
            "peak_host_memory_bytes": budget["peak_host_memory_bytes_max"],
            "deterministic_identity": True,
        }

    monkeypatch.setattr(gate, "measure_case", observation)
    accepted = gate.evaluate(BUDGET, iterations=3, warmup=1)
    regression["enabled"] = True
    rejected = gate.evaluate(BUDGET, iterations=3, warmup=1)

    assert accepted["status"] == "passed"
    assert all(case["passed"] for case in accepted["cases"])
    assert rejected["status"] == "failed"
    failed = next(case for case in rejected["cases"] if not case["passed"])
    assert failed["gate_count"] == 1000
    assert failed["latency_passed"] is False

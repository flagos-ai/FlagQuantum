from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from benchmarks.internal import ir_phase3_batch_d_gate as gate

pytestmark = [pytest.mark.unit, pytest.mark.benchmark_contract]

ROOT = Path(__file__).resolve().parents[2]
BUDGET = ROOT / "tests/fixtures/internal_ir/phase3_batch_d_performance_budget.json"


def test_batch_d_machine_gate_accepts_boundary_and_rejects_regression(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    budgets = json.loads(BUDGET.read_text(encoding="utf-8"))["budgets"]
    budget_by_size = {item["submission_count"]: item for item in budgets}
    regression = {"enabled": False}

    def observation(
        submission_count: int, *, iterations: int, warmup: int
    ) -> dict[str, Any]:
        del iterations, warmup
        budget = budget_by_size[submission_count]
        latency = budget["local_lifecycle_p95_ms_max"]
        if regression["enabled"] and submission_count == 1000:
            latency += 0.001
        return {
            "submission_count": submission_count,
            "local_lifecycle_p95_ms": latency,
            "local_lifecycle_peak_host_memory_bytes": budget[
                "local_lifecycle_peak_host_memory_bytes_max"
            ],
            "deterministic_identity": True,
        }

    monkeypatch.setattr(gate, "measure_case", observation)
    accepted = gate.evaluate(BUDGET, iterations=5, warmup=1)
    regression["enabled"] = True
    rejected = gate.evaluate(BUDGET, iterations=5, warmup=1)

    assert accepted["status"] == "passed"
    assert all(case["passed"] for case in accepted["cases"])
    assert rejected["status"] == "failed"
    failed = next(case for case in rejected["cases"] if not case["passed"])
    assert failed["submission_count"] == 1000
    assert failed["latency_passed"] is False

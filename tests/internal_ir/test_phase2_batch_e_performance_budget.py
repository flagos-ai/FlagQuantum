from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from benchmarks.internal import ir_phase2_batch_e_gate as gate

pytestmark = [pytest.mark.unit, pytest.mark.benchmark_contract]

ROOT = Path(__file__).resolve().parents[2]
BUDGET = (
    ROOT
    / "tests"
    / "fixtures"
    / "internal_ir"
    / "phase2_batch_e_performance_budget_candidate.json"
)


def test_batch_e_machine_gate_accepts_boundaries_and_rejects_regression(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    budgets = json.loads(BUDGET.read_text(encoding="utf-8"))["budgets"]
    budget_by_size = {item["gate_count"]: item for item in budgets}
    regression = {"enabled": False}

    def observation(gate_count: int, *, iterations: int, warmup: int) -> dict[str, Any]:
        del iterations, warmup
        budget = budget_by_size[gate_count]
        memory = budget["cold_peak_host_memory_bytes_max"]
        if regression["enabled"] and gate_count == 10000:
            memory += 1
        return {
            "gate_count": gate_count,
            "cold_miss_p95_ms": budget["cold_miss_p95_ms_max"],
            "cache_hit_p95_ms": budget["cache_hit_p95_ms_max"],
            "identity_bypass_p95_ms": budget["identity_bypass_p95_ms_max"],
            "observed_hit_speedup": budget["cache_hit_speedup_min"],
            "cold_peak_host_memory_bytes": memory,
            "deterministic_identity": True,
            "cache_snapshot": {},
        }

    monkeypatch.setattr(gate, "measure_case", observation)
    accepted = gate.evaluate(BUDGET, iterations=3, warmup=1)
    regression["enabled"] = True
    rejected = gate.evaluate(BUDGET, iterations=3, warmup=1)

    assert accepted["status"] == "passed"
    assert all(case["passed"] for case in accepted["cases"])
    assert rejected["status"] == "failed"
    failed = next(case for case in rejected["cases"] if not case["passed"])
    assert failed["gate_count"] == 10000
    assert failed["memory_passed"] is False

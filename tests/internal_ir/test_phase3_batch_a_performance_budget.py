from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from benchmarks.internal import ir_phase3_batch_a_gate as gate

pytestmark = [pytest.mark.unit, pytest.mark.benchmark_contract]

ROOT = Path(__file__).resolve().parents[2]
BUDGET = (
    ROOT / "tests/fixtures/internal_ir/phase3_batch_a_performance_budget_candidate.json"
)


def test_batch_a_machine_gate_accepts_boundary_and_rejects_regression(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    budgets = json.loads(BUDGET.read_text(encoding="utf-8"))["budgets"]
    budget_by_size = {item["capability_entry_count"]: item for item in budgets}
    regression = {"enabled": False}

    def observation(
        entry_count: int, *, iterations: int, warmup: int
    ) -> dict[str, Any]:
        del iterations, warmup
        budget = budget_by_size[entry_count]
        comparison = budget["compatible_comparison_p95_ms_max"]
        if regression["enabled"] and entry_count == 1000:
            comparison += 0.001
        return {
            "capability_entry_count": entry_count,
            "physical_qubit_capacity": 2,
            "construct_and_fingerprint_p95_ms": budget[
                "construct_and_fingerprint_p95_ms_max"
            ],
            "compatible_comparison_p95_ms": comparison,
            "construct_and_fingerprint_peak_host_memory_bytes": budget[
                "construct_and_fingerprint_peak_host_memory_bytes_max"
            ],
            "canonical_bytes": 1,
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
    assert failed["capability_entry_count"] == 1000
    assert failed["comparison_latency_passed"] is False

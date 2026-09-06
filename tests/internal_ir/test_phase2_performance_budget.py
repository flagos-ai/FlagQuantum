from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from benchmarks.internal import ir_phase2_batch_a_gate as gate

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
BUDGET = ROOT / "tests/fixtures/internal_ir/phase2_batch_a_performance_budget.json"


def _boundary_observation(
    budget: dict[str, Any], *, iterations: int, warmup: int
) -> dict[str, Any]:
    del iterations, warmup
    p95 = float(budget["pipeline_p95_ms_max"])
    return {
        "gate_count": int(budget["gate_count"]),
        "timings_ms": {"min": p95, "p50": p95, "p95": p95, "max": p95},
        "peak_host_memory_bytes": int(budget["peak_host_memory_bytes_max"]),
        "budget": {
            "pipeline_p95_ms_max": p95,
            "peak_host_memory_bytes_max": int(budget["peak_host_memory_bytes_max"]),
        },
        "deterministic_identity": True,
        "latency_passed": True,
        "memory_passed": True,
    }


def test_phase2_batch_a_gate_accepts_budget_and_rejects_regression(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(gate, "measure_case", _boundary_observation)
    accepted = gate.evaluate(BUDGET, iterations=3, warmup=1)

    def regression(
        budget: dict[str, Any], *, iterations: int, warmup: int
    ) -> dict[str, Any]:
        case = _boundary_observation(budget, iterations=iterations, warmup=warmup)
        if case["gate_count"] == 1000:
            case["latency_passed"] = False
        return case

    monkeypatch.setattr(gate, "measure_case", regression)
    rejected = gate.evaluate(BUDGET, iterations=3, warmup=1)

    assert accepted["status"] == "passed"
    assert accepted["growth_passed"] is True
    assert rejected["status"] == "failed"


def test_phase2_performance_gate_never_rewrites_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    before = BUDGET.read_bytes()
    monkeypatch.setattr(gate, "measure_case", _boundary_observation)

    gate.evaluate(BUDGET, iterations=3, warmup=1)

    assert BUDGET.read_bytes() == before

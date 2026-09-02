from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from benchmarks.internal import ir_phase2_batch_d_gate as gate

pytestmark = [pytest.mark.unit, pytest.mark.benchmark_contract]

ROOT = Path(__file__).resolve().parents[2]
BUDGET = (
    ROOT
    / "tests"
    / "fixtures"
    / "internal_ir"
    / "phase2_batch_d_performance_budget_candidate.json"
)


def test_batch_d_machine_gate_accepts_boundaries_and_rejects_regression(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    budgets = json.loads(BUDGET.read_text(encoding="utf-8"))["budgets"]
    budget_by_size = {item["gate_count"]: item for item in budgets}
    regression = {"enabled": False}

    def observation(
        module: object,
        emitter: gate.Emitter,
        *,
        iterations: int,
        warmup: int,
    ) -> dict[str, Any]:
        del module, iterations, warmup
        name = next(name for name, candidate in gate.EMITTERS if candidate is emitter)
        gate_count = current_gate_count["value"]
        budget = budget_by_size[gate_count]
        memory = budget[f"{name}_peak_host_memory_bytes_max"]
        if regression["enabled"] and gate_count == 10000 and name == "qcis_v1":
            memory += 1
        latency = budget[f"{name}_p95_ms_max"]
        return {
            "timings_ms": {"p50": latency, "p95": latency, "max": latency},
            "peak_host_memory_bytes": memory,
            "output_bytes": 1,
            "deterministic_content_hash": True,
        }

    current_gate_count = {"value": 0}
    original_build = gate.build_module

    def build(gate_count: int) -> object:
        current_gate_count["value"] = gate_count
        return original_build(1)

    monkeypatch.setattr(gate, "build_module", build)
    monkeypatch.setattr(gate, "measure_emitter", observation)
    accepted = gate.evaluate(BUDGET, iterations=3, warmup=1)
    regression["enabled"] = True
    rejected = gate.evaluate(BUDGET, iterations=3, warmup=1)

    assert accepted["status"] == "passed"
    assert all(case["passed"] for case in accepted["cases"])
    assert rejected["status"] == "failed"
    failed = next(case for case in rejected["cases"] if not case["passed"])
    assert failed["gate_count"] == 10000
    assert failed["emitters"]["qcis_v1"]["memory_passed"] is False

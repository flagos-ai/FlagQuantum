from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from benchmarks.internal import ir_phase2_batch_a_gate as gate

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
BUDGET = (
    ROOT
    / "tests/fixtures/internal_ir/phase2_performance_budget_successor_candidate.json"
)
ORIGINAL_BUDGET = (
    ROOT / "tests/fixtures/internal_ir/phase2_performance_budget_candidate.json"
)
AUTHORIZATION = ROOT / "contracts/ir-phase2-batch-a-successor-budget-authorization.json"
CANDIDATE = ROOT / "contracts/ir-phase2-batch-a-successor-budget-review-candidate.json"


def test_phase2_successor_budget_is_approved_only_through_bound_authorization() -> None:
    budget = json.loads(BUDGET.read_text(encoding="utf-8"))
    authorization = json.loads(AUTHORIZATION.read_text(encoding="utf-8"))

    assert budget["status"] == "pending_owner_approval"
    assert authorization["decisions"]["successor_budget_approved"] is True
    assert authorization["approved_budget"]["path"] == str(BUDGET.relative_to(ROOT))
    assert hashlib.sha256(BUDGET.read_bytes()).hexdigest() == (
        authorization["approved_budget"]["sha256"]
    )
    assert (
        hashlib.sha256(CANDIDATE.read_bytes()).hexdigest()
        == authorization["review_candidate"]["sha256"]
    )


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


def test_approved_phase2_batch_a_successor_gate_accepts_and_rejects(
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
    original_before = ORIGINAL_BUDGET.read_bytes()
    before = BUDGET.read_bytes()
    monkeypatch.setattr(gate, "measure_case", _boundary_observation)

    gate.evaluate(BUDGET, iterations=3, warmup=1)

    assert BUDGET.read_bytes() == before
    assert ORIGINAL_BUDGET.read_bytes() == original_before

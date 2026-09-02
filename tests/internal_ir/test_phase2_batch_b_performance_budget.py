from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from benchmarks.internal.ir_phase2_batch_b_gate import measure_case

pytestmark = [pytest.mark.unit, pytest.mark.benchmark_contract]

ROOT = Path(__file__).resolve().parents[2]
BUDGET = (
    ROOT / "tests/fixtures/internal_ir/phase2_batch_b_performance_budget_candidate.json"
)


def test_batch_b_representative_machine_budget_passes() -> None:
    budgets = json.loads(BUDGET.read_text(encoding="utf-8"))["budgets"]
    budget = next(item for item in budgets if item["gate_count"] == 1000)
    torch.set_num_threads(1)

    observed = measure_case(1000, iterations=3, warmup=1)

    assert observed["timings_ms"]["p95"] <= budget["pipeline_p95_ms_max"]
    assert observed["peak_host_memory_bytes"] <= budget["peak_host_memory_bytes_max"]
    assert observed["deterministic_identity"] is True

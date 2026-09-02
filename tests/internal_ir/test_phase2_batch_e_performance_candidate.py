from __future__ import annotations

import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
BASELINE = ROOT / "contracts/ir-phase2-batch-e-performance-baseline.json"
CANDIDATE = (
    ROOT
    / "tests"
    / "fixtures"
    / "internal_ir"
    / "phase2_batch_e_performance_budget_candidate.json"
)


def test_candidate_is_unapproved_and_has_headroom_over_observed_envelope() -> None:
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    candidate = json.loads(CANDIDATE.read_text(encoding="utf-8"))
    budgets = {item["gate_count"]: item for item in candidate["budgets"]}

    assert baseline["status"] == "measured_not_approved"
    assert candidate["status"] == "proposed_unapproved"
    assert candidate["decisions"]["budget_approved"] is False
    for observed in baseline["observed_envelope"]:
        budget = budgets[observed["gate_count"]]
        for metric in (
            "cold_miss_p95_ms_max",
            "cache_hit_p95_ms_max",
            "identity_bypass_p95_ms_max",
            "cold_peak_host_memory_bytes_max",
        ):
            assert budget[metric] >= observed[metric] * 1.25
        assert budget["cache_hit_speedup_min"] <= (
            observed["observed_hit_speedup_min"] / 1.25
        )

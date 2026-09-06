from __future__ import annotations

import json
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.benchmark_contract]

ROOT = Path(__file__).resolve().parents[2]
BASELINE = ROOT / "tests/fixtures/internal_ir/phase3_batch_f_performance_baseline.json"
BUDGET = ROOT / "tests/fixtures/internal_ir/phase3_batch_f_performance_budget.json"


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_batch_f_performance_profile_is_internal() -> None:
    baseline = _load(BASELINE)
    budget = _load(BUDGET)

    assert baseline["status"] == "measured_baseline"
    assert budget["status"] == "active_private_regression_budget"
    assert "not a public SLA" in baseline["claim_scope"]
    assert "not a public SLA" in budget["claim_scope"]


def test_batch_f_budget_envelopes_cover_each_observed_baseline() -> None:
    baseline = _load(BASELINE)
    budget_record = _load(BUDGET)
    observed = {item["comparison_count"]: item for item in baseline["cases"]}
    budgets = {item["comparison_count"]: item for item in budget_record["budgets"]}

    assert set(observed) == set(budgets) == {10, 100, 1000, 10000}
    for count, case in observed.items():
        budget = budgets[count]
        assert case["deterministic_evidence_identity"] is True
        assert case["match_shadow_p95_ms"] <= budget["match_shadow_p95_ms_max"]
        assert (
            case["match_shadow_peak_host_memory_bytes"]
            <= budget["match_shadow_peak_host_memory_bytes_max"]
        )

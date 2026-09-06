from __future__ import annotations

import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
BASELINE = (
    ROOT
    / "tests/fixtures/internal_ir/deployment_bridge_stage3_performance_baseline.json"
)
BUDGET = (
    ROOT / "tests/fixtures/internal_ir/deployment_bridge_stage3_performance_budget.json"
)


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_stage3_baseline_is_measurement_only_and_deterministic() -> None:
    baseline = _load(BASELINE)

    assert baseline["status"] == "observed_baseline_not_budget"
    assert [case["evaluation_count"] for case in baseline["cases"]] == [
        10,
        100,
        1000,
        10000,
    ]
    assert all(case["deterministic_evidence_identity"] for case in baseline["cases"])


def test_stage3_budget_is_active_and_above_baseline() -> None:
    baseline = _load(BASELINE)
    budget = _load(BUDGET)

    assert budget["status"] == "active_private_regression_budget"
    budgets = {item["evaluation_count"]: item for item in budget["budgets"]}
    assert set(budgets) == {10, 100, 1000, 10000}
    for case in baseline["cases"]:
        budget = budgets[case["evaluation_count"]]
        assert (
            case["readiness_evaluation_p95_ms"]
            < budget["readiness_evaluation_p95_ms_max"]
        )
        assert (
            case["readiness_evaluation_peak_host_memory_bytes"]
            < budget["readiness_evaluation_peak_host_memory_bytes_max"]
        )

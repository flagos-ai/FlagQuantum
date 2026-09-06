from __future__ import annotations

import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
BASELINE = (
    ROOT
    / "tests/fixtures/internal_ir/deployment_bridge_stage6_performance_baseline.json"
)
BUDGET = (
    ROOT / "tests/fixtures/internal_ir/deployment_bridge_stage6_performance_budget.json"
)


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_stage6_baseline_is_measurement_only_and_deterministic() -> None:
    baseline = _load(BASELINE)

    assert baseline["status"] == "observed_baseline_not_budget"
    assert [case["connector_count"] for case in baseline["cases"]] == [
        10,
        100,
        1000,
        10000,
    ]
    assert all(case["deterministic_evidence_identity"] for case in baseline["cases"])


def test_stage6_budget_is_active_and_above_baseline() -> None:
    baseline = _load(BASELINE)
    budget = _load(BUDGET)

    assert budget["status"] == "active_private_regression_budget"
    budgets = {item["connector_count"]: item for item in budget["budgets"]}
    assert set(budgets) == {10, 100, 1000, 10000}
    for case in baseline["cases"]:
        budget = budgets[case["connector_count"]]
        assert (
            case["scripted_connector_p95_ms"] < budget["scripted_connector_p95_ms_max"]
        )
        assert (
            case["scripted_connector_peak_host_memory_bytes"]
            < budget["scripted_connector_peak_host_memory_bytes_max"]
        )

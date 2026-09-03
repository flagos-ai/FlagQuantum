from __future__ import annotations

import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
BASELINE = ROOT / "contracts/deployment-bridge-stage5-performance-baseline.json"
BUDGET = (
    ROOT
    / "tests/fixtures/internal_ir/deployment_bridge_stage5_performance_budget_candidate.json"
)


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_stage5_baseline_is_measurement_only_and_deterministic() -> None:
    baseline = _load(BASELINE)

    assert baseline["status"] == "observed_baseline_not_budget"
    assert [case["observation_count"] for case in baseline["cases"]] == [
        10,
        100,
        1000,
        10000,
    ]
    assert all(case["deterministic_evidence_identity"] for case in baseline["cases"])


def test_stage5_budget_candidate_is_unapproved_and_above_baseline() -> None:
    baseline = _load(BASELINE)
    candidate = _load(BUDGET)

    assert candidate["status"] == "candidate_awaiting_owner_approval"
    assert candidate["approval"]["approved"] is False
    assert candidate["approval"]["approval_command"] == (
        "approve DEPLOYMENT-BRIDGE-STAGE5-PERFORMANCE-BUDGET"
    )
    budgets = {item["observation_count"]: item for item in candidate["budgets"]}
    assert set(budgets) == {10, 100, 1000, 10000}
    for case in baseline["cases"]:
        budget = budgets[case["observation_count"]]
        assert (
            case["offline_observation_p95_ms"]
            < budget["offline_observation_p95_ms_max"]
        )
        assert (
            case["offline_observation_peak_host_memory_bytes"]
            < budget["offline_observation_peak_host_memory_bytes_max"]
        )

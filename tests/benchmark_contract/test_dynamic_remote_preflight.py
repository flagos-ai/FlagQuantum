"""Contract tests for remote dynamic preflight performance evidence."""

from __future__ import annotations

import pytest

import flagquantum.benchmarking as runners
from flagquantum.benchmarking.dynamic_remote_preflight import (
    SCHEMA,
    run_benchmark,
)

pytestmark = pytest.mark.benchmark_contract


def _small_run() -> dict:
    return run_benchmark(
        feedback_pairs=(1, 2),
        warmup=0,
        iterations=2,
        static_iterations=2,
    )


def test_dynamic_remote_preflight_runner_is_registered() -> None:
    assert "dynamic_remote_preflight" in runners.names()
    spec = runners.describe("dynamic_remote_preflight")
    assert spec.hardware == "CPU only"
    assert callable(runners.resolve("dynamic_remote_preflight"))


def test_payload_is_fail_closed_and_records_no_remote_activity() -> None:
    payload = _small_run()

    assert payload["schema"] == SCHEMA
    assert payload["artifact_class"] == "measured_local_run"
    assert payload["release_gate_allowed"] is False
    assert payload["scalability_claim_allowed"] is False
    assert payload["scalability_blockers"]
    assert not any(payload["remote_activity"].values())
    assert payload["static_classifier_bypass"] == {
        "classified_dynamic": False,
        "timing": payload["static_classifier_bypass"]["timing"],
    }
    for case in payload["preflight_cases"]:
        assert case["compatible"] is False
        assert case["blocker_codes"]
        assert case["target"] == "quafu:Baihua"
        assert case["instruction_count"] == 2 * case["feedback_pairs"]


def test_budget_verdict_is_derived_from_observations() -> None:
    payload = _small_run()
    checks = payload["budget_checks"]

    assert checks["preflight_p95_seconds"] is all(
        case["timing"]["p95_seconds"] <= case["p95_budget_seconds"]
        for case in payload["preflight_cases"]
    )
    for case in payload["preflight_cases"]:
        assert case["p95_budget_seconds"] == (
            payload["budgets"]["preflight_fixed_p95_seconds"]
            + case["instruction_count"]
            * payload["budgets"]["preflight_per_instruction_p95_seconds"]
        )
        assert case["budget_passed"] is (
            case["timing"]["p95_seconds"] <= case["p95_budget_seconds"]
        )
    assert checks["static_classifier_p95_seconds"] is (
        payload["static_classifier_bypass"]["timing"]["p95_seconds"]
        <= payload["budgets"]["static_classifier_p95_seconds"]
    )
    assert payload["budget_passed"] is all(checks.values())


def test_fixed_evaluation_time_makes_decision_identities_reproducible() -> None:
    first = _small_run()
    second = _small_run()

    for left, right in zip(
        first["preflight_cases"], second["preflight_cases"], strict=True
    ):
        assert left["requirement_set_id"] == right["requirement_set_id"]
        assert left["snapshot_id"] == right["snapshot_id"]
        assert left["blocker_codes"] == right["blocker_codes"]


@pytest.mark.parametrize(
    ("keyword", "value", "message"),
    (
        ("feedback_pairs", (), "feedback_pairs"),
        ("feedback_pairs", (0,), "feedback_pairs"),
        ("warmup", -1, "warmup"),
        ("iterations", 0, "iterations"),
        ("static_iterations", 0, "static_iterations"),
    ),
)
def test_dynamic_remote_preflight_rejects_invalid_counts(
    keyword: str, value: object, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        run_benchmark(**{keyword: value})

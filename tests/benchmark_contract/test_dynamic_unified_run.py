"""Contract tests for stable local dynamic plan/run performance evidence."""

from __future__ import annotations

import pytest

import flagquantum.benchmarking as runners
from flagquantum.benchmarking.dynamic_unified_run import SCHEMA, run_benchmark

pytestmark = pytest.mark.benchmark_contract


def _small_run() -> dict:
    return run_benchmark(
        batched_shots=32,
        trajectory_shots=2,
        static_shots=32,
        warmup=0,
        iterations=2,
        planning_iterations=2,
    )


def test_dynamic_unified_run_runner_is_registered() -> None:
    assert "dynamic_unified_run" in runners.names()
    spec = runners.describe("dynamic_unified_run")
    assert spec.hardware == "CPU only"
    assert callable(runners.resolve("dynamic_unified_run"))


def test_dynamic_unified_run_payload_proves_paths_and_correctness() -> None:
    payload = _small_run()

    assert payload["schema"] == SCHEMA
    assert payload["artifact_class"] == "measured_local_run"
    assert payload["claim_evidence_type"] == "local_performance"
    assert payload["distribution_semantics"] == "single_process"
    assert payload["scalability_claim_allowed"] is False
    assert payload["release_gate_allowed"] is False

    cases = {case["name"]: case for case in payload["dynamic_cases"]}
    assert cases["batched"]["execution_strategy"] == "batched_statevector_kernel"
    assert (
        cases["trajectory_memory_fallback"]["execution_strategy"]
        == "trajectory_direct_statevector_kernel"
    )
    for case in cases.values():
        assert case["correctness"] == {
            "automatic_matches_explicit": True,
            "stable_matches_native_oracle": True,
        }
        assert case["ratios"]["explicit_over_native_median"] > 0
        assert case["ratios"]["automatic_over_explicit_median"] > 0
    assert payload["static_sentinel"]["correctness"] == {
        "automatic_matches_explicit": True
    }


def test_budget_verdict_is_derived_from_observations() -> None:
    payload = _small_run()
    checks = payload["budget_checks"]

    assert checks["planning_p95_seconds"] is (
        payload["planning"]["p95_seconds"] <= payload["budgets"]["planning_p95_seconds"]
    )
    assert checks["explicit_over_native_median"] is all(
        case["ratios"]["explicit_over_native_median"]
        <= payload["budgets"]["explicit_over_native_median"]
        for case in payload["dynamic_cases"]
    )
    assert checks["automatic_over_explicit_median"] is all(
        case["ratios"]["automatic_over_explicit_median"]
        <= payload["budgets"]["automatic_over_explicit_median"]
        for case in payload["dynamic_cases"]
    )
    assert checks["static_automatic_over_explicit_median"] is (
        payload["static_sentinel"]["ratios"]["automatic_over_explicit_median"]
        <= payload["budgets"]["static_automatic_over_explicit_median"]
    )
    assert payload["budget_passed"] is all(checks.values())


@pytest.mark.parametrize(
    ("keyword", "value", "message"),
    (
        ("batched_shots", 0, "batched_shots"),
        ("trajectory_shots", 0, "trajectory_shots"),
        ("static_shots", 0, "static_shots"),
        ("iterations", 0, "iterations"),
        ("planning_iterations", 0, "planning_iterations"),
        ("warmup", -1, "warmup"),
    ),
)
def test_dynamic_unified_run_rejects_invalid_measurement_counts(
    keyword: str, value: int, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        run_benchmark(**{keyword: value})

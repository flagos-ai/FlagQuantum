"""Contract tests for terminal-measurement performance evidence."""

from __future__ import annotations

import pytest

import flagquantum.benchmarking as runners
from flagquantum.benchmarking.dynamic_terminal_measurement import (
    SCHEMA,
    run_benchmark,
)

pytestmark = pytest.mark.benchmark_contract


def _small_run() -> dict:
    return run_benchmark(
        widths=(2, 5),
        count_entries=4,
        warmup=0,
        iterations=2,
    )


def test_terminal_measurement_runner_is_registered() -> None:
    assert "dynamic_terminal_measurement" in runners.names()
    spec = runners.describe("dynamic_terminal_measurement")
    assert spec.hardware == "CPU only"
    assert callable(runners.resolve("dynamic_terminal_measurement"))


def test_payload_records_correctness_and_no_remote_activity() -> None:
    payload = _small_run()

    assert payload["schema"] == SCHEMA
    assert payload["artifact_class"] == "measured_local_run"
    assert payload["release_gate_allowed"] is False
    assert payload["scalability_claim_allowed"] is False
    assert not any(payload["remote_activity"].values())
    for case in payload["cases"]:
        assert case["mapping_kind"] == "reverse_permutation"
        assert all(case["correctness"].values())
        assert len(case["normalization_id"]) == 64


def test_budget_verdict_is_derived_from_observations() -> None:
    payload = _small_run()
    budgets = payload["budgets"]

    for case in payload["cases"]:
        assert case["normalization_p95_budget_seconds"] == (
            budgets["normalization_fixed_p95_seconds"]
            + case["width"] * budgets["normalization_per_qubit_p95_seconds"]
        )
        assert case["projection_p95_budget_seconds"] == (
            budgets["projection_fixed_p95_seconds"]
            + case["width"]
            * case["count_entries"]
            * budgets["projection_per_bit_p95_seconds"]
        )
        assert case["normalization_budget_passed"] is (
            case["normalization_timing"]["p95_seconds"]
            <= case["normalization_p95_budget_seconds"]
        )
        assert case["projection_budget_passed"] is (
            case["projection_timing"]["p95_seconds"]
            <= case["projection_p95_budget_seconds"]
        )
    assert payload["budget_passed"] is all(payload["budget_checks"].values())


@pytest.mark.parametrize(
    ("keyword", "value", "message"),
    (
        ("widths", (), "widths"),
        ("widths", (0,), "widths"),
        ("count_entries", 0, "count_entries"),
        ("warmup", -1, "warmup"),
        ("iterations", 0, "iterations"),
    ),
)
def test_terminal_measurement_runner_rejects_invalid_counts(
    keyword: str, value: object, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        run_benchmark(**{keyword: value})

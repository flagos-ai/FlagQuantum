"""Protect timing boundaries and failure reporting of the fixed reference."""

import math

import pytest
import torch

import flagquantum as fq
from flagquantum.benchmarking import ftqc_exact_reference as reference

pytestmark = pytest.mark.benchmark_contract


def test_real_reference_preserves_interference_and_phase_accounting() -> None:
    report = reference.run_benchmark(warmup=1, iterations=2)
    rows = report["observations"]
    assert len(rows) == 3
    assert [row["warmup"] for row in rows] == [True, False, False]
    for row in rows:
        for path in ("explicit", "automatic"):
            assert row[f"{path}_probabilities"] == pytest.approx(
                [math.cos(math.pi / 8) ** 2, math.sin(math.pi / 8) ** 2], abs=1e-14
            )
            assert row[f"{path}_max_absolute_error"] < 1e-14
        assert row["construction_plan_execution_seconds"] == pytest.approx(
            row["construction_seconds"]
            + row["planning_seconds"]
            + row["explicit_execution_seconds"]
        )
    assert report["performance_targets_enabled"] is False
    assert report["scalability_claim_allowed"] is False


def test_execution_failure_is_not_a_successful_observation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(*args: object, **kwargs: object) -> fq.ExecutionResult:
        raise RuntimeError("execution failed")

    monkeypatch.setattr(fq, "run", fail)
    with pytest.raises(RuntimeError, match="execution failed"):
        reference.run_benchmark(warmup=0, iterations=1)


@pytest.mark.parametrize(
    "field,value",
    [("warmup", True), ("warmup", -1), ("iterations", 0), ("iterations", 11)],
)
def test_invalid_repetition_envelope(field: str, value: int) -> None:
    with pytest.raises(ValueError, match=field):
        reference.run_benchmark(**{field: value})


def test_rejects_multi_process_measurement(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WORLD_SIZE", "2")
    with pytest.raises(ValueError, match="one local process"):
        reference.run_benchmark()


@pytest.mark.parametrize(
    "values",
    [
        torch.zeros((2,), dtype=torch.float64),
        torch.zeros((1, 2), dtype=torch.float32),
        torch.tensor([[float("nan"), 0.0]], dtype=torch.float64),
    ],
)
def test_malformed_results_are_not_reported_as_numeric_evidence(
    values: torch.Tensor,
) -> None:
    result = fq.ExecutionResult(
        measurements=(
            fq.MeasurementResult(kind="probabilities", qubits=(0,), value=values),
        )
    )
    with pytest.raises(ValueError, match="H-T-H"):
        reference._probabilities(result)

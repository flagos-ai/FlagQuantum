"""Grouped estimators reject malformed histograms before computing statistics."""

import math

import pytest
import torch

from flagquantum.algorithms import Hamiltonian, HamiltonianTerm
from flagquantum.circuit import Circuit
from flagquantum.deployment import PauliMeasurementPlan, create_pauli_measurement_plan

pytestmark = pytest.mark.unit


@pytest.fixture
def measurement_plan() -> PauliMeasurementPlan:
    return create_pauli_measurement_plan(
        Circuit(1), Hamiltonian((HamiltonianTerm(1.0, {0: "z"}),)), shots=10
    )


@pytest.mark.parametrize(
    ("counts", "error_type", "message"),
    [
        ({"0": 11, "1": -1}, ValueError, "non-negative integers"),
        ({"0": 6.5, "1": 4.5}, TypeError, "non-negative integers"),
        ({"0": True, "1": 9}, TypeError, "non-negative integers"),
        ({"x": 10}, ValueError, "bitstrings must be binary"),
    ],
)
@pytest.mark.parametrize("statistic", ["expectation", "standard_error"])
def test_grouped_statistics_reject_invalid_histograms(
    measurement_plan: PauliMeasurementPlan,
    counts: dict[str, int | float],
    statistic: str,
    error_type: type[Exception],
    message: str,
) -> None:
    estimate = getattr(measurement_plan, statistic)
    with pytest.raises(error_type, match=message):
        estimate((counts,))


@pytest.mark.parametrize(
    ("counts", "mean", "error"),
    [({"0": 10, "1": 0}, 1.0, 0.0), ({"0": 5, "1": 5}, 0.0, math.sqrt(0.1))],
)
def test_grouped_statistics_preserve_valid_histograms(
    measurement_plan: PauliMeasurementPlan,
    counts: dict[str, int],
    mean: float,
    error: float,
) -> None:
    torch.testing.assert_close(
        measurement_plan.expectation((counts,)), torch.tensor([mean])
    )
    torch.testing.assert_close(
        measurement_plan.standard_error((counts,)), torch.tensor([error])
    )

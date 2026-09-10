"""Ungrouped count estimators reject invalid histograms and wire references."""

from collections.abc import Callable, Mapping

import pytest
import torch

from flagquantum.algorithms import Hamiltonian, HamiltonianTerm
from flagquantum.deployment import (
    expectation_z_from_counts,
    hamiltonian_expectation_from_counts,
)

pytestmark = pytest.mark.unit


@pytest.fixture(params=["z", "hamiltonian"])
def estimate(
    request: pytest.FixtureRequest,
) -> Callable[[Mapping[str, int]], torch.Tensor]:
    if request.param == "z":
        return expectation_z_from_counts
    hamiltonian = Hamiltonian((HamiltonianTerm(1.0, {0: "z"}),))
    return lambda counts: hamiltonian_expectation_from_counts(counts, hamiltonian)


@pytest.mark.parametrize(
    ("counts", "error_type"),
    [
        ({"0": 11, "1": -1}, ValueError),
        ({"0": 6.5, "1": 4.5}, TypeError),
        ({"0": True, "1": 9}, TypeError),
        ({"x": 10}, ValueError),
        ({"0": 0}, ValueError),
        ({"0": 2, "11": 2}, ValueError),
        ({"": 10}, ValueError),
    ],
)
def test_estimators_reject_malformed_histograms(
    estimate: Callable[[Mapping[str, int]], torch.Tensor],
    counts: Mapping[str, int],
    error_type: type[Exception],
) -> None:
    with pytest.raises(error_type):
        estimate(counts)


def test_estimators_preserve_valid_results(
    estimate: Callable[[Mapping[str, int]], torch.Tensor],
) -> None:
    actual = estimate({"0": 6, "1": 2})
    torch.testing.assert_close(actual, torch.full_like(actual, 0.5))


@pytest.mark.parametrize("wire", [-1, 1])
def test_z_estimator_rejects_out_of_range_wires(wire: int) -> None:
    with pytest.raises(ValueError, match="outside"):
        expectation_z_from_counts({"0": 10}, wires=wire)


def test_hamiltonian_estimator_rejects_out_of_range_wires() -> None:
    hamiltonian = Hamiltonian((HamiltonianTerm(1.0, {1: "z"}),))
    with pytest.raises(ValueError, match="outside"):
        hamiltonian_expectation_from_counts({"0": 10}, hamiltonian)

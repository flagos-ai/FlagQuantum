"""Measured Hamiltonians require finite real scalar coefficients."""

from dataclasses import replace

import pytest
import torch

from flagquantum.algorithms import Hamiltonian, HamiltonianTerm
from flagquantum.circuit import Circuit
from flagquantum.deployment import (
    create_pauli_measurement_plan,
    hamiltonian_expectation_from_counts,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("grouped", [False, True])
@pytest.mark.parametrize(
    "coefficient",
    [
        float("nan"),
        float("inf"),
        -float("inf"),
        1 + 0.5j,
        torch.ones(2),
        torch.empty(0),
    ],
)
def test_measured_coefficients_reject_nonfinite_or_nonreal_values(
    coefficient: float | complex | torch.Tensor, grouped: bool
) -> None:
    hamiltonian = Hamiltonian((HamiltonianTerm(coefficient, {0: "z"}),))
    if grouped:
        plan = create_pauli_measurement_plan(
            Circuit(1), Hamiltonian((HamiltonianTerm(1.0, {0: "z"}),)), shots=10
        )
        plan = replace(plan, hamiltonian=hamiltonian)
        with pytest.raises(ValueError, match="measured Hamiltonian coefficients"):
            plan.expectation(({"0": 10},))
        with pytest.raises(ValueError, match="measured Hamiltonian coefficients"):
            plan.standard_error(({"0": 10},))
    else:
        with pytest.raises(ValueError, match="measured Hamiltonian coefficients"):
            hamiltonian_expectation_from_counts({"0": 10}, hamiltonian)


@pytest.mark.parametrize("coefficient", [2.0, 2 + 0j, torch.tensor([2.0])])
def test_real_measured_coefficients_preserve_values_and_uncertainty(
    coefficient: float | complex | torch.Tensor,
) -> None:
    hamiltonian = Hamiltonian((HamiltonianTerm(coefficient, {0: "z"}),))
    counts = {"0": 6, "1": 4}
    plan = create_pauli_measurement_plan(Circuit(1), hamiltonian, shots=10)
    expected = torch.tensor([0.4])
    torch.testing.assert_close(
        hamiltonian_expectation_from_counts(counts, hamiltonian), expected
    )
    torch.testing.assert_close(plan.expectation((counts,)), expected)
    torch.testing.assert_close(
        plan.standard_error((counts,)), torch.tensor([(3.84 / 10) ** 0.5])
    )

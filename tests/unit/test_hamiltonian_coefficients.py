"""Complex coefficient inputs share real-energy semantics and preserve gradients."""

import math
import warnings

import pytest
import torch

import flagquantum as fq
from flagquantum.algorithms import Hamiltonian, pauli_term
from flagquantum.simulation.mps.state import MPSState

pytestmark = pytest.mark.unit


def test_mps_chain_combines_complex_coefficients_without_losing_gradients() -> None:
    local = torch.tensor([0.6, 0.8], dtype=torch.complex128)
    state = torch.kron(local, local)
    mps = MPSState.from_statevector(state, 2)
    coefficient = torch.tensor(0.7 + 0.2j, dtype=torch.complex128, requires_grad=True)
    hamiltonian = Hamiltonian(
        [
            pauli_term(coefficient, "Z", (0,)),
            pauli_term(0.2 + 0.3j, "Z", (0,)),
            pauli_term(coefficient, "ZZ", (0, 1)),
        ]
    )
    actual = hamiltonian.expectation(mps)
    expected = sum(term.expectation(state) for term in hamiltonian.terms)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    actual_gradient = torch.autograd.grad(actual.sum(), coefficient)[0]
    expected_gradient = torch.autograd.grad(expected.sum(), coefficient)[0]
    torch.testing.assert_close(
        actual_gradient, expected_gradient, atol=1e-12, rtol=1e-12
    )


@pytest.mark.parametrize("representation", ["circuit", "mps", "statevector", "density"])
@pytest.mark.parametrize("tensor_coefficient", [False, True])
def test_complex_pauli_coefficient_matches_dense_real_expectation(
    representation: str, tensor_coefficient: bool
) -> None:
    state = torch.tensor([0.6, 0.8], dtype=torch.complex128)
    target: fq.Circuit | MPSState | torch.Tensor
    if representation == "circuit":
        target = fq.Circuit(1, dtype=torch.complex128).ry(0, theta=2 * math.acos(0.6))
    elif representation == "mps":
        target = MPSState.from_statevector(state, 1)
    elif representation == "density":
        target = torch.outer(state, state.conj())
    else:
        target = state
    coefficient: complex | torch.Tensor = complex(1 + 2**-40, 0.7)
    if tensor_coefficient:
        coefficient = torch.tensor(
            coefficient, dtype=torch.complex128, requires_grad=True
        )

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = pauli_term(coefficient, "Z", (0,)).expectation(target)
    pauli_expectation = (state.conj() * torch.tensor([1, -1]) * state).sum()
    expected = (
        torch.as_tensor(coefficient, dtype=torch.complex128) * pauli_expectation
    ).real.reshape(1)
    torch.testing.assert_close(result, expected, atol=1e-12, rtol=1e-12)
    assert result.dtype == torch.float64
    if isinstance(coefficient, torch.Tensor):
        result.sum().backward()
        torch.testing.assert_close(coefficient.grad, pauli_expectation)

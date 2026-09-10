"""MPS chain Hamiltonians must not silently omit out-of-range terms."""

import pytest
import torch

from flagquantum.algorithms import Hamiltonian, pauli_term
from flagquantum.simulation.mps.state import MPSState

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("pauli", "wires"),
    [("Z", (-1,)), ("Z", (3,)), ("ZZ", (-1, 0)), ("ZZ", (2, 3))],
)
def test_chain_hamiltonian_rejects_terms_outside_the_mps(
    pauli: str, wires: tuple[int, ...]
) -> None:
    state = MPSState.zero(3)
    hamiltonian = Hamiltonian(
        [pauli_term(1.0, "Z", (0,)), pauli_term(2.0, pauli, wires)]
    )
    with pytest.raises(ValueError, match="outside the MPS"):
        hamiltonian.expectation(state)


@pytest.mark.parametrize(("pauli", "wires"), [("Z", (2,)), ("ZZ", (1, 2))])
def test_chain_hamiltonian_accepts_last_valid_term(
    pauli: str, wires: tuple[int, ...]
) -> None:
    state = MPSState.zero(3, dtype=torch.complex128)
    result = Hamiltonian([pauli_term(2.0, pauli, wires)]).expectation(state)
    torch.testing.assert_close(result, torch.tensor([2.0], dtype=torch.float64))

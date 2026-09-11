"""Hamiltonian construction accepts one-shot wire iterables without losing order."""

import torch

from flagquantum.algorithms import Hamiltonian, HamiltonianTerm


def test_hamiltonian_term_consumes_wire_generator_once() -> None:
    term = HamiltonianTerm(2.0, "xz", (wire for wire in (1, 0)))
    x = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex128)
    z = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex128)

    torch.testing.assert_close(Hamiltonian([term]).matrix(), 2 * torch.kron(z, x))
    assert term.wires == (1, 0)

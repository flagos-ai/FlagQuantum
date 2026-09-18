from __future__ import annotations

import pytest
import torch

import flagquantum as fq
from flagquantum.algorithms.primitives import append_qft, qft
from flagquantum.simulation.matrices import qft_matrix

pytestmark = pytest.mark.unit


def _basis_circuit(index: int, n_wires: int) -> fq.Circuit:
    """Return a circuit preparing the computational basis state ``index``.

    Qubit 0 is the most significant bit, matching ``qft_matrix``.
    """
    circuit = fq.Circuit(n_wires)
    for wire in range(n_wires):
        if (index >> (n_wires - 1 - wire)) & 1:
            circuit.x(wire)
    return circuit


def _apply_to_basis_states(n_wires: int, *, inverse: bool = False) -> torch.Tensor:
    """Return the matrix whose column k is the transform applied to |k>."""
    n_states = 2**n_wires
    applied = torch.zeros(n_states, n_states, dtype=torch.complex64)
    for index in range(n_states):
        circuit = _basis_circuit(index, n_wires)
        append_qft(circuit, list(range(n_wires)), inverse=inverse)
        applied[:, index] = circuit.state().reshape(-1)
    return applied


def test_qft_matches_the_dense_reference() -> None:
    """The circuit and the repository's dense generator agree."""
    for n_wires in (2, 3):
        applied = _apply_to_basis_states(n_wires)
        expected = qft_matrix(n_wires).to(torch.complex64)
        assert torch.allclose(applied, expected, atol=1e-5), n_wires


def test_inverse_qft_is_the_adjoint() -> None:
    """The inverse transform equals the conjugate transpose of the forward one."""
    for n_wires in (2, 3):
        applied = _apply_to_basis_states(n_wires, inverse=True)
        expected = qft_matrix(n_wires).to(torch.complex64).conj().T
        assert torch.allclose(applied, expected, atol=1e-5), n_wires


def test_qft_then_inverse_is_the_identity() -> None:
    """Appending the forward transform and then the inverse leaves a basis state alone."""
    n_wires = 3
    n_states = 2**n_wires
    for index in range(n_states):
        circuit = _basis_circuit(index, n_wires)
        append_qft(circuit, list(range(n_wires)))
        append_qft(circuit, list(range(n_wires)), inverse=True)
        expected = torch.zeros(n_states, dtype=torch.complex64)
        expected[index] = 1.0
        assert torch.allclose(circuit.state().reshape(-1), expected, atol=1e-5)


def test_qft_uses_the_expected_gate_count() -> None:
    """n Hadamards, n(n-1)/2 controlled phases, n//2 swaps."""
    for n_wires in (1, 2, 3, 4):
        expected = n_wires + n_wires * (n_wires - 1) // 2 + n_wires // 2
        assert len(qft(n_wires)) == expected, n_wires


def test_qft_rejects_a_non_positive_wire_count() -> None:
    """A zero-wire transform is refused rather than returning an empty circuit."""
    with pytest.raises(ValueError):
        qft(0)


def test_append_qft_rejects_repeated_wires() -> None:
    """A repeated wire would apply a phase to the wrong qubit."""
    with pytest.raises(ValueError):
        append_qft(fq.Circuit(2), [0, 0])

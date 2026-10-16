from __future__ import annotations

import pytest
import torch

import flagquantum as fq
from flagquantum.algorithms.primitives import append_qft, qft
from flagquantum.simulation.matrices import qft_matrix

pytestmark = pytest.mark.unit


def _basis_circuit(index: int, n_qubits: int) -> fq.Circuit:
    """Return a circuit preparing the computational basis state ``index``.

    Qubit 0 is the most significant bit, matching ``qft_matrix``.
    """
    circuit = fq.Circuit(n_qubits)
    for qubit in range(n_qubits):
        if (index >> (n_qubits - 1 - qubit)) & 1:
            circuit.x(qubit)
    return circuit


def _apply_to_basis_states(n_qubits: int, *, inverse: bool = False) -> torch.Tensor:
    """Return the matrix whose column k is the transform applied to |k>."""
    n_states = 2**n_qubits
    applied = torch.zeros(n_states, n_states, dtype=torch.complex64)
    for index in range(n_states):
        circuit = _basis_circuit(index, n_qubits)
        append_qft(circuit, list(range(n_qubits)), inverse=inverse)
        applied[:, index] = circuit.state().reshape(-1)
    return applied


def test_qft_matches_the_dense_reference() -> None:
    """The circuit and the repository's dense generator agree."""
    for n_qubits in (2, 3):
        applied = _apply_to_basis_states(n_qubits)
        expected = qft_matrix(n_qubits).to(torch.complex64)
        assert torch.allclose(applied, expected, atol=1e-5), n_qubits


def test_inverse_qft_is_the_adjoint() -> None:
    """The inverse transform equals the conjugate transpose of the forward one."""
    for n_qubits in (2, 3):
        applied = _apply_to_basis_states(n_qubits, inverse=True)
        expected = qft_matrix(n_qubits).to(torch.complex64).conj().T
        assert torch.allclose(applied, expected, atol=1e-5), n_qubits


def test_qft_then_inverse_is_the_identity() -> None:
    """Appending the forward transform and then the inverse leaves a basis state alone."""
    n_qubits = 3
    n_states = 2**n_qubits
    for index in range(n_states):
        circuit = _basis_circuit(index, n_qubits)
        append_qft(circuit, list(range(n_qubits)))
        append_qft(circuit, list(range(n_qubits)), inverse=True)
        expected = torch.zeros(n_states, dtype=torch.complex64)
        expected[index] = 1.0
        assert torch.allclose(circuit.state().reshape(-1), expected, atol=1e-5)


def test_qft_uses_the_expected_gate_count() -> None:
    """n Hadamards, n(n-1)/2 controlled phases, n//2 swaps."""
    for n_qubits in (1, 2, 3, 4):
        expected = n_qubits + n_qubits * (n_qubits - 1) // 2 + n_qubits // 2
        assert len(qft(n_qubits)) == expected, n_qubits


def test_qft_rejects_a_non_positive_wire_count() -> None:
    """A zero-qubit transform is refused rather than returning an empty circuit."""
    with pytest.raises(ValueError):
        qft(0)


def test_append_qft_rejects_repeated_wires() -> None:
    """A repeated qubit would apply a phase to the wrong qubit."""
    with pytest.raises(ValueError):
        append_qft(fq.Circuit(2), [0, 0])

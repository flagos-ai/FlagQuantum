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


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("width", [1, 3, 5])
@pytest.mark.parametrize("with_reverse", [False, True])
def test_pinned_qft_order_counts_and_complex_state(width, with_reverse):
    pytest.importorskip("qualtran")
    import cirq
    import numpy as np
    from qualtran.bloqs.basic_gates.rotation import CZPowGate
    from qualtran.bloqs.qft.qft_text_book import QFTTextBook

    from examples.algorithms.binary_field_arithmetic import FieldOperation
    from flagquantum.algorithms.primitives.qft import _qft_circuit

    qubits = tuple(cirq.LineQubit.range(width))
    source = QFTTextBook(width, with_reverse=with_reverse)
    stream = cirq.decompose(
        source.on(*qubits),
        keep=lambda op: op.gate in (cirq.H, cirq.SWAP)
        or isinstance(getattr(op.gate, "bloq", op.gate), CZPowGate),
    )
    reference = []
    for op in stream:
        wires = tuple(q.x for q in op.qubits)
        if op.gate == cirq.H:
            reference.append(("h", wires, 0))
        elif op.gate == cirq.SWAP:
            reference.append(("swap", wires, 0))
        else:
            bloq = getattr(op.gate, "bloq", op.gate)
            reference.append(("cphase", wires, np.pi * bloq.exponent))
    value = torch.arange(1, (1 << width) + 1, dtype=torch.float64)
    state = torch.complex(value, 2 - value)
    state /= state.norm()
    forward = _qft_circuit(width, with_reverse=with_reverse)
    for inverse in (False, True):
        actual = forward.adjoint() if inverse else forward
        expected = (
            [(name, wires, -angle) for name, wires, angle in reversed(reference)]
            if inverse
            else reference
        )
        instructions = tuple(actual.to_ir().instructions)
        assert [
            (g.name, g.wires, g.params.get("theta", 0)) for g in instructions
        ] == expected
        cirq_stream = [
            (
                cirq.H(qubits[wires[0]])
                if name == "h"
                else (
                    cirq.SWAP(*(qubits[q] for q in wires))
                    if name == "swap"
                    else (cirq.CZ ** (angle / np.pi))(*(qubits[q] for q in wires))
                )
            )
            for name, wires, angle in expected
        ]
        wanted = (
            cirq.Simulator(dtype=np.complex128)
            .simulate(
                cirq.Circuit(cirq_stream),
                qubit_order=qubits,
                initial_state=state.numpy(),
            )
            .final_state_vector
        )
        circuit = fq.Circuit(width, inputs=state, dtype=torch.complex128)
        circuit.append(FieldOperation(width, instructions))
        torch.testing.assert_close(
            circuit.state().flatten(), torch.from_numpy(wanted), atol=2e-12, rtol=0
        )

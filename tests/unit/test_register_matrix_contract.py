"""Source and existing physical-IR witnesses for explicit unitary Program leaves."""

import pytest
import torch

try:
    import cirq
    import numpy as np
except ModuleNotFoundError:
    pytest.skip(
        "Cirq reference dependencies are not installed", allow_module_level=True
    )

from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.simulation.unitary import get_unitary

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize("width", [1, 2])
@pytest.mark.parametrize("polarity", [0, 1])
def test_source_matrix_leaf_preserves_phase_order_and_spectator(
    width: int, polarity: int
) -> None:
    from qualtran.cirq_interop._cirq_to_bloq import cirq_optree_to_cbloq

    size = 2**width
    # A phased permutation makes global phase and operand order observable.
    matrix = np.roll(np.eye(size, dtype=np.complex128), 1, axis=0)
    matrix *= np.exp(0.29j)
    qubits = tuple(cirq.LineQubit.range(width + 2))
    operands = tuple(reversed(qubits[:-1]))
    leaf = cirq.MatrixGate(matrix).controlled(control_values=[polarity])
    operation = leaf.on(*operands)
    source = cirq_optree_to_cbloq(cirq.Circuit(operation))
    restored = source.to_cirq_circuit(
        cirq_quregs={"qubits": np.array([[q] for q in qubits[:-1]], dtype=object)}
    )
    expected = cirq.Circuit(operation).unitary(qubit_order=qubits)
    np.testing.assert_array_equal(restored.unitary(qubit_order=qubits), expected)
    physical = CircuitIR(
        n_wires=len(qubits),
        instructions=(
            Instruction(
                "any",
                tuple(q.x for q in operands),
                matrix=torch.tensor(cirq.unitary(leaf), dtype=torch.complex128),
            ),
        ),
    )
    actual = get_unitary(physical, dtype=torch.complex128).numpy()
    np.testing.assert_allclose(actual, expected, atol=2e-12, rtol=0)
    state = np.arange(1, 2 ** len(qubits) + 1, dtype=np.complex128)
    state *= np.exp(0.17j * np.arange(len(state)))
    state /= np.linalg.norm(state)
    np.testing.assert_allclose(actual @ state, expected @ state, atol=2e-12, rtol=0)

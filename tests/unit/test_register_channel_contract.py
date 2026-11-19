"""Frozen SDK channel transport and the existing physical density boundary."""

import pytest
import torch

try:
    import cirq
    import numpy as np
except ModuleNotFoundError:
    pytest.skip(
        "Cirq reference dependencies are not installed", allow_module_level=True
    )

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.circuit import Circuit
from flagquantum.core.ir import Instruction
from flagquantum.simulation.density_matrix import density_matrix_from_ir

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize("kind", ["amplitude", "depolarizing", "two_qubit"])
def test_source_channel_transport_and_native_density(kind: str) -> None:
    from qualtran.cirq_interop._cirq_to_bloq import cirq_optree_to_cbloq

    qubits = tuple(cirq.LineQubit.range(3))
    gate: cirq.Gate
    if kind == "amplitude":
        gate = cirq.amplitude_damp(0.25)
    elif kind == "depolarizing":
        gate = cirq.depolarize(0.2)
    else:
        gate = cirq.depolarize(0.2, n_qubits=2)
    width = cirq.num_qubits(gate)
    operands = tuple(reversed(qubits[:width]))
    source = cirq_optree_to_cbloq(cirq.Circuit(gate.on(*operands)))
    restored = source.to_cirq_circuit(
        cirq_quregs={"qubits": np.array([[q] for q in qubits[:width]], dtype=object)}
    )
    with pytest.raises(NotImplementedError, match="only supported for unitary"):
        source.tensor_contract()
    state = np.arange(1, 9, dtype=np.complex128) * np.exp(0.19j * np.arange(8))
    state /= np.linalg.norm(state)
    expected = (
        cirq.DensityMatrixSimulator(dtype=np.complex128)
        .simulate(
            restored,
            qubit_order=qubits,
            initial_state=state,
        )
        .final_density_matrix
    )
    operators = tuple(torch.tensor(k, dtype=torch.complex128) for k in cirq.kraus(gate))
    native = Circuit(3, inputs=torch.from_numpy(state), dtype=torch.complex128)
    native.append(
        FieldOperation(
            3,
            (
                Instruction(
                    "kraus",
                    tuple(q.x for q in operands),
                    matrix=operators,
                    metadata={"is_channel": True},
                ),
            ),
        )
    )
    actual = density_matrix_from_ir(native).reshape(8, 8)
    torch.testing.assert_close(actual, torch.from_numpy(expected), atol=3e-12, rtol=0)
    torch.testing.assert_close(
        torch.trace(actual), torch.ones((), dtype=torch.complex128), atol=3e-12, rtol=0
    )


def test_source_arbitrary_kraus_wrapper_reports_its_hashability_limit() -> None:
    from qualtran.cirq_interop._cirq_to_bloq import cirq_optree_to_cbloq

    gate = cirq.KrausChannel([np.eye(2)])
    with pytest.raises(TypeError, match="unhashable"):
        cirq_optree_to_cbloq(cirq.Circuit(gate.on(cirq.LineQubit(0))))

"""Frozen diagonal-gate examples, exact phase and original resource categories."""

from dataclasses import asdict
from hashlib import sha1
from importlib.metadata import version
from math import pi
from pathlib import Path

import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from examples.algorithms.diagonal_gates_tutorial import diagonal_gate_examples
from flagquantum.circuit import Circuit
from flagquantum.compiler._component_trace.qec_gate_cost import query_qec_gate_cost
from flagquantum.compiler._component_trace.resource_generalization import PhaseCall
from flagquantum.core.ir import Instruction
from flagquantum.simulation.unitary import get_unitary

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
SOURCE = (
    Path(__file__).resolve().parents[3]
    / "ftqc-reference-env/qualtran/bloqs/basic_gates"
)


def test_immutable_notebook_and_direct_source_leaves() -> None:
    import kickmix

    assert version("qualtran") == "0.7.0"
    assert kickmix.__version__ == "0.1" and hasattr(kickmix, "Simulator")
    for name, digest in (
        ("diag_gates.ipynb", "59ab7dd205aa110d7c35178398b93d02be34cca1"),
        ("z_basis.py", "48231b5d507f076ed3ea95a4d8486f375cca7d01"),
        ("z_basis_test.py", "35de62edcda9ce3230d31012f1fe757c4572bc38"),
        ("s_gate.py", "6111677a1e55ccab022e988598366a0f1bdcfc94"),
        ("s_gate_test.py", "98ed87f00f6788bd20a54eeed8b3770e738f13a8"),
    ):
        data = (SOURCE / name).read_bytes()
        assert sha1(f"blob {len(data)}\0".encode() + data).hexdigest() == digest


@pytest.mark.parametrize("index", range(3))
def test_complete_example_signatures_primitives_and_phase(index: int) -> None:
    import cirq
    from qualtran import QBit
    from qualtran.bloqs.basic_gates.s_gate import _s_gate
    from qualtran.bloqs.basic_gates.z_basis import _cz, _zgate
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    original = (_zgate, _s_gate, _cz)[index].make()
    name, signature, operation, key = diagonal_gate_examples()[index]
    assert all(isinstance(r.dtype, QBit) for r in original.signature)
    assert [(r.name, r.bitsize, r.shape_symbolic, r.side.value) for r in signature] == [
        (r.name, r.bitsize, r.shape_symbolic, r.side.value) for r in original.signature
    ]
    assert operation.instructions == (
        Instruction(name.lower(), tuple(range(operation.n_qubits))),
    )
    native_cost = query_qec_gate_cost(key)
    original_cost = get_cost_value(original, QECGatesCost())
    assert {
        kind: count for kind, count in asdict(native_cost).items() if count
    } == original_cost.asdict()
    circuit = Circuit(operation.n_qubits, dtype=torch.complex128)
    circuit.append(operation)
    matrix = get_unitary(circuit)
    expected = torch.as_tensor(original.tensor_contract(), dtype=torch.complex128)
    torch.testing.assert_close(matrix, expected, atol=1e-12, rtol=0)
    # Make the spectator's amplitudes distinct in every data branch.
    spectator_state = torch.arange(1, 2 * matrix.shape[0] + 1, dtype=torch.float64)
    state = torch.complex(spectator_state, spectator_state.flip(0))
    state /= state.norm()
    extended = Circuit(operation.n_qubits + 1, inputs=state, dtype=torch.complex128)
    extended.append(FieldOperation(operation.n_qubits + 1, operation.instructions))
    torch.testing.assert_close(
        extended.state().flatten(),
        (expected @ state.reshape(-1, 2)).flatten(),
        atol=1e-12,
        rtol=0,
    )
    control = operation.n_qubits
    actual = get_unitary(circuit.control(1, ctrl_qubits=(control,)))
    reference = cirq.unitary((cirq.Z, cirq.S, cirq.CZ)[index].controlled())
    # Native added control is the last physical wire, Cirq lists it first.
    axes = tuple(range(1, control + 1)) + (0,)
    order = (*axes, *(i + control + 1 for i in axes))
    controlled = (
        torch.as_tensor(reference)
        .reshape((2,) * (2 * (control + 1)))
        .permute(order)
        .reshape(actual.shape)
    )
    torch.testing.assert_close(actual, controlled, atol=1e-12, rtol=0)


def test_s_adjoint_square_and_controlled_phase_model() -> None:
    import cirq
    from qualtran.bloqs.basic_gates import SGate
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    _, _, op, _ = diagonal_gate_examples()[1]
    circuit = Circuit(1, dtype=torch.complex128)
    circuit.append(op).append(op)
    torch.testing.assert_close(
        get_unitary(circuit), torch.as_tensor(cirq.unitary(cirq.Z)), atol=1e-12, rtol=0
    )
    inverse = FieldOperation(1, (Instruction("sdg", (0,)),))
    circuit = Circuit(1, dtype=torch.complex128)
    circuit.append(inverse)
    source = SGate().adjoint()
    torch.testing.assert_close(
        get_unitary(circuit),
        torch.as_tensor(source.tensor_contract()),
        atol=1e-12,
        rtol=0,
    )
    cost = query_qec_gate_cost(PhaseCall("s", True))
    assert cost.clifford == get_cost_value(source, QECGatesCost()).clifford == 1
    controlled = circuit.control(1, ctrl_qubits=(1,))
    gates = controlled.to_ir().instructions
    assert gates[0].name == "cphase" and gates[0].wires == (1, 0)
    assert gates[0].params["theta"] == -pi / 2
    torch.testing.assert_close(
        get_unitary(controlled),
        torch.diag(torch.tensor([1, 1, 1, -1j], dtype=torch.complex128)),
        atol=1e-12,
        rtol=0,
    )
    circuit.append(op)
    torch.testing.assert_close(
        get_unitary(circuit), torch.eye(2, dtype=torch.complex128), atol=1e-12, rtol=0
    )

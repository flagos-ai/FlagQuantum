"""Execute owned matrix leaves through typed wrapping, expansion and binding."""

import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.circuit import Circuit
from flagquantum.compiler import estimate_resources
from flagquantum.compiler.registers.component_builder import ComponentProgramBuilder
from flagquantum.compiler.registers.component_gate_function import bit_gate_function
from flagquantum.compiler.registers.component_graph import copy_component_program
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.core.ir.program.register_bindings import RegisterBinding
from flagquantum.core.ir.program.register_types import quantum_bit_type
from flagquantum.core.ir.program.registers import Register
from flagquantum.core.ir.program.signatures import Signature

pytestmark = pytest.mark.unit


def test_matrix_survives_graph_copy_and_append_with_clean_work() -> None:
    matrix = torch.tensor([[0, 1j], [1j, 0]], dtype=torch.complex128)
    registers = (Register("q", quantum_bit_type()),)
    leaf = bit_gate_function(
        "leaf",
        registers,
        CircuitIR(n_wires=1, instructions=(Instruction("any", (0,), matrix=matrix),)),
    )
    builder = ComponentProgramBuilder(Signature(registers), functions=(leaf,))
    work = builder.allocate(dtype=quantum_bit_type())
    forward_work = builder.call_dict("leaf", {"q": work})["q"]
    assert isinstance(forward_work, RegisterBinding)
    # A supplied exact inverse leaves the work clean and tests full global phase.
    inverse = bit_gate_function(
        "inverse",
        registers,
        CircuitIR(
            n_wires=1, instructions=(Instruction("any", (0,), matrix=matrix.mH),)
        ),
    )
    builder.define(inverse)
    restored_work = builder.call_dict("inverse", {"q": forward_work})["q"]
    assert isinstance(restored_work, RegisterBinding)
    builder.free(restored_work)
    q = builder.call_dict("leaf", builder.inputs)["q"]
    module = builder.finalize({"q": q})
    copied = copy_component_program(module)
    bound = bind_register_program(copied, ((0,),), clean=(1,))
    assert estimate_resources(bound.circuit).operation_counts == {"any": 3}
    assert len(bound.events) == 2
    state = torch.tensor([1, 0, 1j, 0], dtype=torch.complex128) / 2**0.5
    expected = torch.tensor([-1, 0, 1j, 0], dtype=torch.complex128) / 2**0.5
    circuit = Circuit(2, inputs=state, dtype=torch.complex128)
    circuit.append(FieldOperation(2, bound.circuit.instructions))
    torch.testing.assert_close(circuit.state().flatten(), expected, atol=2e-12, rtol=0)
    assert torch.count_nonzero(circuit.state().flatten()[1::2]) == 0
    # Physical consumers own fresh storage and cannot mutate semantic rows.
    assert bound.circuit.instructions[0].matrix is not None
    bound.circuit.instructions[0].matrix.zero_()
    rebound = bind_register_program(module, ((0,),), clean=(1,))
    assert rebound.circuit.instructions[0].matrix is not None
    torch.testing.assert_close(rebound.circuit.instructions[0].matrix, matrix)


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("polarity", [0, 1])
def test_matrix_feedback_retains_each_complex_measurement_branch(polarity: int) -> None:
    try:
        import cirq
        import numpy as np
    except ModuleNotFoundError:
        pytest.skip("Cirq reference dependencies are not installed")

    from flagquantum.core.ir.program import ProgramRecord
    from flagquantum.ecosystem.cirq._feedback_import import import_cirq_feedback
    from flagquantum.ecosystem.cirq._register_gate import RegisterCirqGate
    from tests.unit.measurement_checks import forced_branch
    from tests.unit.test_cirq_feedback_import import source_branch

    qubits = tuple(cirq.LineQubit.range(3))
    matrix = np.array([[0, 1j], [1j, 0]], dtype=np.complex128)
    operation = (
        cirq.MatrixGate(matrix)
        .controlled(control_values=[polarity])
        .on(qubits[2], qubits[1])
    )
    source = cirq.Circuit(
        cirq.measure(qubits[0], key="branch"),
        operation.with_classical_controls("branch"),
        cirq.reset(qubits[0]),
    )
    imported = import_cirq_feedback(source, qubit_order=qubits)
    registers = tuple(Register(f"q{i}", quantum_bit_type()) for i in range(3))
    function = bit_gate_function("leaf", registers, imported.ir)
    bound = bind_register_program(
        ProgramRecord((function,)), ((0,), (1,), (2,)), entry="leaf"
    )
    assert bound.circuit.instructions[1].wires == (2, 1)
    assert bound.circuit.instructions[1].metadata["conditions"] == ((0, 1),)
    assert sum(op.name == "any" for op in bound.circuit.instructions) == 1
    exported = cirq.Circuit(
        cirq.decompose_once(
            RegisterCirqGate(
                bound, Signature(registers), feedback_key_prefix="matrix"
            ).on(*qubits)
        )
    )
    reimported = import_cirq_feedback(exported, qubit_order=qubits)
    initial = np.arange(1, 9, dtype=np.complex128) * np.exp(0.23j * np.arange(8))
    initial /= np.linalg.norm(initial)
    for outcome in (0, 1):
        actual = forced_branch(
            torch.from_numpy(initial), bound.circuit.instructions, (outcome,)
        )
        expected = torch.from_numpy(source_branch(source, qubits, initial, (outcome,)))
        torch.testing.assert_close(actual, expected, atol=2e-12, rtol=0)
        restored = forced_branch(
            torch.from_numpy(initial), reimported.ir.instructions, (outcome,)
        )
        torch.testing.assert_close(restored, expected, atol=2e-12, rtol=0)
        assert torch.count_nonzero(actual[4:]) == 0

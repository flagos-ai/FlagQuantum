"""SDK channel leaves retain operators, complete density and branch amplitudes."""

from dataclasses import replace

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
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir import Instruction
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.ecosystem.cirq._register_gate import RegisterCirqGate
from flagquantum.ecosystem.cirq._register_import import import_cirq_register_optree
from flagquantum.simulation.density_matrix import density_matrix_from_ir
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_cirq_feedback_import import source_branch

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize("kind", ["amplitude", "two_qubit", "custom"])
def test_channel_leaf_density_and_every_kraus_branch(kind: str) -> None:
    gate: cirq.Gate
    if kind == "amplitude":
        gate = cirq.amplitude_damp(0.25)
    elif kind == "two_qubit":
        gate = cirq.depolarize(0.2, n_qubits=2)
    else:
        gate = cirq.KrausChannel(
            [
                np.sqrt(0.7) * np.eye(2),
                np.sqrt(0.3) * np.diag([1j, -1j]),
            ]
        )
    width = cirq.num_qubits(gate)
    qubits = tuple(cirq.LineQubit.range(width + 1))
    operands = tuple(reversed(qubits[:width]))
    program = import_cirq_register_optree(gate.on(*operands))
    bound = bind_register_program(program, tuple((i,) for i in range(width)))
    assert len(bound.circuit.instructions) == 1
    assert bound.circuit.instructions[0].name == "kraus"
    assert bound.circuit.instructions[0].wires == tuple(q.x for q in operands)
    assert not bound.events
    registers = program.functions[0].registers
    assert registers is not None
    exported = RegisterCirqGate(bound, Signature(registers))
    restored = import_cirq_register_optree(exported.on(*qubits[:width]))
    rebound = bind_register_program(restored, tuple((i,) for i in range(width)))
    state = np.arange(1, 2 ** (width + 1) + 1, dtype=np.complex128)
    state *= np.exp(0.23j * np.arange(len(state)))
    state /= np.linalg.norm(state)
    expected = (
        cirq.DensityMatrixSimulator(dtype=np.complex128)
        .simulate(
            cirq.Circuit(gate.on(*operands)),
            qubit_order=qubits,
            initial_state=state,
        )
        .final_density_matrix
    )
    for body in (bound.circuit, rebound.circuit):
        circuit = Circuit(
            width + 1, inputs=torch.from_numpy(state), dtype=torch.complex128
        )
        circuit.append(FieldOperation(width + 1, body.instructions))
        actual = density_matrix_from_ir(circuit).reshape(len(state), len(state))
        torch.testing.assert_close(
            actual, torch.from_numpy(expected), atol=3e-12, rtol=0
        )
        assert body.instructions[0].matrix is not None
        for matrix, reference in zip(
            body.instructions[0].matrix, cirq.kraus(gate), strict=True
        ):
            torch.testing.assert_close(
                matrix,
                torch.as_tensor(reference, dtype=torch.complex128),
                atol=0,
                rtol=0,
            )
            branch = replace(
                body.instructions[0], name="any", matrix=matrix, metadata={}
            )
            actual_state = forced_branch(torch.from_numpy(state), (branch,), ())
            source = cirq.Circuit(
                cirq.MatrixGate(reference, unitary_check=False).on(*operands)
            )
            target = source_branch(source, qubits, state, ())
            torch.testing.assert_close(
                actual_state, torch.from_numpy(target), atol=3e-12, rtol=0
            )


@pytest.mark.parametrize("controlled", [False, True])
def test_channel_measurement_feedback_density_and_branches(controlled: bool) -> None:
    data, record, _ = qubits = tuple(cirq.LineQubit.range(3))
    gate = cirq.amplitude_damp(0.3)
    operation = gate.on(data)
    if controlled:
        operation = operation.with_classical_controls("record")
    source = cirq.Circuit(
        cirq.Moment((op,))
        for op in (
            cirq.H(record),
            cirq.measure(record, key="record"),
            operation,
            cirq.Y(data).with_classical_controls("record"),
            cirq.reset(record),
        )
    )
    program = import_cirq_register_optree(source)
    bound = bind_register_program(program, ((0,), (1,)))
    state = np.array([1, 2j, 0, 0, 3j, 4, 0, 0], dtype=np.complex128)
    state /= np.linalg.norm(state)
    registers = program.functions[0].registers
    assert registers is not None
    exported = RegisterCirqGate(
        bound, Signature(registers), feedback_key_prefix="channel"
    )
    restored = import_cirq_register_optree(exported.on(data, record))
    rebound = bind_register_program(restored, ((0,), (1,)))
    circuit = Circuit(3, inputs=torch.from_numpy(state), dtype=torch.complex128)
    circuit.append(FieldOperation(3, bound.circuit.instructions))
    channel = next(op for op in bound.circuit.instructions if op.name == "kraus")
    assert channel.matrix is not None
    # Sum the independently verified unnormalized instrument branches.
    total = torch.zeros((8, 8), dtype=torch.complex128)
    for outcome in (0, 1):
        for operator_index, reference in enumerate(cirq.kraus(gate)):
            if controlled and outcome == 0 and operator_index > 0:
                continue
            linear_channel = cirq.MatrixGate(reference, unitary_check=False).on(data)
            if controlled:
                linear_channel = linear_channel.with_classical_controls("record")
            linear = cirq.Circuit(
                cirq.Moment((op,))
                for op in (
                    cirq.H(record),
                    cirq.measure(record, key="record"),
                    linear_channel,
                    cirq.Y(data).with_classical_controls("record"),
                    cirq.reset(record),
                )
            )
            target = source_branch(linear, qubits, state, (outcome,))
            for body in (bound.circuit, rebound.circuit):
                gates = []
                for op in body.instructions:
                    if op.name == "kraus":
                        assert op.matrix is not None
                        op = replace(
                            op,
                            name="any",
                            matrix=op.matrix[operator_index],
                            metadata={
                                key: value
                                for key, value in op.metadata.items()
                                if key != "is_channel"
                            },
                        )
                    gates.append(op)
                actual = forced_branch(
                    torch.from_numpy(state), tuple(gates), (outcome,)
                )
                torch.testing.assert_close(
                    actual, torch.from_numpy(target), atol=3e-12, rtol=0
                )
                assert torch.count_nonzero(actual[[2, 3, 6, 7]]) == 0
            total += actual[:, None] @ actual.conj()[None, :]
    for body in (bound.circuit, rebound.circuit):
        circuit = Circuit(3, inputs=torch.from_numpy(state), dtype=torch.complex128)
        circuit.append(FieldOperation(3, body.instructions))
        torch.testing.assert_close(
            density_matrix_from_ir(circuit).reshape(8, 8), total, atol=3e-12, rtol=0
        )


def test_imported_channel_owns_data_through_copy_and_serialization() -> None:
    from flagquantum.compiler.registers.component_graph import copy_component_program
    from flagquantum.core.ir import CircuitIR

    first = np.sqrt(0.7) * np.eye(2, dtype=np.complex128)
    second = np.sqrt(0.3) * np.diag([1j, -1j])
    gate = cirq.KrausChannel([first, second])
    program = import_cirq_register_optree(gate.on(cirq.LineQubit(0)))
    first[:] = 0
    second[:] = 0
    copied = copy_component_program(program)
    bound = bind_register_program(copied, ((0,),))
    matrices = bound.circuit.instructions[0].matrix
    assert matrices is not None
    torch.testing.assert_close(
        matrices[0], np.sqrt(0.7) * torch.eye(2, dtype=torch.complex128)
    )
    restored = CircuitIR.from_dict(bound.circuit.to_dict())
    assert restored.to_dict() == bound.circuit.to_dict()
    matrices[0].zero_()
    rebound = bind_register_program(program, ((0,),))
    assert rebound.circuit.instructions[0].matrix is not None
    assert rebound.circuit.instructions[0].matrix[0][0, 0] != 0


def test_available_decomposition_precedes_channel_protocol() -> None:
    class Decomposable(cirq.Gate):
        def _num_qubits_(self) -> int:
            return 1

        def _decompose_(self, qubits: tuple[cirq.Qid, ...]) -> cirq.OP_TREE:
            return [cirq.H(qubits[0])]

        def _kraus_(self) -> object:
            raise AssertionError("available decomposition takes precedence")

    program = import_cirq_register_optree(Decomposable().on(cirq.LineQubit(0)))
    assert bind_register_program(program, ((0,),)).circuit.instructions == (
        Instruction("h", (0,)),
    )

"""Exercise recursive SDK leaves through native register Programs and append."""

from collections.abc import Sequence

import pytest
import torch

try:
    import cirq
    import numpy as np
    from numpy.typing import NDArray
except ModuleNotFoundError:
    pytest.skip(
        "Cirq reference dependencies are not installed", allow_module_level=True
    )

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.circuit import Circuit
from flagquantum.compiler import estimate_resources
from flagquantum.compiler.registers.component_builder import ComponentProgramBuilder
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir import Instruction
from flagquantum.core.ir.program.register_types import quantum_bit_type
from flagquantum.core.ir.program.registers import Register
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.ecosystem.cirq._register_gate import RegisterCirqGate
from flagquantum.ecosystem.cirq._register_import import (
    import_cirq_register_gate,
    import_cirq_register_optree,
)
from flagquantum.ecosystem.cirq.models import CirqConversionError
from flagquantum.simulation.unitary import get_unitary

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


class _UnitaryLeaf(cirq.Gate):
    def __init__(self, matrix: NDArray[np.complex128]) -> None:
        self.matrix_data = matrix

    def _num_qubits_(self) -> int:
        return 1

    def _unitary_(self) -> NDArray[np.complex128]:
        return self.matrix_data


class _NestedWork(cirq.Gate):
    def __init__(self, depth: int, *, borrow: bool = False) -> None:
        self.depth = depth
        self.borrow = borrow

    def _num_qubits_(self) -> int:
        return 1

    def _decompose_with_context_(
        self, qubits: Sequence[cirq.Qid], *, context: cirq.DecompositionContext
    ) -> cirq.OP_TREE:
        if self.depth == 0:
            return cirq.T.on(qubits[0])
        manager = context.qubit_manager
        work = (manager.qborrow(1) if self.borrow else manager.qalloc(1))[0]
        body = (
            cirq.CNOT(qubits[0], work),
            _NestedWork(self.depth - 1).on(work),
            cirq.CNOT(qubits[0], work),
        )
        manager.qfree((work,))
        return body


@pytest.mark.parametrize("width", [1, 2, 3])
def test_original_composite_wrapper_matches_full_operator_and_primitives(
    width: int,
) -> None:
    source = pytest.importorskip("qualtran.cirq_interop._bloq_to_cirq_test")
    upstream = source.BloqAsCirqGate(source.SwapTest(width))
    qubits = tuple(reversed(cirq.LineQubit.range(2 * width)))
    program = import_cirq_register_optree(upstream.on(*qubits))
    bound = bind_register_program(program, tuple((i,) for i in range(2 * width)))
    expected = tuple(
        Instruction("swap", (2 * width - 1 - i, width - 1 - i)) for i in range(width)
    )
    assert bound.circuit.instructions == expected
    assert not bound.events
    np.testing.assert_allclose(
        get_unitary(bound.circuit).numpy(),
        cirq.Circuit(upstream.on(*qubits)).unitary(qubit_order=tuple(sorted(qubits))),
        atol=2e-12,
        rtol=0,
    )
    assert estimate_resources(bound.circuit).operation_counts == {"swap": width}
    registers = program.functions[0].registers
    assert registers is not None
    exported = RegisterCirqGate(bound.circuit, Signature(registers))
    restored = import_cirq_register_optree(exported.on(*sorted(qubits)))
    assert (
        bind_register_program(
            restored, tuple((i,) for i in range(2 * width))
        ).circuit.instructions
        == expected
    )


@pytest.mark.parametrize("control_value", [0, 1])
def test_matrix_leaf_matches_source_through_native_program(control_value: int) -> None:
    source = pytest.importorskip("qualtran.cirq_interop._cirq_to_bloq")
    q0, q1 = cirq.LineQubit.range(2)
    matrix = np.array([[1, 0], [0, np.exp(0.37j)]], dtype=np.complex128)
    leaf = _UnitaryLeaf(matrix)
    operation = leaf.controlled(control_values=[control_value]).on(q1, q0)
    upstream = source.cirq_optree_to_cbloq(cirq.Circuit(operation))
    expected = upstream.to_cirq_circuit(
        cirq_quregs={"qubits": np.array([[q0], [q1]], dtype=object)}
    ).unitary(qubit_order=(q0, q1))
    np.testing.assert_allclose(
        expected,
        cirq.Circuit(operation).unitary(qubit_order=(q0, q1)),
        atol=2e-12,
        rtol=0,
    )
    program = import_cirq_register_optree(operation)
    bound = bind_register_program(program, ((0,), (1,)))
    assert estimate_resources(bound.circuit).operation_counts == {"any": 1}
    np.testing.assert_allclose(
        get_unitary(bound.circuit, dtype=torch.complex128).numpy(),
        expected,
        atol=2e-12,
        rtol=0,
    )
    registers = program.functions[0].registers
    assert registers is not None
    exported = RegisterCirqGate(bound, Signature(registers))
    restored = import_cirq_register_optree(exported.on(q0, q1))
    rebound = bind_register_program(restored, ((0,), (1,)))
    np.testing.assert_allclose(
        get_unitary(rebound.circuit, dtype=torch.complex128).numpy(),
        expected,
        atol=2e-12,
        rtol=0,
    )


def test_nested_workspace_reserves_siblings_and_roundtrips_native_gate() -> None:
    # Deliberately collide with the SDK manager's first candidate name.
    qubit = cirq.NamedQubit("register_work_0")
    module = import_cirq_register_optree(_NestedWork(2).on(qubit))
    bound = bind_register_program(module, ((0,),), clean=(1, 2))
    assert bound.circuit.instructions == (
        Instruction("cx", (0, 1)),
        Instruction("cx", (1, 2)),
        Instruction("t", (2,)),
        Instruction("cx", (1, 2)),
        Instruction("cx", (0, 1)),
    )
    assert bound.outputs == ((0,),)
    assert [(event.kind, event.wires) for event in bound.events] == [
        ("allocate", (1,)),
        ("allocate", (2,)),
        ("free", (2,)),
        ("free", (1,)),
    ]
    state = torch.zeros(8, dtype=torch.complex128)
    state[[0, 4]] = torch.tensor([1, 1j], dtype=torch.complex128) / 2**0.5
    expected = state.clone()
    expected[4] *= np.exp(1j * np.pi / 4)
    circuit = Circuit(3, inputs=state, dtype=torch.complex128)
    circuit.append(FieldOperation(3, bound.circuit.instructions))
    torch.testing.assert_close(circuit.state().flatten(), expected, atol=2e-12, rtol=0)
    assert torch.count_nonzero(circuit.state().flatten()[[1, 2, 3, 5, 6, 7]]) == 0
    assert estimate_resources(bound.circuit).operation_counts == {"cx": 4, "t": 1}
    registers = module.functions[0].registers
    assert registers is not None
    gate = RegisterCirqGate(bound, Signature(registers))
    restored = import_cirq_register_optree(gate.on(qubit))
    restored_bound = bind_register_program(restored, ((0,),), clean=(1, 2))
    assert restored_bound.circuit.instructions == bound.circuit.instructions


def test_explicit_native_definition_precedes_foreign_decomposition() -> None:
    gate = _NestedWork(1, borrow=True)
    bit = quantum_bit_type()
    builder = ComponentProgramBuilder(Signature((Register("q", bit),)), name="identity")
    identity = builder.finalize(builder.inputs).functions[0]
    module = import_cirq_register_optree(
        gate.on(cirq.LineQubit(0)), gate_definitions={gate: identity}
    )
    assert not bind_register_program(module, ((0,),)).circuit.instructions


@pytest.mark.parametrize("entry", ["optree", "gate"])
def test_dirty_decomposition_is_rejected_before_clean_assumption(entry: str) -> None:
    gate = _NestedWork(1, borrow=True)
    with pytest.raises(ValueError, match="clean decomposition workspace"):
        if entry == "gate":
            import_cirq_register_gate(gate)
        else:
            import_cirq_register_optree(gate.on(cirq.LineQubit(0)))


def test_cyclic_missing_and_empty_decompositions_have_distinct_outcomes() -> None:
    class Cyclic(cirq.Gate):
        def _num_qubits_(self) -> int:
            return 1

        def _decompose_(self, qubits: Sequence[cirq.Qid]) -> cirq.OP_TREE:
            return self.on(*qubits)

    class Empty(cirq.Gate):
        def _num_qubits_(self) -> int:
            return 1

        def _decompose_(self, qubits: Sequence[cirq.Qid]) -> cirq.OP_TREE:
            return ()

    class Missing(cirq.Gate):
        def _num_qubits_(self) -> int:
            return 1

    qubit = cirq.LineQubit(0)
    with pytest.raises(ValueError, match="cyclic"):
        import_cirq_register_optree(Cyclic().on(qubit))
    with pytest.raises(CirqConversionError):
        import_cirq_register_optree(Missing().on(qubit))
    assert import_cirq_register_gate(Missing()) is None
    empty = import_cirq_register_optree(Empty().on(qubit))
    bound = bind_register_program(empty, ((0,),))
    assert bound.outputs == ((0,),) and not bound.circuit.instructions

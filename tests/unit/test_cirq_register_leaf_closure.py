"""Exercise recursive SDK leaves through native register Programs and append."""

from collections.abc import Hashable, Sequence

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
from flagquantum.core.ir.program import FunctionRecord
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


@pytest.mark.parametrize("width", [1, 2, 3, 4])
def test_original_tensor_only_wrapper_matches_full_operator(width: int) -> None:
    source = pytest.importorskip("qualtran.cirq_interop._bloq_to_cirq_test")
    gate = source.BloqAsCirqGate(source.SwapTestWithOnlyTensorData(width))
    qubits = tuple(reversed(cirq.LineQubit.range(2 * width)))
    assert cirq.decompose_once(gate.on(*qubits), default=None) is None
    expected = cirq.Circuit(gate.on(*qubits)).unitary(qubit_order=sorted(qubits))
    program = import_cirq_register_optree(gate.on(*qubits))
    bound = bind_register_program(program, tuple((i,) for i in range(2 * width)))
    assert estimate_resources(bound.circuit).operation_counts == {"any": 1}
    assert not bound.events
    actual = get_unitary(bound.circuit, dtype=torch.complex128).numpy()
    np.testing.assert_allclose(actual, expected, atol=2e-12, rtol=0)
    state = np.arange(1, 2 ** (2 * width + 1) + 1, dtype=np.complex128)
    state *= np.exp(0.17j * np.arange(len(state)))
    state /= np.linalg.norm(state)
    circuit = Circuit(
        2 * width + 1, inputs=torch.from_numpy(state), dtype=torch.complex128
    )
    circuit.append(FieldOperation(2 * width + 1, bound.circuit.instructions))
    torch.testing.assert_close(
        circuit.state().flatten(),
        torch.from_numpy(np.kron(expected, np.eye(2)) @ state),
        atol=2e-12,
        rtol=0,
    )
    registers = program.functions[0].registers
    assert registers is not None
    exported = RegisterCirqGate(bound, Signature(registers))
    restored = import_cirq_register_optree(exported.on(*sorted(qubits)))
    rebound = bind_register_program(restored, tuple((i,) for i in range(2 * width)))
    np.testing.assert_allclose(
        get_unitary(rebound.circuit, dtype=torch.complex128).numpy(),
        expected,
        atol=2e-12,
        rtol=0,
    )


def test_available_decomposition_precedes_tensor_fallback() -> None:
    class Decomposable(cirq.Gate):
        def _num_qubits_(self) -> int:
            return 1

        def _decompose_(self, qubits: Sequence[cirq.Qid]) -> cirq.OP_TREE:
            return [cirq.H(qubits[0])]

        def _unitary_(self) -> NDArray[np.complex128]:
            raise AssertionError("decomposable gates must retain their call chain")

    program = import_cirq_register_optree(Decomposable().on(cirq.LineQubit(0)))
    assert bind_register_program(program, ((0,),)).circuit.instructions == (
        Instruction("h", (0,)),
    )


@pytest.mark.parametrize("values", [(1, 1), (0, 1), (1, 0), (0, 0)])
def test_recursive_original_and_erasure_retains_all_histories(
    values: tuple[int, int],
) -> None:
    from itertools import product

    from qualtran import BloqBuilder, QBit
    from qualtran import Register as SourceRegister
    from qualtran import Signature as SourceSignature
    from qualtran.bloqs.mcmt import And
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    from flagquantum.ecosystem.cirq._feedback_import import import_cirq_feedback
    from tests.unit.measurement_checks import forced_branch
    from tests.unit.test_cirq_feedback_import import source_branch

    builder, inputs = BloqBuilder.from_signature(
        SourceSignature((SourceRegister("ctrl", QBit(), shape=(2,)),))
    )
    controls = inputs["ctrl"]
    for _ in range(2):
        controls, target = builder.add(And(*values), ctrl=controls)
        controls = builder.add(
            And(*values, uncompute=True), ctrl=controls, target=target
        )
    composite = builder.finalize(ctrl=controls)
    q0, q1, spectator = cirq.NamedQubit.range(3, prefix="data")
    quregs = {"ctrl": np.array([[q0], [q1]], dtype=object)}
    source_body = composite.to_cirq_circuit(
        cirq_quregs=quregs,
        qubit_manager=cirq.GreedyQubitManager(
            prefix="register_work", maximize_reuse=True
        ),
    )
    signature = Signature((Register("ctrl", quantum_bit_type(), shape=(2,)),))
    program = import_cirq_register_optree(
        source_body, signature=signature, in_quregs=quregs, out_quregs=quregs
    )
    bound = bind_register_program(program, ((0,), (1,)), clean=(2,))
    primitives = (
        cirq.HPowGate,
        cirq.XPowGate,
        cirq.YPowGate,
        cirq.ZPowGate,
        cirq.CXPowGate,
        cirq.CZPowGate,
        cirq.MeasurementGate,
        cirq.ResetChannel,
    )
    source_ops = cirq.decompose(
        source_body,
        context=cirq.DecompositionContext(
            cirq.GreedyQubitManager(prefix="register_work", maximize_reuse=True)
        ),
        keep=lambda op: isinstance(op.without_classical_controls().gate, primitives),
    )
    source = cirq.Circuit(cirq.Moment((op,)) for op in source_ops)
    (work,) = source.all_qubits() - {q0, q1}
    order = (q0, q1, work, spectator)
    expected = import_cirq_feedback(source, qubit_order=order[:3]).ir
    assert bound.circuit.instructions == expected.instructions
    assert len(bound.circuit.instructions) == 34 + 8 * values.count(0)
    assert sum(op.name == "measure" for op in bound.circuit.instructions) == 2
    model = get_cost_value(composite, QECGatesCost())
    assert (model.and_bloq, model.clifford, model.measurement) == (2, 2, 2)
    assert [(event.kind, event.wires) for event in bound.events] == [
        ("allocate", (2,)),
        ("free", (2,)),
    ]
    state = np.zeros(16, dtype=np.complex128)
    indices = [i for i in range(16) if not i & 2]
    state[indices] = np.arange(1, 9) * np.exp(0.19j * np.arange(8))
    state /= np.linalg.norm(state)
    from flagquantum.dynamic import DynamicCircuit
    from flagquantum.runtime.dynamic.execution import run_dynamic

    runtime = DynamicCircuit(4, inputs=torch.from_numpy(state), dtype=torch.complex128)
    runtime.append(FieldOperation(4, bound.circuit.instructions))
    result = run_dynamic(runtime, shots=8, seed=53, strategy="trajectory")
    torch.testing.assert_close(
        result.final_states.reshape(8, -1),
        torch.from_numpy(state).expand(8, -1),
        atol=3e-12,
        rtol=0,
    )
    registers = program.functions[0].registers
    assert registers is not None
    exported = RegisterCirqGate(
        bound, Signature(registers), feedback_key_prefix="nested"
    )
    restored = import_cirq_register_optree(exported.on(q0, q1))
    rebound = bind_register_program(restored, ((0,), (1,)), clean=(2,))
    for history in product((0, 1), repeat=2):
        expected_state = torch.from_numpy(source_branch(source, order, state, history))
        for circuit_ir in (bound.circuit, rebound.circuit):
            actual = forced_branch(
                torch.from_numpy(state), circuit_ir.instructions, history
            )
            torch.testing.assert_close(actual, expected_state, atol=3e-12, rtol=0)
            assert torch.count_nonzero(actual[[i for i in range(16) if i & 2]]) == 0


def test_recursive_feedback_preserves_old_inverted_and_latest_records() -> None:
    from itertools import product

    from tests.unit.measurement_checks import forced_branch
    from tests.unit.test_cirq_feedback_import import source_branch

    a, b = cirq.NamedQubit.range(2, prefix="data")
    key = cirq.MeasurementKey("record")
    circuit = cirq.Circuit(
        cirq.H(a),
        cirq.measure(a, key=key, invert_mask=(True,)),
        cirq.reset(a),
        cirq.H(a),
        cirq.measure(a, key=key),
        _NestedWork(1).on(b).with_classical_controls(cirq.KeyCondition(key, index=0)),
        cirq.Y(b).with_classical_controls(key),
        cirq.reset(a),
    )
    program = import_cirq_register_optree(circuit)
    bound = bind_register_program(program, ((0,), (1,)), clean=(2,))
    source_ops = cirq.decompose(
        circuit,
        context=cirq.DecompositionContext(
            cirq.GreedyQubitManager(prefix="register_work", maximize_reuse=True)
        ),
        keep=lambda op: not isinstance(
            op.without_classical_controls().gate, _NestedWork
        ),
    )
    source = cirq.Circuit(cirq.Moment((op,)) for op in source_ops)
    (work,) = source.all_qubits() - {a, b}
    state = np.array([1, 0, 1j, 0, 2j, 0, 3, 0], dtype=np.complex128)
    state /= np.linalg.norm(state)
    for history in product((0, 1), repeat=2):
        actual = forced_branch(
            torch.from_numpy(state), bound.circuit.instructions, history
        )
        expected = torch.from_numpy(source_branch(source, (a, b, work), state, history))
        torch.testing.assert_close(actual, expected, atol=3e-12, rtol=0)
        assert torch.count_nonzero(actual[1::2]) == 0


@pytest.mark.parametrize("kind", ["undefined", "noisy_measure", "tagged", "definition"])
def test_recursive_feedback_rejects_unrepresented_history(kind: str) -> None:
    a, b = cirq.LineQubit.range(2)
    measurement = cirq.measure(a, key="record")
    definitions: dict[Hashable, FunctionRecord] = {}
    if kind == "undefined":
        operations = [cirq.X(b).with_classical_controls("missing")]
    elif kind == "noisy_measure":
        operations = [
            measurement,
            cirq.MeasurementGate(1, key="other", confusion_map={(0,): np.eye(2)}).on(b),
        ]
    elif kind == "tagged":
        operations = [
            measurement,
            cirq.X(b).with_classical_controls("record").with_tags("tag"),
        ]
    else:
        leaf = _NestedWork(1)
        builder = ComponentProgramBuilder(
            Signature((Register("q", quantum_bit_type()),))
        )
        definitions[leaf] = builder.finalize(builder.inputs).functions[0]
        operations = [measurement, leaf.on(b)]
    with pytest.raises((ValueError, CirqConversionError)):
        import_cirq_register_optree(
            cirq.Circuit(operations), gate_definitions=definitions
        )


@pytest.mark.parametrize("exponent", [1, -1])
def test_register_feedback_retains_iswap_phase(exponent: int) -> None:
    from tests.unit.measurement_checks import forced_branch
    from tests.unit.test_cirq_feedback_import import source_branch

    a, b, record = cirq.LineQubit.range(3)
    circuit = cirq.Circuit(
        cirq.H(record),
        cirq.measure(record, key="record"),
        (cirq.ISWAP**exponent).on(b, a).with_classical_controls("record"),
        cirq.reset(record),
    )
    bound = bind_register_program(
        import_cirq_register_optree(circuit), ((0,), (1,), (2,))
    )
    assert bound.circuit.instructions[2].name == (
        "iswap" if exponent == 1 else "iswapdg"
    )
    assert bound.circuit.instructions[2].wires == (1, 0)
    state = np.array([1, 0, 2j, 0, -3, 0, 4j, 0], dtype=np.complex128)
    state /= np.linalg.norm(state)
    for history in ((0,), (1,)):
        actual = forced_branch(
            torch.from_numpy(state), bound.circuit.instructions, history
        )
        expected = source_branch(circuit, (a, b, record), state, history)
        torch.testing.assert_close(
            actual, torch.from_numpy(expected), atol=3e-12, rtol=0
        )
        assert torch.count_nonzero(actual[1::2]) == 0

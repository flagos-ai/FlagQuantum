"""Check source OnEach wiring, full complex action and modeled multiplicity."""

from collections import Counter
from dataclasses import asdict, replace
from typing import Any, Literal, SupportsIndex

import pytest
import torch

import flagquantum as fq
from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.compiler._component_trace.qec_gate_cost import query_qec_gate_cost
from flagquantum.compiler._component_trace.resource_generalization import (
    CliffordCall,
    OnEachCall,
    RotationCall,
)
from flagquantum.compiler._component_trace.selected_call_count import (
    count_selected_calls,
)
from flagquantum.compiler.registers.bits.bitwise import bitwise_not_ir, on_each_ir
from flagquantum.compiler.registers.bits.on_each import on_each_word_program
from flagquantum.compiler.registers.component_gate_function import bit_gate_function
from flagquantum.compiler.registers.component_graph import iter_component_calls
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.core.ir.program import FunctionRecord, verify_program_record
from flagquantum.core.ir.program.register_types import unsigned_register_type
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.simulation.classical_on_each import on_each_classical_value
from flagquantum.simulation.classical_pauli_values import x_classical_value
from flagquantum.simulation.integer_encoding import integer_from_bits, integer_to_bits
from flagquantum.simulation.unitary import get_unitary
from tests.unit.register_source_graph import (
    SourcePorts,
    SourceResult,
    walk_source_graph,
)

pytestmark = pytest.mark.unit


def bit_child(opcode: str) -> FunctionRecord:
    return bit_gate_function(
        opcode, tuple(Signature.build(q=1)), CircuitIR(1, (Instruction(opcode, (0,)),))
    )


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("opcode,width", [("x", 10), ("h", 5), ("h", 10)])
def test_original_word_program_and_resource_counts(
    opcode: Literal["x", "h"], width: int
) -> None:
    pytest.importorskip("qualtran")
    from qualtran.bloqs.basic_gates.hadamard import Hadamard
    from qualtran.bloqs.basic_gates.on_each import OnEach
    from qualtran.bloqs.basic_gates.x_basis import XGate
    from qualtran.resource_counting.generalizers import ignore_split_join
    from qualtran.testing import assert_valid_bloq_decomposition

    leaf = XGate() if opcode == "x" else Hadamard()
    source = OnEach(width, leaf)
    assert_valid_bloq_decomposition(source)
    body = source.decompose_bloq()
    source_calls = [item.bloq for item, _, _ in body.iter_bloqsoqs()]
    assert [type(call).__name__ for call in source_calls] == [
        "Split",
        *([type(leaf).__name__] * width),
        "Join",
    ]
    record = on_each_word_program(width, bit_child(opcode))
    assert verify_program_record(record) == ()
    main = next(function for function in record.functions if function.name == "main")
    assert main.registers is not None
    assert Signature(main.registers).n_qubits() == source.signature.n_qubits()
    calls = [
        main.blocks[0].operations[i].callee for i, _, _ in iter_component_calls(record)
    ]
    assert len(calls) == width + 2 and calls[1:-1] == [opcode] * width
    wires = tuple(reversed(range(width)))
    bound = bind_register_program(record, (wires,))
    assert bound.outputs == (wires,) and not bound.events
    instructions = bound.circuit.instructions
    assert instructions == tuple(Instruction(opcode, (wire,)) for wire in wires)

    key = CliffordCall(opcode, 1)
    _, leaves = source.call_graph()
    declared = count_selected_calls(OnEachCall(key, width), (key,))
    assert (
        declared[key]
        == leaves[leaf]
        == Counter(gate.name for gate in instructions)[opcode]
    )
    assert source.bloq_counts(generalizer=ignore_split_join) == body.bloq_counts(
        generalizer=ignore_split_join
    )
    if opcode == "h" and width == 5:
        circuit = fq.Circuit(width, dtype=torch.complex128)
        circuit.append(FieldOperation(width, instructions))
        torch.testing.assert_close(
            get_unitary(circuit),
            torch.as_tensor(source.tensor_contract()),
            rtol=0,
            atol=2e-12,
        )


@pytest.mark.ftqc_reference
def test_original_ten_bit_classical_action_and_refusal() -> None:
    pytest.importorskip("qualtran")
    from qualtran.bloqs.basic_gates.hadamard import Hadamard
    from qualtran.bloqs.basic_gates.on_each import OnEach
    from qualtran.bloqs.basic_gates.x_basis import XGate

    def apply_x(bit: object) -> tuple[object, complex]:
        assert isinstance(bit, SupportsIndex)
        return x_classical_value(bit.__index__()), 1.0

    def unsupported_h(bit: object) -> tuple[object, complex]:
        assert bit == 0
        raise NotImplementedError("H is not classically simulable.")

    args = (
        0,
        10,
        lambda q: integer_to_bits(q, 10),
        lambda bits: integer_from_bits(bits, 10),
    )
    actual, phase = on_each_classical_value(*args, apply_x, lambda _: None)
    assert (actual,) == OnEach(10, XGate()).call_classically(q=0)
    assert actual == 1023 and phase == 1
    with pytest.raises(NotImplementedError, match=r"H is not classically simulable\."):
        OnEach(10, Hadamard()).call_classically(q=0)
    with pytest.raises(NotImplementedError, match=r"H is not classically simulable\."):
        on_each_classical_value(*args, unsupported_h, lambda _: None)


def test_word_program_admission_and_zero_width() -> None:
    child = bit_child("x")
    bound = bind_register_program(on_each_word_program(0, child), ((),), borrowed=(0,))
    assert bound.outputs == ((),) and bound.circuit.instructions == ()
    with pytest.raises(ValueError, match="scalar quantum THRU port named q"):
        on_each_word_program(2, replace(child, registers=tuple(Signature.build(r=1))))
    with pytest.raises(ValueError, match="dtype must match"):
        on_each_word_program(2, child, target_dtype=unsigned_register_type(3))
    with pytest.raises(ValueError, match="nonnegative integer width"):
        on_each_word_program(-1, child)


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("opcode", ["x", "h", "rz"])
@pytest.mark.parametrize("dtype", ["raw", "unsigned"])
def test_on_each_source_order_and_entangled_amplitudes(
    opcode: Literal["x", "h", "rz"], dtype: str
) -> None:
    pytest.importorskip("qualtran")
    from qualtran._infra.data_types import QUInt
    from qualtran.bloqs.basic_gates.hadamard import Hadamard
    from qualtran.bloqs.basic_gates.on_each import OnEach
    from qualtran.bloqs.basic_gates.rotation import Rz
    from qualtran.bloqs.basic_gates.x_basis import XGate
    from qualtran.resource_counting._bloq_counts import QECGatesCost
    from qualtran.resource_counting._costing import get_cost_value
    from qualtran.resource_counting._qubit_counts import QubitCount

    angle = 0.37
    leaf = {"x": XGate(), "h": Hadamard(), "rz": Rz(angle)}[opcode]
    wires = (3, 0, 2)
    source = OnEach(3, leaf, QUInt(3) if dtype == "unsigned" else None)
    body = Instruction(opcode, (1,), {"theta": angle} if opcode == "rz" else {})
    gates = on_each_ir(body, wires)

    def source_leaf(
        bloq: Any, inputs: SourcePorts, desired: SourcePorts
    ) -> SourceResult | None:
        if bloq != leaf:
            return None
        params = {"theta": bloq.angle} if isinstance(bloq, Rz) else {}
        return (Instruction(opcode, inputs["q"], params),), inputs

    expected, outputs = walk_source_graph(
        source, {"q": wires}, {"q": wires}, source_leaf
    )
    assert outputs == {"q": wires}
    assert gates == expected
    assert all(g.wires == (wire,) for g, wire in zip(gates, wires, strict=True))
    assert all(g.params is not body.params for g in gates)
    if opcode == "x":
        assert bitwise_not_ir(wires) == gates

    key = RotationCall("rz", angle) if opcode == "rz" else CliffordCall(opcode, 1)
    modeled = query_qec_gate_cost(OnEachCall(key, len(wires)))
    assert {name: value for name, value in asdict(modeled).items() if value} == (
        get_cost_value(source, QECGatesCost()).asdict()
    )
    assert get_cost_value(source, QubitCount()) == len(wires)

    state = torch.tensor(
        [complex(i + 1, i % 3 - 1) for i in range(16)], dtype=torch.complex128
    )
    state /= state.norm()
    matrix = torch.as_tensor(source.tensor_contract(), dtype=torch.complex128)
    order = (*wires, 1)
    inverse = tuple(order.index(wire) for wire in range(4))
    expected_state = (
        (matrix @ state.reshape(2, 2, 2, 2).permute(order).reshape(8, 2))
        .reshape(2, 2, 2, 2)
        .permute(inverse)
        .reshape(-1)
    )
    circuit = fq.Circuit(4, inputs=state, dtype=torch.complex128)
    circuit.append(FieldOperation(4, gates))
    torch.testing.assert_close(
        circuit.state().flatten(), expected_state, rtol=0, atol=2e-12
    )


@pytest.mark.ftqc_reference
def test_zero_and_symbolic_on_each_resource_counts() -> None:
    pytest.importorskip("qualtran")
    from qualtran._infra.bloq import DecomposeTypeError
    from qualtran.bloqs.basic_gates.on_each import OnEach
    from qualtran.bloqs.basic_gates.x_basis import XGate
    from qualtran.resource_counting._bloq_counts import QECGatesCost
    from qualtran.resource_counting._costing import get_cost_value
    from qualtran.resource_counting._qubit_counts import QubitCount
    from sympy import Symbol

    count = Symbol("width", positive=True, integer=True)
    key = CliffordCall("x", 1)
    for width in (0, count):
        source = OnEach(width, XGate())
        call = OnEachCall(key, width)
        assert count_selected_calls(call, (key,)) == {key: width}
        assert (
            query_qec_gate_cost(call).clifford
            == get_cost_value(source, QECGatesCost()).clifford
        )
        assert get_cost_value(source, QubitCount()) == width
    assert on_each_ir(Instruction("x", (0,)), ()) == ()
    zero = OnEach(0, XGate())
    assert [
        type(item.bloq).__name__ for item, _, _ in zero.decompose_bloq().iter_bloqsoqs()
    ] == ["Split", "Join"]
    with pytest.raises(DecomposeTypeError, match="symbolic bitsize"):
        OnEach(count, XGate()).decompose_bloq()


def test_on_each_rejects_dynamic_or_overlapping_binding() -> None:
    with pytest.raises(ValueError, match="nonempty"):
        bitwise_not_ir(())
    with pytest.raises(ValueError, match="distinct"):
        on_each_ir(Instruction("x", (0,)), (1, 1))
    with pytest.raises(ValueError, match="single-qubit"):
        on_each_ir(Instruction("cx", (0, 1)), (2,))
    with pytest.raises(ValueError, match="unconditioned"):
        on_each_ir(Instruction("reset", (0,), metadata={"is_dynamic": True}), (1,))
    with pytest.raises(ValueError, match="unconditioned"):
        on_each_ir(Instruction("x", (0,), metadata={"conditions": ((0, 1),)}), (1,))
    with pytest.raises(ValueError, match="nonnegative"):
        OnEachCall(CliffordCall("x", 1), -1)


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("kind", ("opaque", "uint", "signed", "ones", "fixed"))
@pytest.mark.parametrize("opcode", ("s", "t"))
def test_concrete_dtypes_and_actual_phase_adjoint(
    kind: str, opcode: Literal["s", "t"]
) -> None:
    from qualtran._infra.data_types import QAny, QFxp, QInt, QIntOnesComp, QUInt
    from qualtran.bloqs.basic_gates.on_each import OnEach
    from qualtran.bloqs.basic_gates.s_gate import SGate
    from qualtran.bloqs.basic_gates.t_gate import TGate
    from qualtran.resource_counting._bloq_counts import QECGatesCost
    from qualtran.resource_counting._costing import get_cost_value
    from qualtran.resource_counting._qubit_counts import QubitCount

    from flagquantum.compiler._component_trace.resource_generalization import PhaseCall
    from flagquantum.core.ir.program.register_types import (
        fixed_point_register_type,
        ones_complement_register_type,
        opaque_register_type,
        signed_register_type,
    )
    from flagquantum.ecosystem.cirq._register_emitter import import_register_emitter

    native_types = {
        "opaque": opaque_register_type(3),
        "uint": unsigned_register_type(3),
        "signed": signed_register_type(3),
        "ones": ones_complement_register_type(3),
        "fixed": fixed_point_register_type(3, 2),
    }
    foreign_types = {
        "opaque": QAny(3),
        "uint": QUInt(3),
        "signed": QInt(3),
        "ones": QIntOnesComp(3),
        "fixed": QFxp(3, 2),
    }
    leaf = SGate() if opcode == "s" else TGate()
    for inverse in (False, True):
        code = opcode + "dg" if inverse else opcode
        source = OnEach(3, leaf.adjoint() if inverse else leaf, foreign_types[kind])
        emitter = import_register_emitter(source)
        assert emitter.requirements() == (0, 0)
        instructions = emitter.snapshots[0].instructions
        program = on_each_word_program(
            3, bit_child(code), target_dtype=native_types[kind]
        )
        assert verify_program_record(program) == ()
        bound = bind_register_program(program, ((0, 1, 2),))
        assert bound.circuit.instructions == instructions
        assert bound.outputs == ((0, 1, 2),) and not bound.events
        circuit = fq.Circuit(3, dtype=torch.complex128)
        circuit.append(FieldOperation(3, instructions))
        torch.testing.assert_close(
            get_unitary(circuit),
            torch.as_tensor(source.tensor_contract()),
            rtol=0,
            atol=2e-12,
        )
        assert instructions == tuple(Instruction(code, (wire,)) for wire in range(3))
        assert Counter(gate.name for gate in instructions) == {code: 3}
        actual_inverse = emitter.snapshots[1]
        expected_inverse = tuple(
            Instruction(opcode if inverse else opcode + "dg", (wire,))
            for wire in reversed(range(3))
        )
        assert actual_inverse.instructions == expected_inverse
        source_inverse = torch.as_tensor(source.adjoint().tensor_contract())
        inverse_circuit = fq.Circuit(3, dtype=torch.complex128)
        inverse_circuit.append(FieldOperation(3, actual_inverse.instructions))
        torch.testing.assert_close(
            get_unitary(inverse_circuit), source_inverse, rtol=0, atol=2e-12
        )
        circuit.append(FieldOperation(3, actual_inverse.instructions))
        torch.testing.assert_close(
            get_unitary(circuit),
            torch.eye(8, dtype=torch.complex128),
            rtol=0,
            atol=2e-12,
        )
        call = OnEachCall(PhaseCall(opcode, adjoint=inverse), 3)
        modeled = query_qec_gate_cost(call)
        declared = get_cost_value(source, QECGatesCost())
        assert modeled.t == declared.t == (3 if opcode == "t" else 0)
        assert modeled.clifford == declared.clifford == (3 if opcode == "s" else 0)
        assert get_cost_value(source, QubitCount()) == 3


@pytest.mark.ftqc_reference
def test_constructor_admission_is_distinct_from_executable_body() -> None:
    from qualtran._infra.bloq import Bloq
    from qualtran._infra.composite_bloq import BloqError
    from qualtran._infra.data_types import QBit
    from qualtran._infra.registers import Register
    from qualtran._infra.registers import Signature as SourceSignature
    from qualtran.bloqs.basic_gates.on_each import OnEach
    from qualtran.bloqs.basic_gates.s_gate import SGate
    from qualtran.bloqs.basic_gates.z_basis import ZeroEffect, ZeroState

    class ShapedChild:
        @property
        def signature(self) -> Any:
            return SourceSignature((Register("q", QBit(), shape=(1,)),))

    shaped_type = type("ShapedChild", (Bloq,), {"signature": ShapedChild.signature})

    # All four pass the unchanged constructor, then fail in its actual body.
    for gate, dtype, error in (
        (ZeroState(), None, BloqError),
        (ZeroEffect(), None, AttributeError),
        (shaped_type(), None, IndexError),
        (SGate(), QBit(), IndexError),
    ):
        source = OnEach(3, gate, dtype)
        with pytest.raises(error):
            source.decompose_bloq()

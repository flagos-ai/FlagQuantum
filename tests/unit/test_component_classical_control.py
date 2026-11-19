"""Classical controls propagate through real stored calls and existing guards."""

from collections import Counter
from dataclasses import replace
from itertools import product
from math import pi
from typing import Any

import pytest
import torch

try:
    import numpy as np
except ModuleNotFoundError:
    pytest.importorskip("numpy")

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.compiler.registers.component_builder import ComponentProgramBuilder
from flagquantum.compiler.registers.component_classical_control import (
    classical_control_component_program,
)
from flagquantum.compiler.registers.component_feedback import (
    call_classical_feedback,
    classical_bit_feedback_function,
)
from flagquantum.compiler.registers.component_gate_function import bit_gate_function
from flagquantum.compiler.registers.component_measurement import measure_x_word_function
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.core.ir.program import ProgramRecord, verify_program_record
from flagquantum.core.ir.program.register_types import (
    classical_bit_type,
    quantum_bit_type,
)
from flagquantum.core.ir.program.registers import Register, RegisterSide
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.dynamic import DynamicCircuit
from flagquantum.runtime.dynamic.execution import run_dynamic
from tests.unit.measurement_checks import forced_branch
from tests.unit.register_source_graph import (
    SourcePorts,
    SourceResult,
    assert_source_wire_streams,
    walk_source_graph,
)

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def hierarchy() -> tuple[ProgramRecord, Any]:
    from qualtran import BloqBuilder, QBit
    from qualtran import Register as SourceRegister
    from qualtran import Signature as SourceSignature
    from qualtran.bloqs.basic_gates import CNOT, Hadamard, ZPowGate

    bit = quantum_bit_type()
    circuit = CircuitIR(
        3,
        (
            Instruction("cx", (0, 2)),
            Instruction("phase", (2,), {"theta": pi / 4}),
            Instruction("cx", (0, 2)),
            Instruction("h", (1,)),
        ),
    )
    leaf = bit_gate_function("leaf", (Register("q", bit, (3,)),), circuit)
    builder = ComponentProgramBuilder(
        Signature((Register("q", bit, (2,)),)), (leaf,), name="body"
    )
    q = np.asarray(builder.inputs["q"])
    work = builder.allocate(1)
    w = builder.split(work)[0]
    out = np.asarray(
        builder.call_dict("leaf", {"q": np.asarray([q[0], q[1], w], dtype=object)})["q"]
    )
    builder.free(builder.join([out[2]]))
    body = builder.finalize({"q": out[:2]})
    builder = ComponentProgramBuilder(
        Signature((Register("q", bit, (2,)),)), body.functions
    )
    values = builder.call_dict("body", builder.inputs)
    values = builder.call_dict("body", values)
    module = builder.finalize(values)

    bb, ins = BloqBuilder.from_signature(
        SourceSignature((SourceRegister("q", QBit(), (2,)),))
    )
    sq = ins["q"]
    for _ in range(2):
        w = bb.allocate()
        sq[0], w = bb.add(CNOT(), ctrl=sq[0], target=w)
        w = bb.add(ZPowGate(exponent=0.25), q=w)
        sq[0], w = bb.add(CNOT(), ctrl=sq[0], target=w)
        sq[1] = bb.add(Hadamard(), q=sq[1])
        bb.free(w)
    return module, bb.finalize(q=sq)


def source_stream(source: Any) -> tuple[Instruction, ...]:
    from qualtran.bloqs.basic_gates import CNOT, Hadamard, ZPowGate
    from qualtran.bloqs.bookkeeping import Allocate, Free

    def leaf(
        bloq: Any, inputs: SourcePorts, outputs: SourcePorts
    ) -> SourceResult | None:
        if isinstance(bloq, Allocate):
            return (), {"reg": (2,)}
        if isinstance(bloq, Free):
            assert inputs == {"reg": (2,)}
            return (), {}
        if isinstance(bloq, CNOT):
            return (Instruction("cx", (*inputs["ctrl"], *inputs["target"])),), inputs
        if isinstance(bloq, Hadamard):
            return (Instruction("h", inputs["q"]),), inputs
        if isinstance(bloq, ZPowGate):
            return (
                Instruction("phase", inputs["q"], {"theta": pi * float(bloq.exponent)}),
            ), inputs
        return None

    gates, outputs = walk_source_graph(source, {"q": (0, 1)}, {"q": (0, 1)}, leaf)
    assert outputs == {"q": (0, 1)}
    return gates


@pytest.mark.parametrize("pattern", [(True,), (False,), (True, False)])
def test_hierarchy_source_child_stream_complex_tensor_append_and_clean_work(
    pattern: tuple[bool, ...],
) -> None:
    from qualtran import CBit, CtrlSpec
    from qualtran.bloqs.mcmt.classically_controlled import ClassicallyControlled
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    module, source_child = hierarchy()
    before = repr(module)
    shape = () if len(pattern) == 1 else (len(pattern),)
    controlled = classical_control_component_program(module, ((shape, pattern),))
    assert repr(module) == before and verify_program_record(controlled) == ()
    assert controlled.canonical_payload is None
    root = controlled.functions[0]
    assert root.registers is not None
    assert root.registers[0] == Register("ctrl", classical_bit_type(), shape)
    assert [op.opcode for op in root.blocks[0].operations] == ["call", "call"]
    assert root.blocks[0].operations[0].callee == root.blocks[0].operations[1].callee
    for function in module.functions[1:]:
        assert any(f is function for f in controlled.functions)
    source = ClassicallyControlled(
        source_child, CtrlSpec(CBit(), np.asarray(pattern, dtype=int).reshape(shape))
    )
    direct = source._tensor_data().reshape(2 ** len(pattern), 4, 2 ** len(pattern), 4)
    original_gates = source_stream(source_child)
    original = bind_register_program(module, ((0,), (1,)), clean=(2,))
    assert_source_wire_streams(original.circuit.instructions, original_gates)
    model = get_cost_value(source_child, QECGatesCost())
    assert (model.total_t_count(), model.clifford, model.measurement) == (2, 6, 0)
    # Upstream declares classical action and tensor, without a physical decomposition.
    with pytest.raises(NotImplementedError):
        source.decompose_bloq()
    state = torch.complex(torch.arange(1, 9).double(), torch.arange(8).double())
    state /= state.norm()
    initial = torch.zeros(16, dtype=torch.complex128)
    initial.reshape(4, 2, 2)[:, 0, :] = state.reshape(4, 2)
    for values in product((0, 1), repeat=len(pattern)):
        bound = bind_register_program(
            controlled,
            (*(() for _ in values), (0,), (1,)),
            classical_inputs=(*values, None, None),
            clean=(2,),
        )
        count = Counter(g.name for g in bound.circuit.instructions)
        assert count == {"classical_xor": len(pattern), "cx": 4, "phase": 2, "h": 2}
        assert len(bound.events) == 4
        assert all(event.kind in {"allocate", "free"} for event in bound.events)
        enabled = values == tuple(map(int, pattern))
        computational = tuple(
            g for g in bound.circuit.instructions if g.name != "classical_xor"
        )
        for actual, child_gate in zip(
            computational, original.circuit.instructions, strict=True
        ):
            assert (
                actual.name == child_gate.name
                and actual.wires == child_gate.wires
                and actual.params == child_gate.params
            )
            assert actual.metadata["conditions"] == tuple(enumerate(map(int, pattern)))
        circuit = DynamicCircuit(4, inputs=initial, dtype=torch.complex128)
        circuit.append(FieldOperation(4, bound.circuit.instructions))
        result = run_dynamic(circuit, shots=1, strategy="trajectory")
        branch = result.final_states[0].reshape(4, 2, 2)
        index = sum(value << (len(values) - i - 1) for i, value in enumerate(values))
        matrix = torch.as_tensor(
            direct[index, :, index, :].copy(), dtype=torch.complex128
        )
        expected = (matrix @ state.reshape(4, 2)).flatten()
        torch.testing.assert_close(
            branch[:, 0, :].flatten(), expected, rtol=0, atol=4e-14
        )
        assert branch[:, 1, :].abs().max().item() < 4e-14
        assert result.classical_bits[0].tolist() == list(values)
        if not enabled:
            torch.testing.assert_close(
                branch[:, 0, :].flatten(), state, rtol=0, atol=4e-14
            )


@pytest.mark.parametrize("outer", [False, True])
def test_existing_guard_conjunction_every_measurement_history_and_callback(
    outer: bool,
) -> None:
    from qualtran import CBit, CtrlSpec
    from qualtran.bloqs.basic_gates import XGate
    from qualtran.bloqs.mcmt.classically_controlled import ClassicallyControlled

    bit = quantum_bit_type()
    child = classical_bit_feedback_function(
        "leaf",
        (Register("q", bit),),
        CircuitIR(1, (Instruction("x", (0,)),)),
        (((), (True,)),),
    )
    builder = ComponentProgramBuilder(Signature(child.registers or ()), (child,))
    forward = builder.finalize(builder.call_dict("leaf", builder.inputs))
    controlled = classical_control_component_program(forward, (((), (outer,)),))
    root = controlled.functions[0]
    assert root.registers is not None
    assert tuple(r.name for r in root.registers) == ("ctrl2", "ctrl", "q")
    controls = classical_bit_type()
    caller = ComponentProgramBuilder(
        Signature(
            (
                Register("a", bit, side=RegisterSide.LEFT),
                Register("b", bit, side=RegisterSide.LEFT),
                Register("out_a", controls, side=RegisterSide.RIGHT),
                Register("out_b", controls, side=RegisterSide.RIGHT),
                Register("q", bit),
            )
        ),
        (*controlled.functions, measure_x_word_function("measure")),
        name="caller",
    )
    a = caller.call_dict("measure", {"q": caller.inputs["a"]})["c"]
    b = caller.call_dict("measure", {"q": caller.inputs["b"]})["c"]
    (a,), (b, q) = call_classical_feedback(
        caller, root, (a,), {"ctrl": b, "q": caller.inputs["q"]}
    )
    record = caller.finalize({"out_a": a, "out_b": b, "q": q})
    bound = bind_register_program(record, ((0,), (1,), (2,)), entry="caller")
    assert Counter(g.name for g in bound.circuit.instructions) == {
        "h": 2,
        "measure": 2,
        "x": 1,
    }
    assert len(bound.events) == 0
    source = ClassicallyControlled(
        ClassicallyControlled(XGate(), CtrlSpec(CBit(), 1)),
        CtrlSpec(CBit(), int(outer)),
    )
    amplitude = torch.complex(torch.arange(1, 17).double(), torch.arange(16).double())
    amplitude /= amplitude.norm()
    reference = (
        Instruction("h", (0,)),
        Instruction("measure", (0,), metadata={"classical_bit": 0, "is_dynamic": True}),
        Instruction("h", (1,)),
        Instruction("measure", (1,), metadata={"classical_bit": 1, "is_dynamic": True}),
    )
    for settings in product((0, 1), repeat=2):
        classical = source.call_classically(ctrl2=settings[0], ctrl=settings[1], q=0)
        expected_gates = (
            *reference,
            *((Instruction("x", (2,)),) if classical[-1] else ()),
        )
        torch.testing.assert_close(
            forced_branch(amplitude, bound.circuit.instructions, settings),
            forced_branch(amplitude, expected_gates, settings),
            rtol=0,
            atol=4e-14,
        )


def test_invalid_inputs_and_leaf_replacement_without_consumer_changes() -> None:
    bit = quantum_bit_type()
    leaf = bit_gate_function(
        "main", (Register("q", bit),), CircuitIR(1, (Instruction("h", (0,)),))
    )
    malformed: Any
    for malformed in ((((2,), (True,)),), (((), (1,)),)):
        with pytest.raises(ValueError):
            classical_control_component_program(ProgramRecord((leaf,)), malformed)
    measured = measure_x_word_function("main")
    with pytest.raises(ValueError, match="THRU"):
        classical_control_component_program(
            ProgramRecord((measured,)), (((), (True,)),)
        )
    with pytest.raises(ValueError, match="finite typed"):
        classical_control_component_program(ProgramRecord((leaf,)), (), entry="missing")
    for controls in ((), (((), (True,)),), (((2,), (True, False)),)):
        actual = classical_control_component_program(
            ProgramRecord((leaf,)), controls
        ).functions[0]
        expected = classical_bit_feedback_function(
            "main",
            (Register("q", bit),),
            CircuitIR(1, (Instruction("h", (0,)),)),
            controls,
        )
        assert actual == expected
    invalid = replace(leaf, blocks=())
    with pytest.raises(ValueError):
        classical_control_component_program(ProgramRecord((invalid,)), (((), (True,)),))


@pytest.mark.parametrize("outer", [False, True])
def test_disjunctive_child_guard_matches_flat_replacement_for_all_classical_inputs(
    outer: bool,
) -> None:
    from flagquantum.compiler.registers.control.control_predicate import ControlRegister
    from flagquantum.compiler.registers.primitives.classical_controlled import (
        classically_controlled_ir,
    )
    from flagquantum.core.classical_dataflow import ClassicalXor

    bit = quantum_bit_type()
    base = classical_bit_feedback_function(
        "main",
        (Register("q", bit),),
        CircuitIR(1, (Instruction("x", (0,)),)),
        (((2,), (True, False)),),
    )
    block = base.blocks[0]
    reads = [op for op in block.operations if op.opcode == "read_classical_register"]
    from flagquantum.core.ir.program import ConditionRef

    dnf = (
        (ConditionRef(reads[0].results[-1], True),),
        (ConditionRef(reads[1].results[-1], False),),
    )
    leaf = replace(
        base,
        blocks=(
            replace(
                block,
                operations=tuple(
                    replace(op, condition_clauses=dnf) if op.opcode == "x" else op
                    for op in block.operations
                ),
            ),
        ),
    )
    module = classical_control_component_program(
        ProgramRecord((leaf,)), (((), (outer,)),)
    )
    matrix_state = torch.tensor([1 + 2j, 3 - 4j, -2j, 1 - 1j], dtype=torch.complex128)
    matrix_state /= matrix_state.norm()
    child = Instruction(
        "x", (0,), metadata={"condition_clauses": (((1, 1),), ((2, 0),))}
    )
    expected_child = classically_controlled_ir(
        ((0,),), (ControlRegister(1, (int(outer),)),), (child,)
    )
    for values in product((0, 1), repeat=3):
        bound = bind_register_program(
            module, ((), (), (), (0,)), classical_inputs=(*values, None)
        )
        expected = (
            *(
                ClassicalXor((), i, bool(value)).instruction()
                for i, value in enumerate(values)
            ),
            *expected_child,
        )
        assert_source_wire_streams(bound.circuit.instructions, expected)
        # Both implementations feed the same user-facing append consumer.
        for instructions in (bound.circuit.instructions, expected):
            circuit = DynamicCircuit(2, inputs=matrix_state, dtype=torch.complex128)
            circuit.append(FieldOperation(2, instructions))
            result = run_dynamic(circuit, shots=1, strategy="trajectory")
            enabled = values[0] == int(outer) and (values[1] == 1 or values[2] == 0)
            target = (
                matrix_state.reshape(2, 2).flip(0).flatten()
                if enabled
                else matrix_state
            )
            torch.testing.assert_close(
                result.final_states[0].flatten(), target, rtol=0, atol=3e-14
            )


@pytest.mark.parametrize("effect", ["measure", "reset", "channel"])
def test_reachable_effects_are_rejected_before_transformation(effect: str) -> None:
    circuit = (
        CircuitIR(
            1,
            (
                Instruction(
                    "measure", (0,), metadata={"is_dynamic": True, "classical_bit": 0}
                ),
            ),
        )
        if effect == "measure"
        else CircuitIR(1, (Instruction("reset", (0,), metadata={"is_dynamic": True}),))
    )
    leaf = bit_gate_function("effect", (Register("q", quantum_bit_type()),), circuit)
    if effect == "channel":
        leaf = bit_gate_function(
            "effect",
            (Register("q", quantum_bit_type()),),
            CircuitIR(
                1,
                (
                    Instruction(
                        "kraus",
                        (0,),
                        matrix=(torch.eye(2, dtype=torch.complex128),),
                        metadata={"is_channel": True},
                    ),
                ),
            ),
        )
    builder = ComponentProgramBuilder(
        Signature((Register("q", quantum_bit_type()),)), (leaf,)
    )
    record = builder.finalize(builder.call_dict("effect", builder.inputs))
    before = repr(record)
    assert verify_program_record(record) == ()
    with pytest.raises(ValueError, match="coherent body"):
        classical_control_component_program(record, (((), (True,)),))
    assert repr(record) == before


def test_existing_leaf_channel_admission_is_preserved() -> None:
    leaf = classical_bit_feedback_function(
        "main",
        (Register("q", quantum_bit_type()),),
        CircuitIR(
            1,
            (
                Instruction(
                    "kraus",
                    (0,),
                    matrix=(torch.eye(2, dtype=torch.complex128),),
                    metadata={"is_channel": True},
                ),
            ),
        ),
        (((), (True,)),),
    )
    record = ProgramRecord((leaf,))
    assert verify_program_record(record) == ()
    bound = bind_register_program(record, ((), (0,)), classical_inputs=(1, None))
    assert [g.name for g in bound.circuit.instructions] == ["classical_xor", "kraus"]
    assert bound.circuit.instructions[-1].metadata["conditions"] == ((0, 1),)

"""Classical views preserve slot identity through typed calls and normal append."""

from dataclasses import replace
from math import prod

import pytest
from sympy import Symbol

from flagquantum.compiler.registers.component_adjoint import adjoint_component_calls
from flagquantum.compiler.registers.component_builder import ComponentProgramBuilder
from flagquantum.compiler.registers.component_classical_control import (
    classical_control_component_program,
)
from flagquantum.compiler.registers.layout.component_cast import cast_word_function
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.compiler.registers.register_programs import lower_register_program
from flagquantum.core.ir.program import (
    BlockRecord,
    ConditionRef,
    FunctionRecord,
    Operation,
    ProgramRecord,
    ValueRef,
    verify_program_record,
)
from flagquantum.core.ir.program.register_types import (
    classical_bit_type,
    classical_signed_register_type,
    classical_unsigned_register_type,
    quantum_bit_type,
)
from flagquantum.core.ir.program.register_verifier import (
    executable_classical_register_width,
)
from flagquantum.core.ir.program.registers import Register, RegisterSide, RegisterType
from flagquantum.core.ir.program.signatures import Signature
from tests.unit.test_classical_word_program import TYPES
from tests.unit.test_classical_word_write_contract import FIELD_TYPES
from tests.unit.test_classical_word_writes import execute

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("dtype", (*TYPES, *FIELD_TYPES))
@pytest.mark.parametrize("shape", ((), (2, 2), (0, 2)))
def test_all_classical_families_reverse_shaped_and_unused_raw_encodings(
    dtype: RegisterType, shape: tuple[int, ...]
) -> None:
    width = executable_classical_register_width(dtype)
    source = classical_unsigned_register_type(width)
    forward = cast_word_function("view", source, dtype, shape)
    backward = cast_word_function("inverse", dtype, source, shape)
    assert verify_program_record(ProgramRecord((forward, backward))) == ()
    count = prod(shape)
    for raw in range(1 << width):
        # Raw encodings invalid for a bounded/field numerical value remain stored.
        bound = bind_register_program(
            ProgramRecord((replace(forward, name="main"),)),
            ((),) * count,
            classical_inputs=(raw,) * count,
            borrowed=(0, 1),
        )
        assert bound.classical_outputs == tuple(
            tuple(range(i * width, (i + 1) * width)) for i in range(count)
        )
        assert bound.events == ()
        assert (
            len(bound.circuit.instructions) == count * width
        )  # Entry initialization only.
        _, outputs = execute(bound)
        assert outputs == tuple(
            tuple((raw >> bit) & 1 for bit in reversed(range(width)))
            for _ in range(count)
        )
    builder = ComponentProgramBuilder(
        Signature((Register("c", source, shape),)), (forward, backward)
    )
    viewed = builder.call_dict("view", {"reg": builder.inputs["c"]})
    restored = builder.call_dict("inverse", viewed)
    program = builder.finalize({"c": restored["reg"]})
    inverse = adjoint_component_calls(program, ())
    assert verify_program_record(inverse) == ()
    assert (
        bind_register_program(
            inverse, ((),) * count, classical_inputs=(0,) * count, borrowed=(0, 1)
        ).events
        == ()
    )
    if count:
        with pytest.raises(ValueError, match="physical register binding"):
            lower_register_program(program)
    else:
        assert lower_register_program(program).functions[0].blocks[0].operations == ()


def test_symbolic_equal_width_declarations_and_concrete_binding_refusal() -> None:
    width = Symbol("n", integer=True, positive=True)
    function = cast_word_function(
        "main",
        classical_unsigned_register_type(width),
        classical_signed_register_type(width),
    )
    assert verify_program_record(ProgramRecord((function,))) == ()
    with pytest.raises(ValueError, match="concrete"):
        bind_register_program(
            ProgramRecord((function,)), ((),), classical_inputs=(0,), borrowed=(0, 1)
        )
    with pytest.raises(ValueError, match="preserve raw bit width"):
        cast_word_function(
            "main",
            classical_unsigned_register_type(width),
            classical_signed_register_type(width + 1),
        )


def test_cast_rejects_malformed_metadata_domains_and_stale_ownership() -> None:
    function = cast_word_function(
        "main", classical_unsigned_register_type(2), classical_signed_register_type(2)
    )
    block = function.blocks[0]
    operation = block.operations[0]
    for bad in (
        replace(operation, operands=()),
        replace(operation, results=()),
        replace(operation, results=(ValueRef("entry", 1, "bool"),)),
        replace(operation, register_type=None),
        replace(operation, register_type=quantum_bit_type()),
        replace(operation, register_type=classical_signed_register_type(3)),
        replace(operation, register_type=RegisterType("unknown", 0, 2)),
        replace(operation, literal=1.0),
        replace(
            operation,
            condition_clauses=((ConditionRef(ValueRef("entry", 10, "bool"), True),),),
        ),
    ):
        assert verify_program_record(
            ProgramRecord(
                (replace(function, blocks=(replace(block, operations=(bad,)),)),)
            )
        )
    stale = replace(function, blocks=(replace(block, return_values=block.arguments),))
    assert "value.linear_duplicate" in {
        d.code for d in verify_program_record(ProgramRecord((stale,)))
    }
    for source, target in (
        (classical_bit_type(), quantum_bit_type()),
        (quantum_bit_type(), classical_bit_type()),
    ):
        with pytest.raises(ValueError):
            cast_word_function("main", source, target)


def test_old_snapshots_survive_cast_and_conditional_write() -> None:
    source, target = classical_unsigned_register_type(
        2
    ), classical_signed_register_type(2)
    before, read, view, written, restored = (
        ValueRef("entry", i, "register") for i in range(5)
    )
    old = tuple(ValueRef("entry", 5 + i, "bool") for i in range(2))
    new = tuple(ValueRef("entry", 7 + i, "bool") for i in range(2))
    one = ValueRef("entry", 9, "bool")
    operations = (
        Operation("read_classical_register", (before,), (read, *old)),
        Operation("cast_classical_register", (read,), (view,), register_type=target),
        Operation("bool_one", results=(one,)),
        *(
            Operation("bool_xor", (bit, one), (out,))
            for bit, out in zip(old, new, strict=True)
        ),
        Operation(
            "write_classical_register",
            (view, *new),
            (written,),
            condition_clauses=((ConditionRef(one, True),),),
        ),
        Operation(
            "cast_classical_register", (written,), (restored,), register_type=source
        ),
    )
    function = FunctionRecord(
        "main",
        (
            BlockRecord(
                "entry", (before,), operations=operations, return_values=(restored,)
            ),
        ),
        registers=(Register("c", source),),
    )
    for activation in (False, True):
        controlled = classical_control_component_program(
            ProgramRecord((function,)), (((), (activation,)),)
        )
        for enabled in (0, 1):
            for raw in range(4):
                bound = bind_register_program(
                    controlled,
                    ((), ()),
                    classical_inputs=(enabled, raw),
                    borrowed=(0, 1),
                )
                _, outputs = execute(bound)
                expected = raw ^ (3 if enabled == activation else 0)
                assert outputs == ((enabled,), (expected >> 1, expected & 1))
                assert bound.events == ()
                # Three immutable original slots precede every computed/write slot.
                assert bound.classical_outputs[1] != (1, 2)


def test_measurement_cast_read_feedback_all_complex_branches_and_clean_work() -> None:
    import torch

    from examples.algorithms.binary_field_arithmetic import FieldOperation
    from flagquantum import Circuit
    from flagquantum.compiler.registers.component_gate_function import bit_gate_function
    from flagquantum.core.ir import CircuitIR, Instruction
    from tests.unit.measurement_checks import forced_branch
    from tests.unit.test_program_owned_measurement import measured_function

    bit, word = classical_bit_type(), classical_signed_register_type(1)
    forward = cast_word_function("cast", bit, word)
    backward = cast_word_function("uncast", word, bit)
    c, read, q, out, snapshot = (
        ValueRef("entry", index, kind)
        for index, kind in enumerate(
            ("register", "register", "register", "register", "bool")
        )
    )
    wire, successor = ValueRef("entry", 5, "qubit"), ValueRef("entry", 6, "qubit")
    feedback = FunctionRecord(
        "feedback",
        (
            BlockRecord(
                "entry",
                (c, q),
                operations=(
                    Operation("read_classical_register", (c,), (read, snapshot)),
                    Operation("split_register", (q,), (wire,)),
                    Operation(
                        "z",
                        (wire,),
                        (successor,),
                        condition_clauses=((ConditionRef(snapshot, True),),),
                    ),
                    Operation(
                        "join_register",
                        (successor,),
                        (out,),
                        register_type=quantum_bit_type(),
                    ),
                ),
                return_values=(read, out),
            ),
        ),
        registers=(Register("c", word), Register("q", quantum_bit_type())),
    )
    # Express the quantum gate on a bit register through its ordinary typed leaf.
    gate = bit_gate_function(
        "phase",
        (Register("q", quantum_bit_type()),),
        CircuitIR(1, (Instruction("t", (0,)),)),
    )
    builder = ComponentProgramBuilder(
        Signature(
            (
                Register("q", quantum_bit_type(), side=RegisterSide.LEFT),
                Register("spectator", quantum_bit_type()),
                Register("c", bit, side=RegisterSide.RIGHT),
            )
        ),
        (measured_function("measurement"), forward, backward, feedback, gate),
    )
    measured = builder.call_dict("measurement", {"q": builder.inputs["q"]})["c"]
    viewed = builder.call_dict("cast", {"reg": measured})["reg"]
    phased = builder.call_dict("phase", {"q": builder.inputs["spectator"]})["q"]
    result = builder.call_dict("feedback", {"c": viewed, "q": phased})
    restored = builder.call_dict("uncast", {"reg": result["c"]})["reg"]
    work = builder.allocate(dtype=quantum_bit_type())
    builder.free(work)
    record = builder.finalize({"c": restored, "spectator": result["q"]})
    bound = bind_register_program(record, ((0,), (1,)), clean=(2,))
    assert bound.classical_outputs == ((), (0,))
    assert [g.name for g in bound.circuit.instructions] == [
        "measure",
        "t",
        "z",
    ]
    from flagquantum.core.classical_control import instruction_condition_clauses

    assert instruction_condition_clauses(bound.circuit.instructions[-1]) == (((0, 1),),)
    assert [(e.kind, e.wires) for e in bound.events] == [
        ("allocate", (2,)),
        ("free", (2,)),
    ]
    initial = torch.zeros(8, dtype=torch.complex128)
    initial[::2] = torch.tensor(
        [1 + 2j, 3 - 4j, -2 + 1j, 4 + 3j], dtype=torch.complex128
    )
    initial /= initial.norm()
    mask = torch.arange(8) >> 2
    for branch in (0, 1):
        projected = initial * (mask == branch)
        expected = (
            Circuit(3, inputs=projected, dtype=torch.complex128)
            .append(
                FieldOperation(
                    3,
                    (
                        Instruction("t", (1,)),
                        *((Instruction("z", (1,)),) if branch else ()),
                    ),
                )
            )
            .state()
            .flatten()
        )
        actual = forced_branch(initial, bound.circuit.instructions, (branch,))
        torch.testing.assert_close(actual, expected, atol=2e-14, rtol=0)
        assert torch.count_nonzero(actual[1::2]) == 0

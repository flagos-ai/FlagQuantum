"""Classical word updates lower through immutable SSA slots and normal append."""

from collections import Counter
from dataclasses import replace
from itertools import product
from typing import Any

import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.compiler.registers.classical_word_storage import classical_word_bits
from flagquantum.compiler.registers.component_builder import ComponentProgramBuilder
from flagquantum.compiler.registers.component_classical_control import (
    classical_control_component_program,
)
from flagquantum.compiler.registers.control.control_predicate import ControlRegister
from flagquantum.compiler.registers.primitives.classical_controlled import (
    classically_controlled_ir,
)
from flagquantum.compiler.registers.register_binding import (
    BoundRegisterProgram,
    bind_register_program,
)
from flagquantum.compiler.registers.register_programs import lower_register_program
from flagquantum.core.classical_dataflow import ClassicalXor
from flagquantum.core.ir import CircuitIR, Instruction
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
    classical_bounded_unsigned_register_type,
    quantum_bit_type,
)
from flagquantum.core.ir.program.register_verifier import (
    executable_classical_register_width,
)
from flagquantum.core.ir.program.registers import Register, RegisterSide, RegisterType
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.dynamic import DynamicCircuit
from flagquantum.runtime.dynamic.execution import run_dynamic
from flagquantum.simulation.finite_field_encoding import FiniteFieldEncoding
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_classical_word_binding import CASES
from tests.unit.test_classical_word_write_contract import FIELD_TYPES, write_function

pytestmark = pytest.mark.unit


def execute(
    bound: BoundRegisterProgram,
) -> tuple[torch.Tensor, tuple[tuple[int, ...], ...]]:
    initial = torch.tensor([1 + 2j, 3 - 1j, -2 + 4j, 5 + 3j], dtype=torch.complex128)
    initial /= initial.norm()
    circuit = DynamicCircuit(2, inputs=initial, dtype=torch.complex128)
    circuit.append(FieldOperation(2, bound.circuit.instructions))
    result = run_dynamic(circuit, shots=1, strategy="trajectory")
    slots = result.classical_bits[0].tolist()
    return result.final_states[0].flatten(), tuple(
        tuple(slots[index] for index in word) for word in bound.classical_outputs
    )


@pytest.mark.parametrize(
    "dtype,domain",
    (
        *CASES,
        (classical_bit_type(), (0, 1)),
        (FIELD_TYPES[0], (0, 1, 2)),
        (FIELD_TYPES[1], (0, 1, 2, 4, 5, 6, 8, 9, 10)),
    ),
)
@pytest.mark.parametrize("activation", (False, True))
def test_all_word_families_conditional_outputs_old_snapshots_and_resources(
    dtype: RegisterType,
    domain: tuple[int, ...],
    activation: bool,
) -> None:
    child = write_function(dtype)
    controlled = classical_control_component_program(
        ProgramRecord((child,)),
        (((), (activation,)),),
    )
    width = executable_classical_register_width(dtype)
    for enabled, value in product((0, 1), domain):
        before = classical_word_bits(dtype, value)
        bound = bind_register_program(
            controlled,
            ((), ()),
            classical_inputs=(enabled, value),
            borrowed=(0, 1),
        )
        state, outputs = execute(bound)
        assert outputs == (
            (enabled,),
            tuple(bit ^ int(enabled == activation) for bit in before),
        )
        assert bound.events == ()
        assert Counter(g.name for g in bound.circuit.instructions) == {
            "classical_xor": 2 + 4 * width,
        }
        initial = torch.tensor(
            [1 + 2j, 3 - 1j, -2 + 4j, 5 + 3j], dtype=torch.complex128
        )
        torch.testing.assert_close(state, initial / initial.norm(), atol=4e-14, rtol=0)
        # The old word and the wrapper control retain their original storage.
        instructions = bound.circuit.instructions
        assert all(
            g.metadata.get("classical_bit", -1) > width
            for g in instructions[width + 1 :]
        )


@pytest.mark.parametrize("conditional", (False, True))
@pytest.mark.parametrize("width", (0, 1, 3))
def test_zero_width_and_unconditional_rebinding(width: int, conditional: bool) -> None:
    from flagquantum.core.ir.program.register_types import (
        classical_unsigned_register_type,
    )

    bound = bind_register_program(
        ProgramRecord(
            (
                write_function(
                    classical_unsigned_register_type(width), conditional=conditional
                ),
            )
        ),
        ((),),
        classical_inputs=(0,),
        borrowed=(0, 1),
    )
    _, outputs = execute(bound)
    assert outputs == ((1,) * width,)
    assert len(bound.circuit.instructions) == 1 + width * (4 if conditional else 2)
    assert bound.events == ()
    with pytest.raises(ValueError, match="physical register binding"):
        lower_register_program(
            ProgramRecord((write_function(classical_unsigned_register_type(width)),))
        )


@pytest.mark.parametrize("shape", ((), (2,), (2, 2), (0, 2)))
def test_repeated_nested_calls_shaped_empty_and_owned_aliases(
    shape: tuple[int, ...],
) -> None:
    leaf = replace(write_function(classical_bit_type()), name="flip")
    signature = Signature((Register("c", classical_bit_type(), shape),))
    builder = ComponentProgramBuilder(signature, (leaf,), name="body")
    carried = (
        builder.call_dict("flip", {"c": builder.inputs["c"]}) if not shape else None
    )
    if shape:
        np = pytest.importorskip("numpy")

        array = np.asarray(builder.inputs["c"], dtype=object)
        output = np.empty(array.shape, dtype=object)
        for index in np.ndindex(array.shape):
            output[index] = builder.call_dict("flip", {"c": array[index]})["c"]
        carried = {"c": output}
    assert carried is not None
    body = builder.finalize(carried)
    outer = ComponentProgramBuilder(signature, body.functions)
    current = outer.inputs
    for _ in range(3):
        current = outer.call_dict("body", current)
    record = outer.finalize(current)
    count = len(tuple(signature[0].all_idxs()))
    bound = bind_register_program(
        record,
        ((),) * count,
        classical_inputs=(0,) * count,
        borrowed=(0, 1),
    )
    _, outputs = execute(bound)
    assert outputs == ((1,),) * count
    assert bound.events == ()


@pytest.mark.parametrize("active", (False, True))
def test_old_and_new_snapshots_control_full_complex_state_after_write(
    active: bool,
) -> None:
    function = write_function(classical_bit_type())
    block = function.blocks[0]
    assert block.return_values is not None
    assert function.registers is not None
    c = block.return_values[0]
    old = block.operations[0].results[1]
    q, qr, read, new = (
        ValueRef("entry", i, kind)
        for i, kind in (
            (20, "register"),
            (21, "register"),
            (22, "register"),
            (23, "bool"),
        )
    )
    q0, q1, q2 = (ValueRef("entry", i, "qubit") for i in range(24, 27))
    write = replace(
        block.operations[-1], condition_clauses=((ConditionRef(old, active),),)
    )
    operations = (
        *block.operations[:-1],
        write,
        Operation("read_classical_register", (c,), (read, new)),
        Operation("split_register", (q,), (q0,)),
        Operation("x", (q0,), (q1,), condition_clauses=((ConditionRef(old, True),),)),
        Operation("h", (q1,), (q2,), condition_clauses=((ConditionRef(new, True),),)),
        Operation("join_register", (q2,), (qr,), register_type=quantum_bit_type()),
    )
    record = ProgramRecord(
        (
            replace(
                function,
                blocks=(
                    replace(
                        block,
                        arguments=(*block.arguments, q),
                        operations=operations,
                        return_values=(read, qr),
                    ),
                ),
                registers=(*function.registers, Register("q", quantum_bit_type())),
            ),
        )
    )
    for value in (0, 1):
        bound = bind_register_program(
            record, ((), (0,)), classical_inputs=(value, None), borrowed=(1,)
        )
        state, outputs = execute(bound)
        updated = value ^ int(bool(value) == active)
        assert outputs == ((updated,), ())
        expected_gates = (
            *((Instruction("x", (0,)),) if value else ()),
            *((Instruction("h", (0,)),) if updated else ()),
        )
        initial = torch.tensor(
            [1 + 2j, 3 - 1j, -2 + 4j, 5 + 3j], dtype=torch.complex128
        )
        expected = forced_branch(initial / initial.norm(), expected_gates, ())
        torch.testing.assert_close(state, expected, atol=4e-14, rtol=0)


@pytest.mark.ftqc_reference
def test_original_classical_flip_both_polarities_and_flat_operands() -> None:
    import attrs
    from qualtran._infra.bloq import Bloq
    from qualtran._infra.controlled import CtrlSpec
    from qualtran._infra.data_types import CBit
    from qualtran._infra.registers import Register as SourceRegister
    from qualtran._infra.registers import Signature as SourceSignature
    from qualtran.bloqs.mcmt.classically_controlled import ClassicallyControlled

    @attrs.frozen
    class Flip(Bloq):
        @property
        def signature(self) -> SourceSignature:
            return SourceSignature((SourceRegister("c", CBit()),))

        def on_classical_vals(self, **vals: Any) -> dict[str, int]:
            return {"c": int(vals["c"]) ^ 1}

    child = (ClassicalXor((1,), 1, True).instruction(),)
    for activation, control, value in product((0, 1), repeat=3):
        source = ClassicallyControlled(Flip(), CtrlSpec(CBit(), activation))
        expected = source.call_classically(ctrl=control, c=value)
        gates = (
            ClassicalXor((), 0, bool(control)).instruction(),
            ClassicalXor((), 1, bool(value)).instruction(),
            *classically_controlled_ir(
                ((0,),), (ControlRegister(1, (activation,)),), child
            ),
        )
        guarded = gates[-1]
        assert guarded.params == {"sources": (1,), "constant": True}
        assert guarded.metadata["conditions"] == ((0, activation),)
        bound = BoundRegisterProgram(
            circuit=CircuitIR(2, gates),
            outputs=((), ()),
            events=(),
            classical_outputs=((0,), (1,)),
        )
        assert tuple(word[0] for word in execute(bound)[1]) == expected
        module = classical_control_component_program(
            ProgramRecord((write_function(classical_bit_type()),)),
            (((), (bool(activation),)),),
        )
        bound = bind_register_program(
            module, ((), ()), classical_inputs=(control, value), borrowed=(0, 1)
        )
        assert tuple(word[0] for word in execute(bound)[1]) == expected
    with pytest.raises(ValueError, match="overwrite wrapper controls"):
        classically_controlled_ir(
            ((0,),),
            (ControlRegister(1, (1,)),),
            (ClassicalXor((0,), 0, True).instruction(),),
        )


def test_internal_unused_bits_preserved_external_admission_and_codec_refusal() -> None:
    for dtype in (FIELD_TYPES[0], classical_bounded_unsigned_register_type(2, 3)):
        # Complementing 00 produces raw 11 in an owned internal word.
        bound = bind_register_program(
            ProgramRecord((write_function(dtype),)),
            ((),),
            classical_inputs=(0,),
            borrowed=(0, 1),
        )
        assert execute(bound)[1] == ((1, 1),)
        with pytest.raises(ValueError):
            bind_register_program(
                ProgramRecord((write_function(dtype),)),
                ((),),
                classical_inputs=(3,),
                borrowed=(0, 1),
            )
    with pytest.raises(ValueError):
        FiniteFieldEncoding(3, 1).from_bits((1, 1))


@pytest.mark.parametrize("contradictory", (False, True))
def test_dnf_alias_snapshots_and_idempotent_overlapping_clauses(
    contradictory: bool,
) -> None:
    function = write_function(classical_bit_type())
    block = function.blocks[0]
    old = block.operations[0].results[1]
    owner = block.operations[0].results[0]
    second, alias = ValueRef("entry", 20, "register"), ValueRef("entry", 21, "bool")
    clauses = (
        ((ConditionRef(old, False), ConditionRef(alias, True)),)
        if contradictory
        else ((ConditionRef(old, True),), (ConditionRef(alias, True),))
    )
    write = replace(
        block.operations[-1],
        operands=(second, *block.operations[-1].operands[1:]),
        condition_clauses=clauses,
    )
    function = replace(
        function,
        blocks=(
            replace(
                block,
                operations=(
                    block.operations[0],
                    Operation("read_classical_register", (owner,), (second, alias)),
                    *block.operations[1:-1],
                    write,
                ),
            ),
        ),
    )
    assert verify_program_record(ProgramRecord((function,))) == ()
    for value in (0, 1):
        bound = bind_register_program(
            ProgramRecord((function,)),
            ((),),
            classical_inputs=(value,),
            borrowed=(0, 1),
        )
        assert execute(bound)[1] == ((value if contradictory else 0,),)
        assert len(bound.circuit.instructions) == (4 if contradictory else 5)


def test_overlapping_distinct_dnf_clauses_commit_same_replacement() -> None:
    function = write_function(classical_bit_type())
    block = function.blocks[0]
    old = block.operations[0].results[1]
    one = block.operations[1].results[0]
    write = replace(
        block.operations[-1],
        condition_clauses=(
            (ConditionRef(one, True),),
            (ConditionRef(old, True),),
        ),
    )
    function = replace(
        function,
        blocks=(
            replace(
                block,
                operations=(
                    *block.operations[:-1],
                    write,
                ),
            ),
        ),
    )
    for value in (0, 1):
        bound = bind_register_program(
            ProgramRecord((function,)),
            ((),),
            classical_inputs=(value,),
            borrowed=(0, 1),
        )
        assert execute(bound)[1] == ((value ^ 1,),)
        assert len(bound.circuit.instructions) == 6


def test_nested_boolean_alias_contradiction_forwards_owned_write() -> None:
    function = write_function(classical_bit_type())
    block = function.blocks[0]
    old = block.operations[0].results[1]
    arg = ValueRef("entry", 0, "bool")
    alias_function = FunctionRecord(
        "alias",
        (
            BlockRecord(
                "entry",
                (arg,),
                return_values=(arg, arg),
            ),
        ),
    )
    a, b = (ValueRef("entry", i, "bool") for i in (20, 21))
    write = replace(
        block.operations[-1],
        condition_clauses=((ConditionRef(a, False), ConditionRef(b, True)),),
    )
    function = replace(
        function,
        blocks=(
            replace(
                block,
                operations=(
                    *block.operations[:-1],
                    Operation("call", (old,), (a, b), callee="alias"),
                    write,
                ),
            ),
        ),
    )
    record = ProgramRecord((function, alias_function))
    assert verify_program_record(record) == ()
    for value in (0, 1):
        bound = bind_register_program(
            record, ((),), classical_inputs=(value,), borrowed=(0, 1)
        )
        assert execute(bound)[1] == ((value,),)
        assert len(bound.circuit.instructions) == 3


def test_every_measured_write_history_complex_amplitude_and_clean_reuse() -> None:
    from flagquantum.compiler.registers.component_feedback import (
        classical_bit_feedback_function,
    )
    from flagquantum.compiler.registers.component_measurement import (
        measure_x_word_function,
    )

    bit = quantum_bit_type()
    flip = replace(write_function(classical_bit_type()), name="flip")
    measure = measure_x_word_function("measure")
    feedback = classical_bit_feedback_function(
        "feedback",
        (Register("q", bit),),
        CircuitIR(1, (Instruction("z", (0,)),)),
        (((), (True,)),),
    )
    # Controlled writes are composed after measurement, with the wrapper's
    # control port separately owned from its writable child data.
    controlled = classical_control_component_program(
        ProgramRecord((flip,)), (((), (True,)),), entry="flip"
    )
    builder = ComponentProgramBuilder(
        Signature(
            (
                Register("q", bit),
                Register("ctrl", classical_bit_type(), side=RegisterSide.RIGHT),
                Register("c", classical_bit_type(), side=RegisterSide.RIGHT),
            )
        ),
        (
            *controlled.functions,
            measure,
            feedback,
        ),
    )
    work = builder.allocate(1)
    outcome = builder.call_dict("measure", {"q": work})["c"]
    work2 = builder.allocate(1)
    other = builder.call_dict("measure", {"q": work2})["c"]
    values = builder.call_dict("flip", {"ctrl": outcome, "c": other})
    returned = builder.call_dict(
        "feedback", {"ctrl": values["c"], "q": builder.inputs["q"]}
    )
    record = builder.finalize(
        {"q": returned["q"], "ctrl": values["ctrl"], "c": returned["ctrl"]}
    )
    bound = bind_register_program(
        record, ((0,),), clean=(2,), borrowed=(1,), measured_workspace="reset"
    )
    initial = torch.zeros(8, dtype=torch.complex128)
    logical = torch.tensor([1 + 2j, 3 - 1j, -2 + 4j, 5 + 3j], dtype=torch.complex128)
    logical /= logical.norm()
    initial.reshape(4, 2)[:, 0] = logical
    for a, b in product((0, 1), repeat=2):
        actual = forced_branch(initial, bound.circuit.instructions, (a, b))
        expected = initial / 2
        if a ^ b:
            expected = forced_branch(expected, (Instruction("z", (0,)),), ())
        torch.testing.assert_close(actual, expected, atol=4e-14, rtol=0)
        assert actual.reshape(4, 2)[:, 1].abs().max().item() == 0
    assert sum(g.name == "measure" for g in bound.circuit.instructions) == 2
    assert sum(g.name == "reset" for g in bound.circuit.instructions) == 2
    assert [event.wires for event in bound.events] == [(2,), (2,), (2,), (2,)]


def test_stored_append_path_is_independent_of_numerical_callback_and_flat_wrapper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected(*args: object, **kwargs: object) -> None:
        raise AssertionError("owned write execution must use the stored binding path")

    monkeypatch.setattr(
        "flagquantum.simulation.classical_controlled_values.controlled_classical_values",
        unexpected,
    )
    monkeypatch.setattr(
        "flagquantum.compiler.registers.primitives.classical_controlled.classically_controlled_ir",
        unexpected,
    )
    module = classical_control_component_program(
        ProgramRecord((write_function(classical_bit_type()),)),
        (((), (True,)),),
    )
    bound = bind_register_program(
        module, ((), ()), classical_inputs=(1, 0), borrowed=(0, 1)
    )
    assert execute(bound)[1] == ((1,), (1,))

"""Owned classical ports retain linear identity without quantum-word coercion."""

from dataclasses import replace

import numpy as np
import pytest

from flagquantum.compiler.registers.component_bindings import RegisterBindingError
from flagquantum.compiler.registers.component_builder import ComponentProgramBuilder
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.compiler.registers.register_programs import lower_register_program
from flagquantum.core.ir.program import (
    BlockRecord,
    FunctionRecord,
    Operation,
    ProgramRecord,
    ValueRef,
    verify_program_record,
)
from flagquantum.core.ir.program.register_compatibility import (
    check_register_types_consistent,
)
from flagquantum.core.ir.program.register_types import classical_bit_type
from flagquantum.core.ir.program.register_verifier import register_port_types
from flagquantum.core.ir.program.registers import Register, RegisterSide, RegisterType
from flagquantum.core.ir.program.signatures import Signature

pytestmark = pytest.mark.unit


def identity(name: str, registers: tuple[Register, ...]) -> FunctionRecord:
    signature = Signature(registers)
    arguments = tuple(
        ValueRef("entry", index, "register")
        for index, _ in enumerate(
            (reg, idx) for reg in signature.lefts() for idx in reg.all_idxs()
        )
    )
    return FunctionRecord(
        name,
        (BlockRecord("entry", arguments, return_values=arguments),),
        registers=tuple(signature),
    )


def codes(*functions: FunctionRecord) -> set[str]:
    return {item.code for item in verify_program_record(ProgramRecord(functions))}


@pytest.mark.parametrize("shape", [(), (2,), (2, 3), (0, 2)])
def test_mixed_owned_ports_preserve_declaration_order_and_linear_array_elements(
    shape: tuple[int, ...],
) -> None:
    registers = (
        Register("classical", classical_bit_type(), shape),
        Register("quantum", RegisterType("uint", 2)),
        Register("empty_word", RegisterType("opaque", 0)),
    )
    leaf = identity("leaf", registers)
    builder = ComponentProgramBuilder(Signature(registers), [leaf])
    inputs = builder.inputs
    original = {value.value for value in np.asarray(inputs["classical"]).flat}
    for _ in range(2):
        inputs = builder.call_dict("leaf", dict(reversed(inputs.items())))
        returned = np.asarray(inputs["classical"])
        assert returned.shape == shape
        assert {value.index for value in returned.flat} == set(registers[0].all_idxs())
        current = {value.value for value in returned.flat}
        assert not original & current
        original |= current
    module = builder.finalize(inputs)
    assert verify_program_record(module) == ()
    ports = register_port_types(module.functions[0], RegisterSide.RIGHT)
    assert ports == (
        (classical_bit_type(),) * len(tuple(registers[0].all_idxs()))
        + (RegisterType("uint", 2), RegisterType("opaque", 0))
    )
    assert len(module.functions[0].blocks[0].operations) == 2


@pytest.mark.parametrize("failure", ["stale", "duplicate", "unused", "bool"])
def test_classical_ports_keep_owned_consumption_and_do_not_alias_feedback(
    failure: str,
) -> None:
    function = identity("main", (Register("c", classical_bit_type(), (2,)),))
    block = function.blocks[0]
    first, second = block.arguments
    if failure == "stale":
        leaf = identity("leaf", (Register("c", classical_bit_type()),))
        after = ValueRef("entry", 2, "register")
        block = replace(
            block,
            operations=(Operation("call", (first,), (after,), callee="leaf"),),
        )
        expected = "value.linear_duplicate"
    elif failure == "duplicate":
        leaf = identity("leaf", ())
        block = replace(block, return_values=(first, first))
        expected = "value.linear_duplicate"
    elif failure == "unused":
        leaf = identity("leaf", ())
        block = replace(block, return_values=(first,))
        expected = "value.linear_unconsumed"
    else:
        leaf = identity("leaf", ())
        feedback = ValueRef("entry", 0, "bool")
        block = replace(
            block, arguments=(feedback, second), return_values=(feedback, second)
        )
        expected = "program.register_structure"
    assert expected in codes(replace(function, blocks=(block,)), leaf)


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("boundary", ["call", "return"])
def test_loose_metadata_never_converts_cbit_to_zero_width_quantum_word(
    reverse, boundary
):
    classical, quantum = classical_bit_type(), RegisterType("opaque", 0)
    assert check_register_types_consistent(classical, quantum)
    actual, expected = (quantum, classical) if reverse else (classical, quantum)
    before, after = (ValueRef("entry", i, "register") for i in range(2))
    if boundary == "call":
        leaf = identity("leaf", (Register("c", expected),))
        block = BlockRecord(
            "entry",
            (before,),
            operations=(Operation("call", (before,), (after,), callee="leaf"),),
            return_values=(after,),
        )
    else:
        leaf = identity("leaf", ())
        block = BlockRecord("entry", (before,), return_values=(before,))
    function = FunctionRecord(
        "main",
        (block,),
        registers=(
            Register("c", actual, side=RegisterSide.LEFT),
            Register("c", expected, side=RegisterSide.RIGHT),
        ),
    )
    assert "program.register_type" in codes(function, leaf)


@pytest.mark.parametrize("opcode", ["split_register", "free_register"])
def test_quantum_structure_operations_reject_classical_handles_diagnostically(
    opcode: str,
) -> None:
    before = ValueRef("entry", 0, "register")
    operation = Operation(
        opcode,
        (before,),
        workspace_dirty=False if opcode == "free_register" else None,
    )
    function = FunctionRecord(
        "main",
        (BlockRecord("entry", (before,), operations=(operation,), return_values=()),),
        registers=(Register("c", classical_bit_type(), side=RegisterSide.LEFT),),
    )
    assert "program.register_type" in codes(function)


@pytest.mark.parametrize("opcode", ["join_register", "allocate_register"])
def test_quantum_creation_does_not_create_classical_handles(opcode: str) -> None:
    after = ValueRef("entry", 0, "register")
    operation = Operation(
        opcode,
        results=(after,),
        register_type=classical_bit_type(),
        workspace_dirty=False if opcode == "allocate_register" else None,
    )
    function = FunctionRecord(
        "main",
        (BlockRecord("entry", operations=(operation,), return_values=(after,)),),
        registers=(Register("c", classical_bit_type(), side=RegisterSide.RIGHT),),
    )
    assert "program.register_type" in codes(function)


@pytest.mark.parametrize(
    "dtype",
    [
        RegisterType("bit", False, 1),
        RegisterType("bit", 0, True),
        RegisterType("bit", 0.0, 1),
        RegisterType("bit", 0, 1.0),
        RegisterType("bit", 0, 2),
        RegisterType("bit", 1, 1),
        RegisterType("bit", 0, 1, (("storage", "classical"),)),
        RegisterType("uint", 0, 1),
        RegisterType("opaque", 0, 1),
    ],
)
def test_classical_port_admission_is_exact_and_does_not_admit_other_descriptors(
    dtype: RegisterType,
) -> None:
    assert "program.register_type" in codes(identity("main", (Register("c", dtype),)))


def test_classical_identity_stays_outside_physical_quantum_lowering() -> None:
    module = ProgramRecord((identity("main", (Register("c", classical_bit_type()),)),))
    assert verify_program_record(module) == ()
    with pytest.raises(ValueError, match="unsupported executable register"):
        lower_register_program(module)
    with pytest.raises(ValueError, match="unsupported executable register"):
        bind_register_program(module, ((),), clean=(0,))


@pytest.mark.ftqc_reference
def test_nested_cbit_identity_calls_match_original_cast_ownership() -> None:
    from qualtran import BloqBuilder, CBit
    from qualtran import Register as SourceRegister
    from qualtran import Signature as SourceSignature
    from qualtran.bloqs.bookkeeping import Cast

    signature = Signature((Register("reg", classical_bit_type()),))
    native = ComponentProgramBuilder(signature, [identity("cast", tuple(signature))])
    source, source_inputs = BloqBuilder.from_signature(
        SourceSignature((SourceRegister("reg", CBit()),))
    )
    native_inputs = native.inputs
    stale = native_inputs["reg"]
    source_stale = source_inputs["reg"]
    for _ in range(2):
        native_inputs = native.call_dict("cast", native_inputs)
        source_inputs = {"reg": source.add(Cast(CBit(), CBit()), **source_inputs)}
    composite = source.finalize(**source_inputs)
    module = native.finalize(native_inputs)
    assert verify_program_record(module) == ()
    assert (
        len(module.functions[0].blocks[0].operations)
        == len(composite.bloq_instances)
        == 2
    )
    assert [composite.call_classically(reg=bit) for bit in (0, 1)] == [(0,), (1,)]
    with pytest.raises(RegisterBindingError, match="available"):
        native.call_dict("cast", {"reg": stale})
    with pytest.raises(ValueError, match="available"):
        source.add(Cast(CBit(), CBit()), reg=source_stale)

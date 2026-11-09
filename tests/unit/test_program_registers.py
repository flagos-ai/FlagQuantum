"""Typed words share the existing Program definition and ownership rules."""

from dataclasses import replace

import pytest

from flagquantum.core.ir.program import (
    BlockRecord,
    ConditionRef,
    FunctionRecord,
    Operation,
    ProgramRecord,
    ValueRef,
    program_to_circuit,
    verify_program_record,
)
from flagquantum.core.ir.program.registers import Register, RegisterSide, RegisterType

pytestmark = pytest.mark.unit


def word_function(width: int, name: str = "main") -> FunctionRecord:
    dtype = RegisterType("uint", width)
    before = ValueRef("entry", 0, "register")
    bits = tuple(ValueRef("entry", i + 1, "qubit") for i in range(width))
    after = ValueRef("entry", width + 1, "register")
    block = BlockRecord(
        "entry",
        (before,),
        operations=(
            Operation("split_register", (before,), bits),
            Operation("join_register", bits, (after,), register_type=dtype),
        ),
        return_values=(after,),
    )
    return FunctionRecord(name, (block,), registers=(Register("word", dtype),))


def codes(*functions: FunctionRecord) -> set[str]:
    return {item.code for item in verify_program_record(ProgramRecord(functions))}


@pytest.mark.parametrize(
    "encoding,width",
    [
        ("bit", 1),
        ("opaque", 0),
        ("opaque", 1),
        ("opaque", 3),
        ("uint", 0),
        ("uint", 1),
        ("uint", 3),
    ],
)
def test_concrete_words_infer_rebinding_types_from_producers(width, encoding):
    function = word_function(width)
    dtype = RegisterType(encoding, width)
    function = replace(function, registers=(Register("word", dtype),))
    assert codes(function) == set()
    # Loose source dtype policy permits these same-width return encodings.
    assert program_to_circuit(ProgramRecord((function,))).diagnostics


@pytest.mark.parametrize("shape", [(), (2,), (2, 3), (0, 2)])
@pytest.mark.parametrize("width", [0, 3])
def test_every_declared_element_requires_its_own_owned_value(shape, width):
    register = Register("word", RegisterType("uint", width), shape)
    values = tuple(
        ValueRef("entry", index, "register")
        for index, _ in enumerate(register.all_idxs())
    )
    block = BlockRecord("entry", values, return_values=values)
    function = FunctionRecord("main", (block,), registers=(register,))
    assert codes(function) == set()
    if len(values) > 1:
        alias = replace(block, arguments=values[:-1] + (values[0],))
        assert "program.value_redefined" in codes(replace(function, blocks=(alias,)))


@pytest.mark.parametrize(
    "mutation,expected",
    [
        ("missing_return", "program.call_structure"),
        ("empty_return", "value.linear_unconsumed"),
        ("stale_return", "value.linear_duplicate"),
        ("duplicate_bits", "value.linear_duplicate"),
        ("missing_split_bit", "program.result_kind"),
        ("wrong_join_width", "program.register_type"),
        ("missing_join_type", "program.register_type"),
        ("split_type", "program.register_structure"),
        ("gate_type", "program.register_structure"),
        ("missing_ports", "program.register_structure"),
        ("gate_input_kind", "program.register_structure"),
    ],
)
def test_malformed_words_fail_before_lowering(mutation, expected):
    function = word_function(3)
    block = function.blocks[0]
    split, join = block.operations
    if mutation == "missing_return":
        block = replace(block, return_values=None)
    elif mutation == "empty_return":
        block = replace(block, return_values=())
    elif mutation == "stale_return":
        block = replace(block, return_values=block.arguments)
    elif mutation == "duplicate_bits":
        join = replace(join, operands=(join.operands[0],) * 3)
    elif mutation == "missing_split_bit":
        split = replace(split, results=split.results[:-1])
    elif mutation == "wrong_join_width":
        join = replace(join, register_type=RegisterType("uint", 2))
    elif mutation == "missing_join_type":
        join = replace(join, register_type=None)
    elif mutation == "split_type":
        split = replace(split, register_type=RegisterType("uint", 3))
    elif mutation == "gate_type":
        split = replace(split, opcode="x", register_type=RegisterType("uint", 3))
    elif mutation == "missing_ports":
        function = replace(function, registers=None)
    elif mutation == "gate_input_kind":
        block = replace(block, arguments=(ValueRef("entry", 0, "qubit"),))
    block = replace(block, operations=(split, join))
    assert expected in codes(replace(function, blocks=(block,)))


@pytest.mark.parametrize(
    "metadata",
    [
        {"parameters": (("theta", ValueRef("entry", 9, "scalar")),)},
        {"literal": 0.5},
        {"condition_clauses": ((ConditionRef(ValueRef("entry", 9, "bool"), True),),)},
        {"callee": "main"},
    ],
)
@pytest.mark.parametrize("operation_index", [0, 1])
def test_rebindings_refuse_primitive_and_call_metadata(metadata, operation_index):
    function = word_function(1)
    block = function.blocks[0]
    operations = list(block.operations)
    operations[operation_index] = replace(operations[operation_index], **metadata)
    assert "program.register_structure" in codes(
        replace(function, blocks=(replace(block, operations=tuple(operations)),))
    )


@pytest.mark.parametrize(
    "dtype",
    [
        RegisterType("uint", -1),
        RegisterType("uint", 1.5),
        RegisterType("uint", True),
        RegisterType("bit", 2),
        RegisterType("signed", 3),
        RegisterType("uint", 3, 1),
        RegisterType("uint", 3, parameters=(("bound", 4),)),
    ],
)
def test_metadata_type_support_does_not_imply_executable_admission(dtype):
    function = word_function(3)
    assert "program.register_type" in codes(
        replace(function, registers=(Register("word", dtype),))
    )


@pytest.mark.parametrize("shape", [(-1,), (True,), (1.5,)])
def test_executable_shape_validation_is_distinct_from_metadata_creation(shape):
    function = word_function(3)
    assert "program.register_type" in codes(
        replace(function, registers=(Register("word", RegisterType("uint", 3), shape),))
    )


def test_opposite_side_names_and_producer_call_types_are_resolved():
    dtype = RegisterType("uint", 3)
    leaf = word_function(3, "leaf")
    before, after = (ValueRef("entry", i, "register") for i in range(2))
    block = BlockRecord(
        "entry",
        (before,),
        operations=(Operation("call", (before,), (after,), callee="leaf"),),
        return_values=(after,),
    )
    main = FunctionRecord(
        "main",
        (block,),
        registers=(
            Register("same", dtype, side=RegisterSide.LEFT),
            Register("same", RegisterType("opaque", 3), side=RegisterSide.RIGHT),
        ),
    )
    assert codes(main, leaf) == set()
    bad = replace(main, registers=(Register("same", RegisterType("uint", 2)),))
    assert "program.register_type" in codes(bad, leaf)
    with pytest.raises(ValueError, match="exactly"):
        # Physical schemas cannot silently acquire these internal fields.
        from flagquantum.core.ir.program.physical_codec import (
            native_program_from_dict,
            native_program_to_dict,
        )
        from flagquantum.core.ising import IsingBraidProgram

        payload = native_program_to_dict(IsingBraidProgram().exchange(0).to_ir())
        payload["functions"][0]["registers"] = None
        native_program_from_dict(payload)


def test_default_field_hashes_keep_original_gate_identity():
    value = ValueRef("entry", 0, "qubit")
    operation = Operation("x", (value,), (ValueRef("entry", 1, "qubit"),))
    assert hash(operation) == hash(
        ("x", operation.operands, operation.results, (), None, ())
    )
    block = BlockRecord("entry", (value,), operations=(operation,))
    function = FunctionRecord("main", (block,))
    assert hash(function) == hash(("main", (block,), "entry"))
    assert hash(replace(function, registers=())) == hash(
        ("main", (block,), "entry", ())
    )


def test_symbolic_width_and_shape_fail_at_executable_boundary():
    from sympy import Symbol

    function = word_function(1)
    symbol = Symbol("n", integer=True, nonnegative=True)
    for register in (
        Register("word", RegisterType("uint", symbol)),
        Register("word", RegisterType("uint", 1), (symbol,)),
    ):
        assert "program.register_type" in codes(
            replace(function, registers=(register,))
        )


@pytest.mark.parametrize(
    "case",
    [
        "duplicate_name",
        "mutable_ports",
        "invalid_side",
        "missing_input",
        "extra_return",
        "undefined_split",
    ],
)
def test_malformed_port_declarations_and_missing_producer_types_are_diagnostic(case):
    function = word_function(1)
    block = function.blocks[0]
    expected = "program.register_type"
    if case == "duplicate_name":
        function = replace(function, registers=function.registers * 2)
    elif case == "mutable_ports":
        function = replace(function, registers=list(function.registers))
    elif case == "invalid_side":
        function = replace(
            function,
            registers=(
                Register("word", RegisterType("uint", 1), side=RegisterSide(0)),
            ),
        )
    elif case == "missing_input":
        block = replace(block, arguments=())
        expected = "program.register_structure"
    elif case == "extra_return":
        block = replace(
            block, return_values=(*block.return_values, block.return_values[0])
        )
        expected = "program.register_structure"
    else:
        split, join = block.operations
        split = replace(split, operands=(ValueRef("entry", 10, "register"),))
        block = replace(block, operations=(split, join))
    assert expected in codes(replace(function, blocks=(block,)))


@pytest.mark.parametrize("field", ["registers", "register_type"])
def test_physical_source_schema_keeps_original_function_and_operation_keys(field):
    from flagquantum.core.ir.program.physical_codec import (
        native_program_from_dict,
        native_program_to_dict,
    )
    from flagquantum.core.ising import IsingBraidProgram
    from flagquantum.errors import SerializationError

    payload = native_program_to_dict(IsingBraidProgram().exchange(0).to_ir())
    assert verify_program_record(native_program_from_dict(payload)) == ()
    if field == "registers":
        payload["functions"][0][field] = None
    else:
        payload["functions"][0]["blocks"][0]["operations"][0][field] = None
    with pytest.raises(SerializationError, match="exactly fields"):
        native_program_from_dict(payload)

"""Workspace lifetime typing reuses Program SSA and exact consumption."""

from dataclasses import replace

import pytest

from flagquantum.core.ir.program import (
    BlockRecord,
    ConditionRef,
    FunctionRecord,
    Operation,
    ProgramRecord,
    ValueRef,
    verify_program_record,
)
from flagquantum.core.ir.program.registers import Register, RegisterSide, RegisterType

pytestmark = pytest.mark.unit


def workspace_function(width=3, dirty=False):
    value = ValueRef("entry", 0, "register")
    return FunctionRecord(
        "main",
        (
            BlockRecord(
                "entry",
                operations=(
                    Operation(
                        "allocate_register",
                        results=(value,),
                        register_type=RegisterType("uint", width),
                        workspace_dirty=dirty,
                    ),
                    Operation(
                        "free_register", operands=(value,), workspace_dirty=dirty
                    ),
                ),
                return_values=(),
            ),
        ),
        registers=(),
    )


def codes(function):
    return {d.code for d in verify_program_record(ProgramRecord((function,)))}


@pytest.mark.parametrize("width", [0, 1, 3])
@pytest.mark.parametrize("dirty", [False, True])
def test_workspace_ownership_is_independent_of_width_and_flag_matching(width, dirty):
    function = workspace_function(width, dirty)
    assert codes(function) == set()
    block = function.blocks[0]
    allocate, free = block.operations
    assert (
        codes(
            replace(
                function,
                blocks=(
                    replace(
                        block,
                        operations=(allocate, replace(free, workspace_dirty=not dirty)),
                    ),
                ),
            )
        )
        == set()
    )
    assert "value.linear_unconsumed" in codes(
        replace(function, blocks=(replace(block, operations=(allocate,)),))
    )
    assert "value.linear_duplicate" in codes(
        replace(function, blocks=(replace(block, operations=(allocate, free, free)),))
    )


@pytest.mark.parametrize("which", [0, 1])
@pytest.mark.parametrize("flag", [None, 0, 1, "dirty"])
def test_workspace_requires_explicit_builtin_boolean(which, flag):
    function = workspace_function()
    block = function.blocks[0]
    operations = list(block.operations)
    operations[which] = replace(operations[which], workspace_dirty=flag)
    assert "program.register_structure" in codes(
        replace(function, blocks=(replace(block, operations=tuple(operations)),))
    )


@pytest.mark.parametrize(
    "metadata",
    [
        {"parameters": (("theta", ValueRef("entry", 9)),)},
        {"literal": 0.5},
        {"condition_clauses": ((ConditionRef(ValueRef("entry", 9, "bool"), True),),)},
        {"callee": "main"},
        {"physical_data": object()},
    ],
)
@pytest.mark.parametrize("which", [0, 1])
def test_workspace_refuses_gate_and_call_metadata(which, metadata):
    function = workspace_function()
    block = function.blocks[0]
    operations = list(block.operations)
    operations[which] = replace(operations[which], **metadata)
    assert "program.register_structure" in codes(
        replace(function, blocks=(replace(block, operations=tuple(operations)),))
    )


@pytest.mark.parametrize(
    "mutation,expected",
    [
        ("allocate_input", "program.operand_arity"),
        ("allocate_output", "program.result_kind"),
        ("free_input", "program.operand_arity"),
        ("free_output", "program.result_kind"),
        ("free_type", "program.register_structure"),
        ("missing_type", "program.register_type"),
        ("unknown_type", "program.register_type"),
        ("negative_width", "program.register_type"),
        ("classic_type", "program.register_type"),
        ("missing_ports", "program.register_structure"),
        ("unknown_producer", "program.register_type"),
    ],
)
def test_malformed_workspace_is_diagnostic(mutation, expected):
    function = workspace_function()
    block = function.blocks[0]
    allocate, free = block.operations
    value = allocate.results[0]
    if mutation == "allocate_input":
        allocate = replace(allocate, operands=(value,))
    elif mutation == "allocate_output":
        allocate = replace(allocate, results=(replace(value, kind="qubit"),))
    elif mutation == "free_input":
        free = replace(free, operands=())
    elif mutation == "free_output":
        free = replace(free, results=(ValueRef("entry", 1, "register"),))
    elif mutation == "free_type":
        free = replace(free, register_type=allocate.register_type)
    elif mutation == "missing_type":
        allocate = replace(allocate, register_type=None)
    elif mutation == "unknown_type":
        allocate = replace(allocate, register_type=object())
    elif mutation == "negative_width":
        allocate = replace(allocate, register_type=RegisterType("uint", -1))
    elif mutation == "classic_type":
        allocate = replace(allocate, register_type=RegisterType("uint", 3, 1))
    elif mutation == "missing_ports":
        function = replace(function, registers=None)
    else:
        free = replace(free, operands=(ValueRef("entry", 9, "register"),))
    assert expected in codes(
        replace(function, blocks=(replace(block, operations=(allocate, free)),))
    )


def test_input_release_and_allocated_output_have_no_implicit_pair_constraint():
    function = workspace_function()
    block = function.blocks[0]
    allocated = block.operations[0].results[0]
    before = ValueRef("entry", 1, "register")
    block = replace(
        block,
        arguments=(before,),
        operations=(
            block.operations[0],
            replace(block.operations[1], operands=(before,)),
        ),
        return_values=(allocated,),
    )
    dtype = RegisterType("uint", 3)
    function = replace(
        function,
        blocks=(block,),
        registers=(
            Register("input", dtype, side=RegisterSide.LEFT),
            Register("output", dtype, side=RegisterSide.RIGHT),
        ),
    )
    assert codes(function) == set()


def test_dirty_metadata_on_other_operations_is_rejected_without_register_ports():
    function = FunctionRecord(
        "main",
        (
            BlockRecord(
                "entry",
                operations=(
                    Operation(
                        "bool_zero",
                        results=(ValueRef("entry", 0, "bool"),),
                        workspace_dirty=False,
                    ),
                ),
                return_values=(),
            ),
        ),
    )
    assert "program.register_structure" in codes(function)


def test_workspace_metadata_does_not_change_old_hash_or_physical_schema():
    from flagquantum.core.ir.program.physical_codec import (
        native_program_from_dict,
        native_program_to_dict,
    )
    from flagquantum.core.ising import IsingBraidProgram
    from flagquantum.errors import SerializationError

    operation = Operation("bool_zero", results=(ValueRef("entry", 0, "bool"),))
    assert hash(operation) == hash(("bool_zero", (), operation.results, (), None, ()))
    assert hash(replace(operation, workspace_dirty=False)) == hash(
        ("bool_zero", (), operation.results, (), None, (), False)
    )
    payload = native_program_to_dict(IsingBraidProgram().exchange(0).to_ir())
    assert verify_program_record(native_program_from_dict(payload)) == ()
    payload["functions"][0]["blocks"][0]["operations"][0]["workspace_dirty"] = None
    with pytest.raises(SerializationError, match="exactly fields"):
        native_program_from_dict(payload)

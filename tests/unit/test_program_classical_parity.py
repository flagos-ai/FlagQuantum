"""Owned parity successors and shareable feedback snapshots in existing SSA."""

from __future__ import annotations

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
from flagquantum.core.ir.program.register_types import (
    classical_bit_type,
    quantum_bit_type,
)
from flagquantum.core.ir.program.registers import Register, RegisterSide

pytestmark = pytest.mark.unit


def parity_function(count: int) -> FunctionRecord:
    inputs = tuple(ValueRef("entry", i, "register") for i in range(count))
    outputs = tuple(ValueRef("entry", count + i, "register") for i in range(count + 1))
    return FunctionRecord(
        "main",
        (
            BlockRecord(
                "entry",
                inputs,
                operations=(Operation("parity_register", inputs, outputs),),
                return_values=outputs,
            ),
        ),
        registers=(
            Register("x", classical_bit_type(), (count,)),
            Register("parity", classical_bit_type(), side=RegisterSide.RIGHT),
        ),
    )


def read_function() -> FunctionRecord:
    c, successor = (ValueRef("entry", i, "register") for i in range(2))
    snapshot = ValueRef("entry", 2, "bool")
    return FunctionRecord(
        "main",
        (
            BlockRecord(
                "entry",
                (c,),
                operations=(
                    Operation("read_classical_register", (c,), (successor, snapshot)),
                ),
                return_values=(successor,),
            ),
        ),
        registers=(Register("c", classical_bit_type()),),
    )


def codes(function: FunctionRecord) -> set[str]:
    return {item.code for item in verify_program_record(ProgramRecord((function,)))}


@pytest.mark.parametrize("count", [0, 1, 2, 5])
def test_parity_derives_one_owned_successor_per_input_and_one_output(
    count: int,
) -> None:
    assert codes(parity_function(count)) == set()


def test_feedback_snapshot_can_fan_out_while_classical_owner_is_forwarded() -> None:
    function = read_function()
    block = function.blocks[0]
    snapshot = block.operations[0].results[-1]
    first, second = (ValueRef("entry", i, "bool") for i in (3, 4))
    operations = (
        *block.operations,
        Operation("bool_xor", (snapshot, snapshot), (first,)),
        Operation("bool_xor", (snapshot, first), (second,)),
    )
    assert (
        codes(replace(function, blocks=(replace(block, operations=operations),)))
        == set()
    )


@pytest.mark.parametrize("function", [parity_function(2), read_function()])
def test_classical_operations_reject_quantum_input_and_undeclared_ports(
    function: FunctionRecord,
) -> None:
    assert function.registers is not None
    port = function.registers[0]
    wrong = replace(
        function,
        registers=(
            Register(port.name, quantum_bit_type(), port.shape, port.side),
            *function.registers[1:],
        ),
    )
    assert "program.register_type" in codes(wrong)
    assert "program.register_structure" in codes(replace(function, registers=None))


@pytest.mark.parametrize("function", [parity_function(2), read_function()])
@pytest.mark.parametrize(
    "defect", ["arity", "result", "condition", "type", "flag", "literal", "parameter"]
)
def test_classical_operations_reject_malformed_shapes_and_metadata(
    function: FunctionRecord, defect: str
) -> None:
    block = function.blocks[0]
    op = block.operations[0]
    if defect == "arity":
        # Parity arity follows its operands, so its now-short result tuple fails too.
        op = replace(op, operands=op.operands + (block.arguments[0],))
    elif defect == "result":
        op = replace(op, results=op.results[:-1])
    elif defect == "condition":
        op = replace(
            op,
            condition_clauses=((ConditionRef(ValueRef("entry", 100, "bool"), True),),),
        )
    elif defect == "type":
        op = replace(op, register_type=classical_bit_type())
    elif defect == "flag":
        op = replace(op, workspace_dirty=False)
    elif defect == "literal":
        op = replace(op, literal=1.0)
    else:
        op = replace(op, parameters=(("x", ValueRef("entry", 100, "scalar")),))
    assert codes(replace(function, blocks=(replace(block, operations=(op,)),)))


@pytest.mark.parametrize("function", [parity_function(2), read_function()])
def test_classical_operations_reject_reusing_consumed_owner(
    function: FunctionRecord,
) -> None:
    block = function.blocks[0]
    assert block.return_values is not None
    wrong = replace(block, return_values=(block.arguments[0], *block.return_values[1:]))
    assert "value.linear_duplicate" in codes(replace(function, blocks=(wrong,)))

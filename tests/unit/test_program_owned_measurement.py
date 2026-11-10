"""Destructive QBit to owned CBit typing and single consumption."""

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
from flagquantum.core.ir.program.physical import RydbergPresence
from flagquantum.core.ir.program.register_types import (
    classical_bit_type,
    quantum_bit_type,
)
from flagquantum.core.ir.program.registers import Register, RegisterSide, RegisterType

pytestmark = pytest.mark.unit


def measured_function(name: str = "main") -> FunctionRecord:
    q, c = (ValueRef("entry", i, "register") for i in range(2))
    return FunctionRecord(
        name,
        (
            BlockRecord(
                "entry",
                (q,),
                operations=(Operation("measure_register", (q,), (c,)),),
                return_values=(c,),
            ),
        ),
        registers=(
            Register("q", quantum_bit_type(), side=RegisterSide.LEFT),
            Register("c", classical_bit_type(), side=RegisterSide.RIGHT),
        ),
    )


def test_measurement_consumes_quantum_and_defines_owned_classical_port() -> None:
    assert verify_program_record(ProgramRecord((measured_function(),))) == ()


@pytest.mark.parametrize(
    "dtype",
    [classical_bit_type(), RegisterType("uint", 1), RegisterType("opaque", 0)],
)
def test_measurement_rejects_other_owned_input_types(dtype: RegisterType) -> None:
    function = measured_function()
    assert function.registers is not None
    function = replace(
        function,
        registers=(
            Register("q", dtype, side=RegisterSide.LEFT),
            *function.registers[1:],
        ),
    )
    assert "program.register_type" in {
        item.code for item in verify_program_record(ProgramRecord((function,)))
    }


@pytest.mark.parametrize(
    "operation,expected",
    [
        (Operation("measure_register"), "program.operand_arity"),
        (
            replace(measured_function().blocks[0].operations[0], results=()),
            "program.result_kind",
        ),
        (
            replace(measured_function().blocks[0].operations[0], literal=1.0),
            "program.register_structure",
        ),
        (
            replace(
                measured_function().blocks[0].operations[0],
                physical_data=RydbergPresence("atom", 0.0, 1.0),
            ),
            "program.register_structure",
        ),
        (
            replace(measured_function().blocks[0].operations[0], callee="other"),
            "program.register_structure",
        ),
        (
            replace(
                measured_function().blocks[0].operations[0],
                register_type=quantum_bit_type(),
            ),
            "program.register_structure",
        ),
        (
            replace(measured_function().blocks[0].operations[0], workspace_dirty=False),
            "program.register_structure",
        ),
        (
            replace(
                measured_function().blocks[0].operations[0],
                condition_clauses=(
                    (ConditionRef(ValueRef("entry", 9, "bool"), True),),
                ),
            ),
            "program.register_structure",
        ),
    ],
)
def test_measurement_rejects_malformed_shape_and_foreign_metadata(
    operation: Operation, expected: str
) -> None:
    function = measured_function()
    block = function.blocks[0]
    function = replace(function, blocks=(replace(block, operations=(operation,)),))
    assert expected in {
        item.code for item in verify_program_record(ProgramRecord((function,)))
    }


def test_measurement_cannot_return_consumed_qubit_or_duplicate_classical_owner() -> (
    None
):
    function = measured_function()
    block = function.blocks[0]
    assert block.return_values is not None
    for returned in ((block.arguments[0],), block.return_values * 2):
        malformed = replace(function, blocks=(replace(block, return_values=returned),))
        assert verify_program_record(ProgramRecord((malformed,)))
    without_ports = replace(function, registers=None)
    assert "program.register_structure" in {
        item.code for item in verify_program_record(ProgramRecord((without_ports,)))
    }

"""Owned classical discard is a typed single-consumer operation."""

from __future__ import annotations

from dataclasses import replace

import pytest

from flagquantum.core.ir.program import (
    BlockRecord,
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
from tests.unit.test_program_classical_parity import codes

pytestmark = pytest.mark.unit


def discard_function() -> FunctionRecord:
    c = ValueRef("entry", 0, "register")
    return FunctionRecord(
        "discard",
        (
            BlockRecord(
                "entry",
                (c,),
                operations=(Operation("discard_classical_register", (c,)),),
                return_values=(),
            ),
        ),
        registers=(Register("c", classical_bit_type(), side=RegisterSide.LEFT),),
    )


def test_discard_accepts_one_classical_owner_without_outputs() -> None:
    assert verify_program_record(ProgramRecord((discard_function(),))) == ()


@pytest.mark.parametrize(
    "defect",
    [
        "missing",
        "extra",
        "result",
        "quantum",
        "undeclared",
        "duplicate",
        "stale",
        "literal",
        "descriptor",
        "dirty",
    ],
)
def test_discard_rejects_invalid_type_shape_metadata_and_lifetime(defect: str) -> None:
    function = discard_function()
    block = function.blocks[0]
    op = block.operations[0]
    if defect == "missing":
        op = replace(op, operands=())
    elif defect == "extra":
        op = replace(op, operands=op.operands * 2)
    elif defect == "result":
        op = replace(op, results=(ValueRef("entry", 1, "register"),))
    elif defect == "quantum":
        function = replace(
            function,
            registers=(Register("c", quantum_bit_type(), side=RegisterSide.LEFT),),
        )
    elif defect == "undeclared":
        function = replace(function, registers=None)
    elif defect == "literal":
        op = replace(op, literal=1.0)
    elif defect == "descriptor":
        op = replace(op, register_type=classical_bit_type())
    elif defect == "dirty":
        op = replace(op, workspace_dirty=False)
    operations = (op, op) if defect == "duplicate" else (op,)
    returned = block.arguments if defect == "stale" else ()
    if defect == "stale":
        function = replace(function, registers=(Register("c", classical_bit_type()),))
    assert codes(
        replace(
            function,
            blocks=(replace(block, operations=operations, return_values=returned),),
        )
    )

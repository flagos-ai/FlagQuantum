"""Owned writes preserve nominal words, old snapshots and canonical conditions."""

from dataclasses import replace

import pytest
from sympy import Symbol

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
    classical_unsigned_register_type,
    quantum_bit_type,
)
from flagquantum.core.ir.program.register_verifier import (
    executable_classical_register_width,
)
from flagquantum.core.ir.program.registers import Register, RegisterType
from tests.unit.test_classical_word_program import TYPES

pytestmark = pytest.mark.unit
FIELD_TYPES = (
    RegisterType(
        "finite_field",
        0,
        2,
        (("storage", "classical"), ("characteristic", 3), ("degree", 1), ("order", 3)),
    ),
    RegisterType(
        "finite_field_polynomial",
        0,
        4,
        (
            ("storage", "classical"),
            ("polynomial_degree", 1),
            ("characteristic", 3),
            ("degree", 1),
            ("order", 3),
        ),
    ),
)


def write_function(dtype: RegisterType, *, conditional: bool = False) -> FunctionRecord:
    """Complement all bits using immutable old snapshots and return the owner."""
    width = executable_classical_register_width(dtype)
    before, read, after = (ValueRef("entry", i, "register") for i in range(3))
    one = ValueRef("entry", 3, "bool")
    old = tuple(ValueRef("entry", 4 + i, "bool") for i in range(width))
    new = tuple(ValueRef("entry", 4 + width + i, "bool") for i in range(width))
    operations = (
        Operation("read_classical_register", (before,), (read, *old)),
        Operation("bool_one", results=(one,)),
        *(
            Operation("bool_xor", (value, one), (result,))
            for value, result in zip(old, new, strict=True)
        ),
        Operation(
            "write_classical_register",
            (read, *new),
            (after,),
            condition_clauses=((ConditionRef(one, True),),) if conditional else (),
        ),
    )
    return FunctionRecord(
        "main",
        (
            BlockRecord(
                "entry", (before,), operations=operations, return_values=(after,)
            ),
        ),
        registers=(Register("c", dtype),),
    )


@pytest.mark.parametrize("dtype", (*TYPES, *FIELD_TYPES))
@pytest.mark.parametrize("conditional", (False, True))
def test_write_nominal_type_width_and_conditional_successor(
    dtype: RegisterType,
    conditional: bool,
) -> None:
    function = write_function(dtype, conditional=conditional)
    assert verify_program_record(ProgramRecord((function,))) == ()
    block = function.blocks[0]
    write = block.operations[-1]
    for malformed in (
        replace(write, operands=(*write.operands, ValueRef("entry", 3, "bool"))),
        replace(write, results=()),
        replace(write, results=(ValueRef("entry", 2, "bool"),)),
        replace(write, literal=1.0),
        replace(write, register_type=dtype),
    ):
        bad = replace(
            function,
            blocks=(replace(block, operations=(*block.operations[:-1], malformed)),),
        )
        assert verify_program_record(ProgramRecord((bad,)))
    stale = replace(
        function,
        blocks=(replace(block, return_values=(block.operations[0].results[0],)),),
    )
    assert "value.linear_duplicate" in {
        d.code for d in verify_program_record(ProgramRecord((stale,)))
    }


def test_write_rejects_quantum_symbolic_and_undefined_conditions() -> None:
    function = write_function(classical_unsigned_register_type(1), conditional=True)
    block = function.blocks[0]
    write = block.operations[-1]
    invalid = replace(
        write, condition_clauses=((ConditionRef(ValueRef("entry", 99, "bool"), True),),)
    )
    bad = replace(
        function, blocks=(replace(block, operations=(*block.operations[:-1], invalid)),)
    )
    assert "program.condition_undefined" in {
        d.code for d in verify_program_record(ProgramRecord((bad,)))
    }
    for dtype in (quantum_bit_type(), classical_unsigned_register_type(Symbol("n"))):
        bad = replace(function, registers=(Register("c", dtype),))
        assert verify_program_record(ProgramRecord((bad,)))

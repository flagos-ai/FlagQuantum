"""Typed classical reads preserve owned word types and read-only MSB snapshots."""

from dataclasses import replace

import pytest
from sympy import Symbol

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
    classical_bounded_unsigned_register_type,
    classical_fixed_point_register_type,
    classical_montgomery_register_type,
    classical_ones_complement_register_type,
    classical_signed_register_type,
    classical_unsigned_register_type,
)
from flagquantum.core.ir.program.register_verifier import (
    executable_classical_register_width,
    register_port_types,
)
from flagquantum.core.ir.program.registers import Register, RegisterSide, RegisterType

pytestmark = pytest.mark.unit
TYPES = (
    classical_bit_type(),
    classical_unsigned_register_type(3),
    classical_signed_register_type(3),
    classical_ones_complement_register_type(3),
    classical_bounded_unsigned_register_type(3, 5),
    classical_montgomery_register_type(3, 5),
    classical_fixed_point_register_type(3, 1, True),
    classical_unsigned_register_type(0),
)


def read_function(dtype: RegisterType, *, delta: int = 0) -> FunctionRecord:
    width = executable_classical_register_width(dtype)
    before, after = (ValueRef("entry", i, "register") for i in range(2))
    snapshots = tuple(ValueRef("entry", 2 + i, "bool") for i in range(width + delta))
    return FunctionRecord(
        "main",
        (
            BlockRecord(
                "entry",
                (before,),
                operations=(
                    Operation(
                        "read_classical_register", (before,), (after, *snapshots)
                    ),
                ),
                return_values=(after,),
            ),
        ),
        registers=(Register("c", dtype),),
    )


@pytest.mark.parametrize("dtype", TYPES)
def test_read_arity_and_identical_successor(dtype: RegisterType) -> None:
    function = read_function(dtype)
    assert verify_program_record(ProgramRecord((function,))) == ()
    assert register_port_types(function, RegisterSide.RIGHT) == (dtype,)
    operation = function.blocks[0].operations[0]
    assert tuple(v.kind for v in operation.results) == (
        "register",
        *("bool",) * executable_classical_register_width(dtype),
    )
    malformed = read_function(dtype, delta=1)
    assert verify_program_record(ProgramRecord((malformed,)))


@pytest.mark.parametrize("dtype", TYPES)
def test_linear_successor_cannot_return_stale_input(dtype: RegisterType) -> None:
    function = read_function(dtype)
    block = function.blocks[0]
    function = replace(
        function, blocks=(replace(block, return_values=block.arguments),)
    )
    assert "value.linear_duplicate" in {
        d.code for d in verify_program_record(ProgramRecord((function,)))
    }


@pytest.mark.parametrize(
    "dtype",
    [
        classical_unsigned_register_type(Symbol("n", integer=True, positive=True)),
        classical_montgomery_register_type(3, Symbol("p", integer=True, positive=True)),
    ],
)
def test_symbolic_identity_declaration_and_read_refusal(dtype: RegisterType) -> None:
    value = ValueRef("entry", 0, "register")
    function = FunctionRecord(
        "main",
        (BlockRecord("entry", (value,), return_values=(value,)),),
        registers=(Register("c", dtype),),
    )
    assert verify_program_record(ProgramRecord((function,))) == ()
    with pytest.raises(ValueError, match="concrete"):
        executable_classical_register_width(dtype)


@pytest.mark.parametrize(
    "dtype",
    [
        RegisterType("uint", 1, 3, (("storage", "classical"),)),
        RegisterType("uint", 0, 3, (("bound", 5), ("storage", "classical"))),
        RegisterType("uint", 0, 3, (("storage", "classical"), ("bound", True))),
        RegisterType("unknown", 0, 3, (("storage", "classical"),)),
        RegisterType(
            "fixed_point",
            0,
            3,
            (("storage", "classical"), ("fractional_width", 1), ("signed", 1)),
        ),
    ],
)
def test_invalid_descriptors_fail_canonical_admission(dtype: RegisterType) -> None:
    with pytest.raises(ValueError):
        executable_classical_register_width(dtype)

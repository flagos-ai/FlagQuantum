"""Proposal 120 retains symbolic declarations and rejects finite shape use."""

from dataclasses import replace
from typing import Any

import pytest
import sympy

from flagquantum.core.ir.program import (
    BlockRecord,
    FunctionRecord,
    Operation,
    ProgramRecord,
    verify_program_record,
)
from flagquantum.core.ir.program.register_verifier import register_port_types
from flagquantum.core.ir.program.registers import Register, RegisterSide, RegisterType
from flagquantum.core.value_math import SymbolicInt

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "shape",
    [
        (sympy.Symbol("n"),),
        (2, sympy.Symbol("n")),
        (sympy.Symbol("n"), 0, 3),
        (sympy.Integer(2),),
    ],
)
@pytest.mark.parametrize(
    "side", [RegisterSide.LEFT, RegisterSide.RIGHT, RegisterSide.THRU]
)
def test_symbolic_shape_is_metadata_until_finite_use(
    shape: tuple[SymbolicInt, ...], side: RegisterSide
) -> None:
    register = Register("word", RegisterType("uint", 2), shape, side)
    declaration = FunctionRecord("symbolic", (), registers=(register,))
    assert verify_program_record(ProgramRecord((declaration,))) == ()
    assert declaration.registers == (register,)
    for direction in (RegisterSide.LEFT, RegisterSide.RIGHT):
        with pytest.raises(ValueError, match="executable array dimensions"):
            register_port_types(declaration, direction)
    body = replace(declaration, blocks=(BlockRecord("entry", (), return_values=()),))
    assert any(
        d.code == "program.register_type"
        for d in verify_program_record(ProgramRecord((body,)))
    )
    caller = FunctionRecord(
        "caller",
        (
            BlockRecord(
                "entry",
                (),
                operations=(Operation("call", (), (), callee="symbolic"),),
                return_values=(),
            ),
        ),
        registers=(),
    )
    diagnostics = verify_program_record(ProgramRecord((caller, declaration)))
    assert any(
        d.code == "program.register_type" and "array dimensions" in d.message
        for d in diagnostics
    )


@pytest.mark.parametrize("dimension", [-1, True, 1.5, "n"])
def test_invalid_concrete_dimensions_still_fail(dimension: Any) -> None:
    register = Register("word", RegisterType("uint", 2), (dimension,))
    declaration = FunctionRecord("invalid", (), registers=(register,))
    assert any(
        d.code == "program.register_type"
        for d in verify_program_record(ProgramRecord((declaration,)))
    )

"""Triangular-index wire declarations preserve all source ports and boundaries."""

from typing import Any

import pytest
import sympy

from flagquantum.compiler.formats.legacy_component_resolution import (
    resolve_component_program,
)
from flagquantum.compiler.formats.legacy_contiguous_index_factories import (
    contiguous_index_factories,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir.program import verify_program_record
from flagquantum.core.ir.program.registers import RegisterSide
from flagquantum.errors import SerializationError
from tests.unit.test_legacy_arithmetic_factories import _library

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
_NAME = "qualtran.bloqs.arithmetic.conversions.contiguous_index.ToContiguousIndex"


@pytest.mark.parametrize(
    "width,result",
    ((1, 1), (2, 4), (4, 8), (5, 10), (sympy.Symbol("n"), sympy.Symbol("m"))),
)
def test_complete_constructor_original_wire_ports_model_and_physical_refusal(
    width: Any, result: Any
) -> None:
    from qualtran import DecomposeNotImplementedError
    from qualtran.bloqs.arithmetic.conversions.contiguous_index import ToContiguousIndex
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    source = ToContiguousIndex(width, result)
    library = _library(source, 0)
    program = resolve_component_program(library, 0, contiguous_index_factories())
    assert verify_program_record(program) == ()
    registers = program.functions[0].registers
    assert registers == library.entries[0].declaration.registers
    assert registers is not None
    assert tuple(r.name for r in registers) == ("mu", "nu", "s")
    assert all(r.side is RegisterSide.THRU for r in registers)
    assert not program.functions[0].blocks
    with pytest.raises(DecomposeNotImplementedError):
        source.decompose_bloq()
    with pytest.raises(ValueError, match="bodies"):
        bind_register_program(program, ())
    assert get_cost_value(source, QECGatesCost()).total_t_count() == 4 * (
        width**2 + width - 1
    )


def test_invalid_fields_and_signature_refuse() -> None:
    factory = contiguous_index_factories()[_NAME]
    for arguments in (
        {},
        {"bitsize": 2, "s_bitsize": 4, "extra": 1},
        {"bitsize": "bad", "s_bitsize": 4},
    ):
        with pytest.raises(SerializationError):
            factory("bad", None, arguments)
    with pytest.raises(SerializationError, match="signature"):
        factory("bad", (), {"bitsize": 2, "s_bitsize": 4})

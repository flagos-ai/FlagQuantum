"""Source signed conversion declarations remain separate from native execution."""

from collections import Counter
from typing import Any

import pytest
import sympy

from flagquantum.compiler.formats.legacy_component_resolution import (
    resolve_component_program,
)
from flagquantum.compiler.formats.legacy_signed_encoding_factories import (
    signed_encoding_factories,
)
from flagquantum.compiler.registers.bits.signed_encoding_component import (
    signed_encoding_program,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir.program import verify_program_record
from flagquantum.errors import SerializationError
from tests.unit.test_legacy_arithmetic_factories import _library

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize(
    "width", (0, 2, 3, 5, 10, sympy.Symbol("n", positive=True, integer=True))
)
def test_source_exact_sided_constructor_retains_missing_body_and_model(
    width: Any,
) -> None:
    from qualtran import DecomposeNotImplementedError
    from qualtran.bloqs.arithmetic.conversions.ones_complement_to_twos_complement import (
        SignedIntegerToTwosComplement,
    )
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    source = SignedIntegerToTwosComplement(width)
    library = _library(source, 0)
    program = resolve_component_program(library, 0, signed_encoding_factories())
    assert verify_program_record(program) == ()
    function = program.functions[0]
    assert function.registers == library.entries[0].declaration.registers
    assert function.registers is not None
    assert [(r.name, r.side.name, r.dtype.encoding) for r in function.registers] == [
        ("x", "LEFT", "ones_complement"),
        ("y", "RIGHT", "signed"),
    ]
    assert not function.blocks
    with pytest.raises(ValueError, match="bodies"):
        bind_register_program(program, ())
    with pytest.raises(DecomposeNotImplementedError):
        source.decompose_bloq()
    model = get_cost_value(source, QECGatesCost())
    assert model.toffoli == width - 2
    if type(width) is not int or width < 2:
        return
    bound = bind_register_program(
        signed_encoding_program(width),
        (tuple(range(width)),),
        clean=tuple(range(width, 2 * width - 2)),
    )
    counts = Counter(g.name for g in bound.circuit.instructions)
    assert counts["t"] + counts["tdg"] == 4 * (width - 2)
    assert counts["measure"] == counts["reset"] == width - 2
    assert counts["cx"] == 7 * (width - 2) + 1
    assert model.clifford == model.measurement == 0


def test_actual_signed_conversion_example_and_invalid_metadata() -> None:
    from qualtran.bloqs.arithmetic.conversions.ones_complement_to_twos_complement import (
        _signed_to_twos,
    )

    source = _signed_to_twos.make()
    library = _library(source, 0)
    result = resolve_component_program(library, 0, signed_encoding_factories())
    assert result.functions[0].registers == library.entries[0].declaration.registers
    factory = next(iter(signed_encoding_factories().values()))
    for arguments in ({}, {"bitsize": True}, {"bitsize": 3, "extra": 1}):
        with pytest.raises(SerializationError):
            factory("bad", None, arguments)
    with pytest.raises(SerializationError, match="signature"):
        factory("bad", (), {"bitsize": 3})
    with pytest.raises(ValueError, match="bitsize"):
        factory("bad", None, {"bitsize": 1})

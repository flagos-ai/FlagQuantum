"""Trigonometric constructors retain source signatures and model-only boundaries."""

from importlib import import_module
from typing import Any

import pytest
import sympy

from flagquantum.compiler.formats.legacy_component_resolution import (
    resolve_component_program,
)
from flagquantum.compiler.formats.legacy_trigonometric_factories import (
    trigonometric_factories,
)
from flagquantum.compiler.registers.fixed_point.arcsine_declaration import (
    ArcsineDeclaration,
)
from flagquantum.compiler.registers.lookup.arctangent_declaration import (
    ArctangentDeclaration,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir.program import verify_program_record
from flagquantum.errors import SerializationError
from tests.unit.test_legacy_arithmetic_factories import _library

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
_PREFIX = "qualtran.bloqs.arithmetic.trigonometric."


@pytest.mark.parametrize(
    "kind,arguments",
    (
        ("arcsin", {"bitsize": 10, "num_frac": 7}),
        ("arcsin", {"bitsize": 4, "num_frac": 4, "num_iters": 3, "degree": 6}),
        ("arcsin", {"bitsize": sympy.Symbol("n"), "num_frac": sympy.Symbol("p")}),
        ("arctan", {"selection_bitsize": 1, "target_bitsize": 1}),
        ("arctan", {"selection_bitsize": 3, "target_bitsize": 5}),
        (
            "arctan",
            {
                "selection_bitsize": sympy.Symbol("n"),
                "target_bitsize": sympy.Symbol("m"),
            },
        ),
    ),
)
def test_complete_constructor_wire_and_shared_declaration_signature(
    kind: str, arguments: dict[str, Any]
) -> None:
    from qualtran import DecomposeNotImplementedError, DecomposeTypeError

    cls = "ArcSin" if kind == "arcsin" else "ArcTan"
    source = getattr(import_module(_PREFIX + kind), cls)(**arguments)
    native = (
        ArcsineDeclaration(**arguments)
        if kind == "arcsin"
        else ArctangentDeclaration(**arguments)
    )
    library = _library(source, 0)
    program = resolve_component_program(library, 0, trigonometric_factories())
    assert verify_program_record(program) == ()
    # Symbol assumptions are intentionally absent from the original wire.
    assert program.functions[0].registers == tuple(native.signature)
    assert program.functions[0].registers == library.entries[0].declaration.registers
    assert not program.functions[0].blocks
    expected_error = (
        DecomposeTypeError
        if kind == "arctan" and isinstance(arguments["selection_bitsize"], sympy.Expr)
        else DecomposeNotImplementedError
    )
    with pytest.raises(expected_error):
        source.decompose_bloq()
    with pytest.raises(ValueError, match="bodies"):
        bind_register_program(program, ())


def test_default_iteration_degree_and_invalid_metadata_reuse_original_owner() -> None:
    factories = trigonometric_factories()
    key = _PREFIX + "arcsin.ArcSin"
    implicit = factories[key]("arcsin", None, {"bitsize": 4, "num_frac": 3})
    explicit = factories[key](
        "arcsin", None, {"bitsize": 4, "num_frac": 3, "num_iters": 4, "degree": 4}
    )
    assert implicit == explicit
    for arguments in (
        {},
        {"bitsize": 4, "num_frac": 3, "extra": 1},
        {"bitsize": 4, "num_frac": "bad"},
    ):
        with pytest.raises(SerializationError):
            factories[key]("bad", None, arguments)
    with pytest.raises(ValueError):
        factories[key]("bad", None, {"bitsize": 4, "num_frac": 5})
    with pytest.raises(SerializationError, match="signature"):
        factories[key]("bad", (), {"bitsize": 4, "num_frac": 3})

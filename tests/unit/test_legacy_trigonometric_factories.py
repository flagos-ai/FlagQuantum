"""Trigonometric constructors retain source signatures and model-only boundaries."""

from importlib import import_module
from typing import Any

import pytest
import sympy

from flagquantum.compiler.formats.legacy_component_resolution import (
    resolve_component_program,
)
from flagquantum.compiler.formats.legacy_trigonometric_factories import (
    arctan_lookup_factory,
    trigonometric_factories,
)
from flagquantum.compiler.registers.fixed_point.arcsine_declaration import (
    ArcsineDeclaration,
)
from flagquantum.compiler.registers.lookup.arctan_lookup_component import (
    arctan_lookup_program,
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
_KEY = _PREFIX + "arctan.ArcTan"


@pytest.mark.parametrize(
    "kind,arguments",
    (
        ("arcsin", {"bitsize": 10, "num_frac": 7}),
        ("arcsin", {"bitsize": 0, "num_frac": 0}),
        ("arcsin", {"bitsize": 0, "num_frac": -1}),
        ("arcsin", {"bitsize": 0, "num_frac": -2}),
        ("arcsin", {"bitsize": 4, "num_frac": 4, "num_iters": 3, "degree": 6}),
        ("arcsin", {"bitsize": sympy.Symbol("n"), "num_frac": sympy.Symbol("p")}),
        ("arctan", {"selection_bitsize": 1, "target_bitsize": 1}),
        ("arctan", {"selection_bitsize": 0, "target_bitsize": 3}),
        ("arctan", {"selection_bitsize": 3, "target_bitsize": 0}),
        ("arctan", {"selection_bitsize": 0, "target_bitsize": 0}),
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


@pytest.mark.parametrize("depth", (0, 1, 2))
@pytest.mark.parametrize("source,target", ((0, 0), (0, 3), (2, 0), (2, 8)))
def test_actual_constructor_replacement_retains_model_only_default(
    depth: int, source: int, target: int
) -> None:
    from qualtran.bloqs.arithmetic.trigonometric import ArcTan

    library = _library(ArcTan(source, target), depth)
    default = resolve_component_program(library, 0, trigonometric_factories())
    assert not default.functions[0].blocks
    with pytest.raises(ValueError, match="bodies"):
        bind_register_program(default, ())
    factories = trigonometric_factories()
    factories[_KEY] = arctan_lookup_factory
    concrete = resolve_component_program(library, 0, factories)
    assert verify_program_record(concrete) == ()
    assert concrete.functions[0].registers == default.functions[0].registers
    logical = source + 1 + target
    ports = tuple(
        p
        for p in (
            tuple(range(source)),
            (source,),
            tuple(range(source + 1, logical)),
        )
        if p
    )
    work = tuple(range(logical, logical + max(0, source - 1)))
    bound = bind_register_program(concrete, ports, clean=work)
    direct = bind_register_program(
        arctan_lookup_program(source, target), ports, clean=work
    )
    assert bound.circuit.instructions == direct.circuit.instructions


def test_explicit_lookup_preserves_profile_and_constructor_validation() -> None:
    for source, target in ((-1, 8), (13, 8), (2, -1), (2, 33), (True, 8)):
        with pytest.raises(ValueError):
            arctan_lookup_program(source, target)
    arguments = {"selection_bitsize": 2, "target_bitsize": 8}
    with pytest.raises(SerializationError, match="signature"):
        arctan_lookup_factory("bad", (), arguments)
    with pytest.raises(SerializationError, match="arguments"):
        arctan_lookup_factory("bad", None, {**arguments, "unexpected": 1})
    with pytest.raises(SerializationError, match="concrete"):
        arctan_lookup_factory(
            "symbolic", None, {**arguments, "target_bitsize": sympy.Symbol("p")}
        )

"""Product constructors retain their original typed no-decomposition boundary."""

from importlib import import_module
from typing import Any

import pytest
import sympy

from flagquantum.compiler.formats.legacy_component_resolution import (
    resolve_component_program,
)
from flagquantum.compiler.formats.legacy_product_model_factories import (
    product_model_factories,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir.program import FunctionRecord, verify_program_record
from flagquantum.core.ir.program.registers import RegisterSide
from flagquantum.errors import SerializationError
from tests.unit.test_legacy_arithmetic_factories import _library

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
_PREFIX = "qualtran.bloqs.arithmetic.multiplication."
_EXAMPLES = (
    "_plus_equal_product",
    "_square",
    "_sum_of_squares",
    "_product",
    "_scale_int_by_real",
    "_multiply_two_reals",
    "_square_real_number",
    "_invert_real_number",
)


@pytest.mark.parametrize("example", _EXAMPLES)
def test_complete_original_examples_round_trip_as_model_declarations(
    example: str,
) -> None:
    from qualtran import DecomposeNotImplementedError

    source = getattr(import_module(_PREFIX[:-1]), example)()
    library = _library(source, 0)
    program = resolve_component_program(library, 0, product_model_factories())
    assert verify_program_record(program) == ()
    assert not program.functions[0].blocks
    assert program.functions[0].registers == library.entries[0].declaration.registers
    with pytest.raises(DecomposeNotImplementedError):
        source.decompose_bloq()
    # No executable body exists at the unchanged source root.
    with pytest.raises(ValueError, match="bodies"):
        bind_register_program(program, ())


@pytest.mark.parametrize(
    "kind,direction_key", (("Square", "uncompute"), ("PlusEqualProduct", "is_adjoint"))
)
@pytest.mark.parametrize("direction", (False, True))
def test_direction_changes_only_the_original_output_ownership(
    kind: str, direction_key: str, direction: bool
) -> None:
    arguments: dict[str, Any] = (
        {"bitsize": 2}
        if kind == "Square"
        else {"a_bitsize": 1, "b_bitsize": 2, "result_bitsize": 3}
    )
    arguments[direction_key] = direction
    source = getattr(import_module(_PREFIX[:-1]), kind)(**arguments)
    program = resolve_component_program(
        _library(source, 0), 0, product_model_factories()
    )
    registers = program.functions[0].registers
    assert registers is not None
    expected = (
        RegisterSide.THRU
        if kind == "PlusEqualProduct"
        else RegisterSide.LEFT if direction else RegisterSide.RIGHT
    )
    assert registers[-1].side is expected


@pytest.mark.parametrize(
    "kind,arguments",
    (
        ("Square", {"bitsize": sympy.Symbol("n", positive=True, integer=True)}),
        (
            "SumOfSquares",
            {"bitsize": sympy.Symbol("n", positive=True, integer=True), "k": 3},
        ),
        (
            "Product",
            {
                "a_bitsize": sympy.Symbol("n", positive=True, integer=True),
                "b_bitsize": sympy.Symbol("n", positive=True, integer=True),
            },
        ),
        (
            "PlusEqualProduct",
            {
                "a_bitsize": 1,
                "b_bitsize": 2,
                "result_bitsize": sympy.Symbol("r", positive=True, integer=True),
            },
        ),
        (
            "ScaleIntByReal",
            {
                "r_bitsize": sympy.Symbol("r", positive=True, integer=True),
                "i_bitsize": 2,
            },
        ),
        (
            "MultiplyTwoReals",
            {"bitsize": sympy.Symbol("n", positive=True, integer=True)},
        ),
        (
            "SquareRealNumber",
            {"bitsize": sympy.Symbol("n", positive=True, integer=True)},
        ),
        (
            "InvertRealNumber",
            {"bitsize": sympy.Symbol("n", positive=True, integer=True), "num_frac": 2},
        ),
    ),
)
def test_symbolic_declarations_keep_original_width_and_format(
    kind: str, arguments: dict[str, Any]
) -> None:
    source = getattr(import_module(_PREFIX[:-1]), kind)(**arguments)
    if kind == "Product":
        from qualtran.serialization.bloq import bloqs_from_proto, bloqs_to_proto

        restored = bloqs_from_proto(bloqs_to_proto(source, max_depth=0))[0]
        # Wire text loses the positive/real assumptions used by Python max.
        with pytest.raises(TypeError):
            _ = restored.signature
        with pytest.raises(TypeError):
            resolve_component_program(_library(source, 0), 0, product_model_factories())
        return
    program = resolve_component_program(
        _library(source, 0), 0, product_model_factories()
    )
    assert verify_program_record(program) == ()
    assert not program.functions[0].blocks


def test_original_symbolic_shape_comparison_failures_and_invalid_arguments() -> None:
    module = import_module(_PREFIX[:-1])
    n, m = sympy.symbols("n m", integer=True, positive=True)
    with pytest.raises(AttributeError, match="bit_length"):
        _ = module.SumOfSquares(n, m).signature
    with pytest.raises(TypeError):
        _ = module.Product(n, m).signature
    factories = product_model_factories()
    with pytest.raises(SerializationError, match="concrete k"):
        factories[_PREFIX + "SumOfSquares"]("bad", None, {"bitsize": n, "k": m})
    with pytest.raises(TypeError):
        factories[_PREFIX + "Product"]("bad", None, {"a_bitsize": n, "b_bitsize": m})
    cases: tuple[tuple[str, dict[str, Any]], ...] = (
        ("Square", {}),
        ("Square", {"bitsize": 2, "extra": 1}),
        ("Square", {"bitsize": 2, "uncompute": 2}),
        ("PlusEqualProduct", {"a_bitsize": 2, "b_bitsize": 2, "result_bitsize": 3}),
        ("SquareRealNumber", {"bitsize": 2}),
        ("InvertRealNumber", {"bitsize": 3, "num_frac": 3}),
    )
    for kind, arguments in cases:
        with pytest.raises(SerializationError):
            factories[_PREFIX + kind]("bad", None, arguments)
    declaration = factories[_PREFIX + "Square"]("square", None, {"bitsize": 2})
    assert isinstance(declaration, FunctionRecord)
    with pytest.raises(SerializationError, match="signature"):
        factories[_PREFIX + "Square"]("square", (), {"bitsize": 2})

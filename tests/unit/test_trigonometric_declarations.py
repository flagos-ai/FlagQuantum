"""Generic trigonometric declarations preserve frozen constructors and models."""

import pytest
import sympy

from flagquantum.compiler._component_trace.call_graph import SymbolAllocator
from flagquantum.compiler.registers.fixed_point.arcsine_declaration import (
    ArcsineDeclaration,
)
from flagquantum.compiler.registers.lookup.arctangent_declaration import (
    ArctangentDeclaration,
)
from tests.unit.test_subtraction_foreign_file import native_signature

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize(
    "width,fraction",
    ((0, 0), (0, -1), (0, -2), (1, 0), (1, 1), (4, -2), (4, 0), (4, 4), (10, 7)),
)
@pytest.mark.parametrize("iterations,degree", ((4, 4), (3, 6), (0, 0)))
def test_complete_arcsine_fields_signature_and_model(
    width: int, fraction: int, iterations: int, degree: int
) -> None:
    from qualtran.bloqs.arithmetic.trigonometric.arcsin import ArcSin
    from qualtran.bloqs.basic_gates import Toffoli

    source, native = ArcSin(width, fraction, iterations, degree), ArcsineDeclaration(
        width, fraction, iterations, degree
    )
    assert native.signature == native_signature(source.signature)
    assert native.direct_calls() == {"toffoli": source.bloq_counts()[Toffoli()]}
    assert (
        ArcsineDeclaration(width, fraction).num_iters
        == source.__class__(width, fraction).num_iters
        == 4
    )
    assert ArcsineDeclaration(width, fraction).degree == 4


def test_arcsine_symbolic_validation_is_deferred_like_original() -> None:
    from qualtran.bloqs.arithmetic.trigonometric.arcsin import ArcSin
    from qualtran.bloqs.basic_gates import Toffoli

    n, f, m, d = sympy.symbols("n f m d")
    source, native = ArcSin(n, f, m, d), ArcsineDeclaration(n, f, m, d)
    assert native.signature == native_signature(source.signature)
    assert native.direct_calls() == {"toffoli": source.bloq_counts()[Toffoli()]}
    # Equality is permitted despite the original error text claiming strict <.
    assert ArcsineDeclaration(4, 4).signature == native_signature(
        ArcSin(4, 4).signature
    )
    for constructor in (ArcSin, ArcsineDeclaration):
        with pytest.raises(ValueError):
            constructor(4, 5)


@pytest.mark.parametrize("left,right", ((1, 1), (1, 2), (3, 5), (4, 6), (-1, 2)))
def test_arctangent_dimensions_self_adjoint_and_fresh_model_symbols(
    left: int, right: int
) -> None:
    from qualtran.bloqs.arithmetic.trigonometric.arctan import ArcTan
    from qualtran.resource_counting import SympySymbolAllocator

    source, native = ArcTan(left, right), ArctangentDeclaration(left, right)
    assert native.register_dimensions() == source.registers()
    assert native.adjoint() is native and source.adjoint() is source
    if left >= 0:
        assert native.signature == native_signature(source.signature)
    symbols, original_symbols = SymbolAllocator(), SympySymbolAllocator()
    for i in range(3):
        actual, original = native.direct_calls(symbols), source.build_call_graph(
            original_symbols
        )
        assert (
            list(actual.values()) == list(original.values()) == [sympy.Symbol(f"_c{i}")]
        )
        child = next(iter(original))
        assert next(iter(actual)) == (
            "plus_equal_product",
            child.a_bitsize,
            child.b_bitsize,
            child.result_bitsize,
        )


def test_arctangent_symbolic_registers_and_original_rebinding_failure() -> None:
    from qualtran.bloqs.arithmetic.trigonometric.arctan import ArcTan

    n, p = sympy.symbols("n p")
    for left, right in ((n, 3), (4, p), (n, p), (sympy.Integer(2), 3)):
        source, native = ArcTan(left, right), ArctangentDeclaration(left, right)
        assert native.signature == native_signature(source.signature)
        for getter in (source.registers, native.register_dimensions):
            with pytest.raises(TypeError):
                getter()
        assert next(iter(native.direct_calls())) == (
            "plus_equal_product",
            right,
            right,
            2 * right,
        )
    with pytest.raises(NotImplementedError):
        ArcTan(2, 3).with_registers((2,), (2,), (2,))

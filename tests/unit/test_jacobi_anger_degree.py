"""Source threshold visitation and symbolic degree formulas remain separate."""

import numpy as np
import pytest
import sympy
from scipy import special

from flagquantum.simulation.numerics.jacobi_anger_degree import (
    degree_jacobi_anger_approximation,
)

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize("t", [-3.0, 0.0, 2.0, 3.0, 5.0, 10.0])
@pytest.mark.parametrize("precision", [1e-5, 1e-7, 1e-10])
def test_degree_and_every_bessel_call_match_source(
    t: float, precision: float, monkeypatch: pytest.MonkeyPatch
) -> None:
    from qualtran.linalg.polynomial.jacobi_anger_approximations import (
        degree_jacobi_anger_approximation as source,
    )

    jv = special.jv
    original_calls: list[tuple[int, float]] = []
    native_calls: list[tuple[int, float]] = []

    def original(order: int, argument: float) -> float:
        original_calls.append((order, argument))
        return float(jv(order, argument))

    def native(order: int, argument: float) -> float:
        native_calls.append((order, argument))
        return float(jv(order, argument))

    monkeypatch.setattr(special, "jv", original)
    expected = source(t, precision=precision)
    result = degree_jacobi_anger_approximation(t, precision=precision, bessel_jn=native)
    assert type(result) is int and result == expected
    assert native_calls == original_calls
    assert not np.isclose(jv(result, t), 0, atol=precision)
    assert np.isclose(jv(result + 1, t), 0, atol=precision)


@pytest.mark.parametrize("parameters", [(1.0, 1.0)])
def test_source_assertion_boundary_is_retained(parameters: tuple[float, float]) -> None:
    from qualtran.linalg.polynomial.jacobi_anger_approximations import (
        degree_jacobi_anger_approximation as source,
    )

    t, precision = parameters
    with pytest.raises(AssertionError):
        source(t, precision=precision)
    with pytest.raises(AssertionError):
        degree_jacobi_anger_approximation(t, precision=precision, bessel_jn=special.jv)


@pytest.mark.parametrize("case", [0, 1, 2, 3])
def test_symbolic_formula_and_positive_constant_match_source(case: int) -> None:
    from qualtran.linalg.polynomial.jacobi_anger_approximations import (
        degree_jacobi_anger_approximation as source,
    )

    t = sympy.Symbol("t", real=True)
    epsilon = sympy.Symbol("epsilon", positive=True)
    parameters = [
        (t, epsilon),
        (2, epsilon),
        (t, 1e-5),
        (sympy.Integer(2), sympy.Rational(1, 100)),
    ]
    argument, precision = parameters[case]
    actual = degree_jacobi_anger_approximation(argument, precision=precision)
    expected = source(argument, precision=precision)
    assert actual == expected
    assert isinstance(actual, sympy.Expr)
    constant = sympy.Symbol("C_{JA}", positive=True)
    assert constant in actual.free_symbols
    assert constant.is_positive is True


def test_default_concrete_search_matches_source() -> None:
    from qualtran.linalg.polynomial.jacobi_anger_approximations import (
        degree_jacobi_anger_approximation as source,
    )

    assert degree_jacobi_anger_approximation(2, precision=1e-5) == source(
        2, precision=1e-5
    )


def test_scalar_provider_replacement_and_errors() -> None:
    calls: dict[int, float] = {}

    def record(order: int, t: float) -> float:
        calls[order] = float(special.jv(order, t))
        return calls[order]

    result = degree_jacobi_anger_approximation(3, precision=1e-7, bessel_jn=record)

    def cached(order: int, t: float) -> float:
        assert t == 3
        return calls[order]

    assert (
        degree_jacobi_anger_approximation(3, precision=1e-7, bessel_jn=cached) == result
    )

    def refuse(order: int, t: float) -> float:
        raise ArithmeticError("test scalar Bessel failure")

    with pytest.raises(ArithmeticError, match="test scalar Bessel failure"):
        degree_jacobi_anger_approximation(3, precision=1e-7, bessel_jn=refuse)

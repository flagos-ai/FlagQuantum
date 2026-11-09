"""Compare real and optional symbolic resource arithmetic with the source."""

import pytest

from flagquantum.compiler._component_trace.symbolic_math import (
    bit_length,
    ceil,
    floor,
    log2,
    smax,
    smin,
)

pytestmark = pytest.mark.unit


@pytest.mark.ftqc_reference
def test_symbolic_math_matches_locked_expressions_and_substitutions() -> None:
    pytest.importorskip("qualtran")
    sympy = pytest.importorskip("sympy")
    from qualtran import symbolics as source

    n = sympy.Symbol("n", positive=True, integer=True)
    x = sympy.Symbol("x", nonnegative=True)
    comparisons = (
        (log2(n), source.log2(n)),
        (ceil(x), source.ceil(x)),
        (floor(x), source.floor(x)),
        (bit_length(n - 1), source.bit_length(n - 1)),
        (bit_length(x), source.bit_length(x)),
        (smax(3, n, x), source.smax(3, n, x)),
        (smin(iter((3, n, x))), source.smin(iter((3, n, x)))),
    )
    for actual, expected in comparisons:
        assert actual == expected
        assert actual.subs({n: 17, x: sympy.Rational(7, 2)}) == expected.subs(
            {n: 17, x: sympy.Rational(7, 2)}
        )


def test_concrete_math_and_empty_extrema() -> None:
    for i in range(1024):
        assert bit_length(i) == bit_length(i + 0.5) == i.bit_length()
    assert ceil(-1.25) == -1
    assert floor(-1.25) == -2
    assert log2(2**10000) == 10000
    assert smax((1, 2)) == 2
    assert smin(3) == 3
    with pytest.raises(ValueError, match="at least 1 argument"):
        smax()
    with pytest.raises(ValueError, match="empty sequence"):
        smin(iter(()))

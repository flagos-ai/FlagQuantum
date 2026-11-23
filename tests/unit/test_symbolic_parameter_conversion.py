"""Share scalar interchange conversion across library and SDK consumers."""

from math import isclose
from typing import Any

import pytest
import sympy
import torch

from flagquantum.compiler.formats.symbolic_parameters import symbolic_parameter_value
from flagquantum.core.parameters import Parameter, ParameterExpression
from flagquantum.ecosystem.cirq.conversion import _flagquantum_parameter_value

pytestmark = pytest.mark.unit


def test_shared_converter_preserves_all_supported_nodes_and_sdk_consumer() -> None:
    cirq = pytest.importorskip("cirq")

    t = sympy.Symbol("theta")
    expressions = (
        t,
        sympy.pi,
        sympy.E,
        sympy.Rational(2, 3),
        t + 2,
        3 * t,
        1 / t,
        t ** sympy.Rational(1, 2),
        sympy.sin(t),
        sympy.cos(t),
        sympy.exp(t),
        sympy.log(t),
        sympy.sin(t) + sympy.log(t + 1) / 3,
    )
    for expression in expressions:
        value = symbolic_parameter_value(expression)
        assert value == _flagquantum_parameter_value(cirq, expression)
        actual = (
            value.bind({"theta": 0.37})
            if isinstance(value, (Parameter, ParameterExpression))
            else value
        )
        assert isclose(
            actual, float(expression.subs(t, 0.37)), rel_tol=0, abs_tol=2e-15
        )
        scaled = _flagquantum_parameter_value(cirq, expression, scale_by_pi=True)
        expected = symbolic_parameter_value(expression * sympy.pi)
        assert scaled == expected
    value = symbolic_parameter_value(sympy.sin(t) + sympy.exp(t))
    assert isinstance(value, ParameterExpression)
    angle = torch.tensor(0.37, dtype=torch.float64, requires_grad=True)
    torch.testing.assert_close(
        torch.autograd.grad(value.bind({"theta": angle}), angle)[0],
        torch.cos(angle) + torch.exp(angle),
        rtol=0,
        atol=2e-15,
    )


@pytest.mark.parametrize(
    "expression", (sympy.oo, sympy.nan, sympy.I, sympy.tan(sympy.Symbol("x")))
)
def test_invalid_expression_retains_explicit_failure(expression: Any) -> None:
    with pytest.raises(ValueError):
        symbolic_parameter_value(expression)


def test_symbol_identity_and_sdk_discovery_remain_checked() -> None:
    cirq = pytest.importorskip("cirq")

    expression = sympy.Symbol("t") + sympy.Symbol("t", positive=True)
    with pytest.raises(ValueError, match="distinct symbols"):
        symbolic_parameter_value(expression)
    with pytest.raises(ValueError, match="distinct Cirq symbols"):
        _flagquantum_parameter_value(cirq, expression)

    class MissingSymbols:
        @staticmethod
        def parameter_symbols(value: Any) -> set[str]:
            return set()

    with pytest.raises(ValueError, match="symbol discovery disagree"):
        _flagquantum_parameter_value(MissingSymbols(), sympy.Symbol("t"))

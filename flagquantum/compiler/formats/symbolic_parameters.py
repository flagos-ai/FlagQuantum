"""Translate symbolic scalar interchange into existing owned parameter nodes."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

from ...core.parameters import Parameter, ParameterExpression

if TYPE_CHECKING:
    import sympy


def symbolic_parameter_value(
    symbolic: sympy.Expr,
) -> float | Parameter | ParameterExpression:
    """Translate supported real expressions into owned arithmetic nodes.

    Distinct symbolic identities must have distinct owned parameter names.
    Unsupported functions and nonfinite or complex constants fail at conversion.
    Numerical domains of unresolved nodes remain checked by Core binding.
    """
    import sympy

    symbols = tuple(symbolic.free_symbols)
    parameters_by_name = {str(symbol): Parameter(str(symbol)) for symbol in symbols}
    if len(parameters_by_name) != len(symbols):
        raise ValueError("distinct symbols share the same name")
    parameters_by_symbol = {
        symbol: parameters_by_name[str(symbol)] for symbol in symbols
    }

    def convert(
        node: Any, *, exact_integers: bool = False
    ) -> float | Parameter | ParameterExpression:
        if isinstance(node, sympy.Symbol):
            try:
                return parameters_by_symbol[node]
            except KeyError as exc:
                raise ValueError(f"unknown symbol {node!s}") from exc
        if isinstance(node, (sympy.Number, sympy.NumberSymbol)):
            if exact_integers and isinstance(node, sympy.Integer):
                return int(node)
            converted = complex(node)
            if converted.imag != 0 or not math.isfinite(converted.real):
                raise ValueError(f"constant {node!s} is not a finite real scalar")
            return float(converted.real)
        if isinstance(node, sympy.Add):
            args = tuple(
                convert(arg, exact_integers=exact_integers) for arg in node.args
            )
            if not args:
                return 0.0
            result = args[0]
            for arg in args[1:]:
                result = result + arg
            return result
        if isinstance(node, sympy.Mul):
            args = tuple(
                convert(arg, exact_integers=exact_integers) for arg in node.args
            )
            if not args:
                return 1.0
            result = args[0]
            for arg in args[1:]:
                result = result * arg
            return result
        if isinstance(node, sympy.Pow):
            base, exponent = node.args
            if exponent == -1:
                return ParameterExpression(
                    "div",
                    (
                        1 if exact_integers else 1.0,
                        convert(base, exact_integers=exact_integers),
                    ),
                )
            return ParameterExpression(
                "pow",
                (
                    convert(base, exact_integers=exact_integers),
                    convert(exponent, exact_integers=exact_integers),
                ),
            )
        functions = {
            sympy.sin: "sin",
            sympy.cos: "cos",
            sympy.exp: "exp",
            sympy.log: "log",
            sympy.acos: "acos",
            sympy.floor: "floor",
        }
        if node.func in functions and len(node.args) == 1:
            # New discrete/domain-limited trees must retain integer coefficients.
            # Old supported expressions keep their byte-identical float leaves.
            return ParameterExpression(
                functions[node.func],
                (
                    convert(
                        node.args[0],
                        exact_integers=exact_integers
                        or node.func in {sympy.acos, sympy.floor},
                    ),
                ),
            )
        function_name = getattr(getattr(node, "func", None), "__name__", None)
        raise ValueError(
            f"operation {function_name or type(node).__name__!r} is unsupported"
        )

    return convert(symbolic)

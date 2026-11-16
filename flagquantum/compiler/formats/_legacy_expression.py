"""Decode an explicit arithmetic grammar for legacy symbolic wire values."""

from __future__ import annotations

import ast
import operator
from typing import TYPE_CHECKING

from ...errors import SerializationError

if TYPE_CHECKING:
    from sympy import Expr

_BINARY = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
}
_CONSTANTS = {"E", "I", "pi", "oo", "nan", "zoo"}
_FUNCTIONS = {
    "Abs",
    "Max",
    "Min",
    "ceiling",
    "floor",
    "sqrt",
    "exp",
    "log",
    "sin",
    "cos",
    "tan",
    "asin",
    "acos",
    "atan",
    "sign",
}
_CURVE_FUNCTIONS = {"ec_times_x", "ec_times_y"}
_MAX_EXPRESSION_LENGTH = 4096
_MAX_INTEGER_BITS = 16384


def _check_integer_size(value: Expr) -> None:
    from sympy import Rational

    if (
        isinstance(value, Rational)
        and max(value.p.bit_length(), value.q.bit_length()) > _MAX_INTEGER_BITS
    ):
        raise SerializationError("Legacy symbolic integer exceeds the decode budget")


def parse_legacy_expression(text: str) -> Expr:
    """Build trusted SymPy nodes from syntax without evaluating Python calls."""
    import sympy
    from sympy import Float, Integer, Symbol

    if len(text) > _MAX_EXPRESSION_LENGTH:
        raise SerializationError("Legacy symbolic expression exceeds the decode budget")

    def evaluate(node: ast.expr) -> Expr:
        if isinstance(node, ast.Constant):
            if type(node.value) is int:
                return Integer(node.value)
            if type(node.value) is float:
                return Float(ast.get_source_segment(text, node) or str(node.value))
        elif isinstance(node, ast.Name) and not node.id.startswith("_"):
            return getattr(sympy, node.id) if node.id in _CONSTANTS else Symbol(node.id)
        elif isinstance(node, ast.UnaryOp) and isinstance(
            node.op, (ast.UAdd, ast.USub)
        ):
            value = evaluate(node.operand)
            return value if isinstance(node.op, ast.UAdd) else -value
        elif isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
            left, right = evaluate(node.left), evaluate(node.right)
            if (
                isinstance(node.op, ast.Pow)
                and right.is_Rational
                and (
                    abs(right) > _MAX_INTEGER_BITS
                    or left.is_Rational
                    and abs(right) > 1
                    and abs(right) * max(left.p.bit_length(), left.q.bit_length())
                    > _MAX_INTEGER_BITS
                )
            ):
                raise SerializationError(
                    "Legacy symbolic exponent exceeds the decode budget"
                )
            result = _BINARY[type(node.op)](left, right)
            _check_integer_size(result)
            return result
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and not node.keywords
        ):
            name = node.func.id
            if name in _CURVE_FUNCTIONS and len(node.args) == 2:
                return sympy.Function(name)(*(evaluate(arg) for arg in node.args))
            if name in _FUNCTIONS:
                result = getattr(sympy, name)(*(evaluate(arg) for arg in node.args))
                if isinstance(result, sympy.Expr):
                    _check_integer_size(result)
                    return result
        raise SerializationError(
            f"Unsupported legacy symbolic syntax: {type(node).__name__}"
        )

    try:
        return evaluate(ast.parse(text, mode="eval").body)
    except (SyntaxError, TypeError, ValueError, RecursionError, OverflowError) as error:
        if isinstance(error, SerializationError):
            raise
        raise SerializationError("Invalid legacy symbolic expression") from error

"""Decode the arithmetic and piecewise grammar for legacy symbolic values."""

from __future__ import annotations

import ast
import operator
import re
from typing import TYPE_CHECKING

from ...errors import SerializationError

if TYPE_CHECKING:
    from sympy import Expr
    from sympy.logic.boolalg import Boolean

_BINARY = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
}
_RELATIONS = {
    ast.Eq: "Eq",
    ast.NotEq: "Ne",
    ast.Lt: "Lt",
    ast.LtE: "Le",
    ast.Gt: "Gt",
    ast.GtE: "Ge",
}
_BOOLEAN_INFIX = {ast.BitAnd: "And", ast.BitOr: "Or", ast.BitXor: "Xor"}
_BOOLEAN_CALLS = {"And", "Or", "Xor", "Not", "Implies", "Equivalent", "ITE"}
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
    "Mod",
    "KroneckerDelta",
    "binomial",
    "factorial",
}
_CURVE_FUNCTIONS = {"ec_times_x", "ec_times_y"}
_MAX_EXPRESSION_LENGTH = 4096
_MAX_INTEGER_BITS = 16384
_TEXT_SYMBOL = re.compile(
    r"(?<![\w\\])(?:[A-Za-z]\w*_)?\\text\{[A-Za-z][A-Za-z0-9_]*\}(?![\w\\{])"
)


def _check_integer_size(value: Expr) -> None:
    from sympy import Rational

    if (
        isinstance(value, Rational)
        and max(value.p.bit_length(), value.q.bit_length()) > _MAX_INTEGER_BITS
    ):
        raise SerializationError("Legacy symbolic integer exceeds the decode budget")


def _check_power_budget(left: Expr | int | float, right: Expr | int | float) -> None:
    from sympy import Rational

    if isinstance(right, (int, Rational)) and (
        abs(right) > _MAX_INTEGER_BITS
        or isinstance(left, (int, Rational))
        and abs(right) > 1
        and abs(right) * max(left.numerator.bit_length(), left.denominator.bit_length())
        > _MAX_INTEGER_BITS
    ):
        raise SerializationError("Legacy symbolic exponent exceeds the decode budget")


def _check_combinatorial_budget(name: str, values: tuple[Expr, ...]) -> None:
    from sympy import Integer, Rational

    if name == "factorial" and len(values) == 1:
        value = values[0]
        if isinstance(value, Integer) and value > _MAX_INTEGER_BITS:
            raise SerializationError(
                "Legacy symbolic factorial exceeds the decode budget"
            )
    if name == "binomial" and len(values) == 2:
        n, k = values
        if isinstance(k, Integer) and k > 0:
            if isinstance(n, Integer) and n >= 0:
                if k > n:
                    return
                k = min(k, n - k)
            if k > _MAX_INTEGER_BITS or (
                isinstance(n, Rational)
                and k * max(n.p.bit_length(), n.q.bit_length()) > 4 * _MAX_INTEGER_BITS
            ):
                raise SerializationError(
                    "Legacy symbolic binomial exceeds the decode budget"
                )


def parse_legacy_expression(text: str) -> Expr:
    """Build trusted SymPy nodes from syntax without evaluating Python calls."""
    import sympy
    from sympy import Expr, Float, Integer, Symbol

    if len(text) > _MAX_EXPRESSION_LENGTH:
        raise SerializationError("Legacy symbolic expression exceeds the decode budget")
    # Source signature widths print these atomic symbol names as TeX text,
    # while constructor arguments carry the same names in structured terms.
    # Replace only that lexical atom; the existing AST whitelist still owns
    # all operators and calls. No Python or general TeX parser is evaluated.
    symbols: dict[str, str] = {}
    prefix = "fq_text_symbol_"
    while prefix in text:
        prefix += "x"

    def text_symbol(match: re.Match[str]) -> str:
        name = f"{prefix}{len(symbols)}"
        symbols[name] = match.group()
        return name

    text = _TEXT_SYMBOL.sub(text_symbol, text)

    def condition(node: ast.expr) -> Boolean:
        if isinstance(node, ast.Constant) and type(node.value) is bool:
            return sympy.true if node.value else sympy.false
        if isinstance(node, ast.Name) and not node.id.startswith("__"):
            value = evaluate(node)
            if isinstance(value, Symbol):
                return value
        if isinstance(node, ast.Compare) and len(node.ops) == 1:
            relation = _RELATIONS.get(type(node.ops[0]))
            if relation is not None:
                return getattr(sympy, relation)(
                    evaluate(node.left), evaluate(node.comparators[0])
                )
        if isinstance(node, ast.BinOp) and type(node.op) in _BOOLEAN_INFIX:
            operation = _BOOLEAN_INFIX[type(node.op)]
            return getattr(sympy, operation)(
                condition(node.left), condition(node.right)
            )
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Invert):
            return sympy.Not(condition(node.operand))
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and not node.keywords
        ):
            name = node.func.id
            if name in _RELATIONS.values() and len(node.args) == 2:
                return getattr(sympy, name)(*(evaluate(arg) for arg in node.args))
            if name in _BOOLEAN_CALLS:
                return getattr(sympy, name)(*(condition(arg) for arg in node.args))
        raise SerializationError(
            f"Unsupported legacy symbolic predicate: {type(node).__name__}"
        )

    def factors(node: ast.expr) -> list[Expr]:
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
            return [*factors(node.left), *factors(node.right)]
        return [evaluate(node)]

    def evaluate(node: ast.expr) -> Expr:
        if isinstance(node, ast.Constant):
            if type(node.value) is int:
                literal = Integer(node.value)
                _check_integer_size(literal)
                return literal
            if type(node.value) is float:
                return Float(ast.get_source_segment(text, node) or str(node.value))
        elif isinstance(node, ast.Name) and not node.id.startswith("__"):
            if node.id in symbols:
                return Symbol(symbols[node.id])
            return getattr(sympy, node.id) if node.id in _CONSTANTS else Symbol(node.id)
        elif isinstance(node, ast.UnaryOp) and isinstance(
            node.op, (ast.UAdd, ast.USub)
        ):
            value = evaluate(node.operand)
            return value if isinstance(node.op, ast.UAdd) else -value
        elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
            # Construct one product so intermediate scalar multiplication cannot
            # distribute into a sum and change the source's printed wire value.
            result = sympy.Mul(*factors(node))
            _check_integer_size(result)
            return result
        elif isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
            left, right = evaluate(node.left), evaluate(node.right)
            if isinstance(node.op, ast.Pow):
                _check_power_budget(left, right)
            result = _BINARY[type(node.op)](left, right)
            _check_integer_size(result)
            return result
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and not node.keywords
        ):
            name = node.func.id
            if name == "log2" and len(node.args) == 1:
                from sympy.codegen.cfunctions import log2

                return log2(evaluate(node.args[0]))
            if name in _CURVE_FUNCTIONS and len(node.args) == 2:
                return sympy.Function(name)(*(evaluate(arg) for arg in node.args))
            if name == "Piecewise" and node.args:
                branches = []
                for arg in node.args:
                    if not isinstance(arg, ast.Tuple) or len(arg.elts) != 2:
                        raise SerializationError("Unsupported legacy Piecewise branch")
                    branches.append((evaluate(arg.elts[0]), condition(arg.elts[1])))
                piecewise = sympy.Piecewise(*branches)
                if isinstance(piecewise, Expr):
                    return piecewise
            if name in _FUNCTIONS:
                values = tuple(evaluate(arg) for arg in node.args)
                _check_combinatorial_budget(name, values)
                result = getattr(sympy, name)(*values)
                if isinstance(result, sympy.Expr):
                    _check_integer_size(result)
                    return result
        raise SerializationError(
            f"Unsupported legacy symbolic syntax: {type(node).__name__}"
        )

    try:
        return evaluate(ast.parse(text, mode="eval").body)
    except (
        SyntaxError,
        TypeError,
        ValueError,
        RecursionError,
        ArithmeticError,
    ) as error:
        if isinstance(error, SerializationError):
            raise
        raise SerializationError("Invalid legacy symbolic expression") from error

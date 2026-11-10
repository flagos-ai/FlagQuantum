"""Independent Python-AST consumer of the shared bounded fixed-point domain."""

from __future__ import annotations

import ast
from fractions import Fraction
from typing import NoReturn

import pytest

from flagquantum.compiler._construction.fixed_point import (
    FixedPointExpression,
    FixedPointScalar,
    fixed_point_number,
    fixed_point_scalar,
)


def _fail(code: str, message: str) -> NoReturn:
    raise ValueError(f"{code}: {message}")


def _reject(node: ast.AST, code: str, message: str) -> NoReturn:
    raise ValueError(f"line {getattr(node, 'lineno', 0)} {code}: {message}")


def _text(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.BinOp):
        return {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Pow: "**"}.get(
            type(node.op), "unsupported"
        )
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return "-"
    return "unsupported"


def _children(node: ast.AST) -> tuple[ast.AST, ...]:
    if isinstance(node, ast.BinOp):
        return node.left, node.right
    if isinstance(node, ast.UnaryOp):
        return (node.operand,)
    return ()


def _literal(node: ast.AST) -> Fraction | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return Fraction(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        value = _literal(node.operand)
        return None if value is None else -value
    return None


_EVALUATOR: FixedPointExpression[ast.AST] = FixedPointExpression(
    text=_text, children=_children, literal=_literal, reject=_reject
)


def _evaluate(source: str, **values: FixedPointScalar) -> FixedPointScalar:
    return _EVALUATOR.evaluate(ast.parse(source, mode="eval").body, values)


@pytest.mark.parametrize(
    ("literal", "expected"),
    [
        (Fraction(1, 512), Fraction(0)),
        (Fraction(-1, 512), Fraction(-1, 256)),
        (Fraction(3, 512), Fraction(1, 256)),
        (Fraction(-3, 512), Fraction(-1, 128)),
        (Fraction(1, 256) - Fraction(1, 2**22), Fraction(0)),
        (-Fraction(1, 256) - Fraction(1, 2**22), -Fraction(1, 256)),
    ],
)
def test_literal_staging_and_signed_rounding(
    literal: Fraction, expected: Fraction
) -> None:
    result = fixed_point_number(_fail, literal)
    assert result.low == result.high == expected
    assert result.values == (expected,) * 32


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("0.5 + 0.125", Fraction(5, 8)),
        ("0.5 - 0.125", Fraction(3, 8)),
        ("0.101 * -0.5", Fraction(-13, 256)),
        ("0.101 ** 2", Fraction(1, 128)),
        ("-x", Fraction(-1, 2)),
        ("x ** 3", Fraction(1, 8)),
        ("x ** 4", Fraction(1, 16)),
    ],
)
def test_ast_consumer_arithmetic(source: str, expected: Fraction) -> None:
    result = _evaluate(source, x=fixed_point_number(_fail, Fraction(1, 2)))
    assert result.values == (expected,) * 32


def test_signed_interval_and_correlated_branch_values() -> None:
    values = tuple(Fraction(row % 3 - 1, 2) for row in range(32))
    x = fixed_point_scalar(_fail, Fraction(-1, 2), Fraction(1, 2), 1, values)
    assert (x.width, x.signed, x.fraction) == (2, True, 1)
    square = _evaluate("x ** 2", x=x)
    assert (square.low, square.high) == (Fraction(0), Fraction(1, 4))
    assert square.values == tuple(value * value for value in values)
    zero = _evaluate("x - x", x=x)
    assert zero.values == (Fraction(0),) * 32
    assert (zero.low, zero.high) == (Fraction(-1), Fraction(1))


@pytest.mark.parametrize(
    "source", ["x ** 0", "x ** 5", "x ** 1.5", "x ** x", "missing", "f()"]
)
def test_ast_refusals_retain_source_location(source: str) -> None:
    with pytest.raises(ValueError, match="line 1 precision"):
        _evaluate(source, x=fixed_point_number(_fail, Fraction(1, 2)))


@pytest.mark.parametrize(
    ("low", "high", "fraction", "values", "reason"),
    [
        (0, 2**20, 0, (Fraction(0),) * 32, "twenty bits"),
        (0, 1, 0, (Fraction(1, 2),) * 32, "inferred format"),
        (0, 1, 0, (Fraction(2),) * 32, "inferred format"),
        (0, 1, 0, (Fraction(0),) * 64, "branch rows"),
        (0, 1, 0, (), "branch rows"),
        (1, 0, 0, (Fraction(0),) * 32, "invalid quantized"),
        (0, 1, -1, (Fraction(0),) * 32, "invalid quantized"),
    ],
)
def test_domain_and_budget_refusals(
    low: int, high: int, fraction: int, values: tuple[Fraction, ...], reason: str
) -> None:
    with pytest.raises(ValueError, match=reason):
        fixed_point_scalar(_fail, Fraction(low), Fraction(high), fraction, values)


@pytest.mark.parametrize("coefficient", ["0.450158158078553", "0.431573"])
def test_python_ast_brownian_recurrence_against_integer_oracle(
    coefficient: str,
) -> None:
    domains = {
        name: fixed_point_scalar(
            _fail,
            Fraction(0),
            Fraction(1),
            0,
            tuple(Fraction(row >> bit & 1) for row in range(32)),
        )
        for bit, name in enumerate(("t", "a0", "a1", "a2", "a3"))
    }
    domains["u1"] = _evaluate("-6.283185307179586 * (t - 0.5)", **domains)
    domains["u2"] = _evaluate("u1 * u1 - 1", **domains)
    domains["u3"] = _evaluate("u1 * u2 - u1", **domains)
    domains["sine"] = _evaluate("-5 * (t - 0.5)**2 + 1", **domains)
    domains["b"] = _evaluate(
        f"a0*t + ({coefficient}*sine) * (a1*u1 + a2*u2 + a3*u3)", **domains
    )
    price = _evaluate("2*b - t + 1", **domains)
    # Independent integer recurrence at scale 256; no expression evaluator,
    # scalar inference or quantization helper is used by this oracle.
    cosine = int(-6.283185307179586 * 256) - 1
    weight = int(float(coefficient) * 256)
    expected = []
    for row in range(32):
        t, a0, a1, a2, a3 = (row >> bit & 1 for bit in range(5))
        u1 = cosine * (2 * t - 1) // 2
        u2 = u1 * u1 // 256 - 256
        u3 = u1 * u2 // 256 - u1
        b = (
            a0 * t * 256
            + ((weight * -64) // 256) * (a1 * u1 + a2 * u2 + a3 * u3) // 256
        )
        expected.append(Fraction(2 * b - t * 256 + 256, 256))
    assert price.values == tuple(expected)
    assert (price.width, price.signed, price.fraction) == (15, True, 8)

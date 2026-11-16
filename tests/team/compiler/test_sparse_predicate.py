"""Sparse predicate truth evaluation is bounded and independent of phase ANF."""

import ast
from typing import NoReturn

import pytest

from flagquantum.compiler._construction.profile import uint_type
from flagquantum.compiler._hybrid.model import Value, ValueId
from flagquantum.compiler._sparse_predicate import sparse_predicate

pytestmark = pytest.mark.unit


def evaluate(text: str, widths: dict[str, int]) -> tuple[tuple[int, ...], int]:
    expression = ast.parse(text, mode="eval").body
    ports = {
        name: Value(ValueId("predicate", i), uint_type(width))
        for i, (name, width) in enumerate(widths.items())
    }

    def word(node: ast.expr) -> tuple[tuple[Value, int], ...] | None:
        if isinstance(node, ast.Name) and node.id in ports:
            return tuple((ports[node.id], bit) for bit in range(widths[node.id]))
        return None

    def number(node: ast.expr) -> float:
        if isinstance(node, ast.Constant) and type(node.value) is int:
            return float(node.value)
        raise ValueError("not a literal")

    def children(node: ast.expr) -> tuple[ast.expr, ...]:
        if isinstance(node, ast.BinOp):
            return node.left, node.right
        if isinstance(node, ast.Compare):
            return node.left, node.comparators[0]
        if isinstance(node, ast.UnaryOp):
            return (node.operand,)
        return ()

    symbols = {
        ast.Add: "+",
        ast.Sub: "-",
        ast.Mult: "*",
        ast.Pow: "**",
        ast.Mod: "%",
        ast.BitAnd: "&",
        ast.BitOr: "|",
        ast.BitXor: "^",
        ast.Invert: "~",
        ast.USub: "-",
        ast.Eq: "==",
        ast.NotEq: "!=",
        ast.Lt: "<",
        ast.LtE: "<=",
        ast.Gt: ">",
        ast.GtE: ">=",
    }

    def operator(node: ast.expr) -> str:
        if isinstance(node, ast.Constant):
            return str(node.value)
        if isinstance(node, ast.Compare):
            return symbols[type(node.ops[0])]
        if isinstance(node, (ast.UnaryOp, ast.BinOp)):
            return symbols.get(type(node.op), "")
        return ""

    def fail(node: ast.expr, code: str, message: str) -> NoReturn:
        raise ValueError(code + ": " + message)

    truth, reads = sparse_predicate(
        expression, word, number, operator_of=operator, operands_of=children, fail=fail
    )
    return truth, len(reads)


@pytest.mark.parametrize("width", [9, 10, 11, 12])
def test_exact_full_domain_sparse_predicate(width: int) -> None:
    truth, bits = evaluate("((x+y)<9) & (((x%4)*(y%2))==1)", {"x": 6, "y": width - 6})
    expected = tuple(
        x | (y << 6)
        for y in range(1 << (width - 6))
        for x in range(64)
        if x + y < 9 and (x % 4) * (y % 2) == 1
    )
    assert truth == expected
    assert bits == width


@pytest.mark.parametrize(
    "expression,expected",
    [
        ("x==x", (-1,)),
        ("x!=x", ()),
        ("(x-x)==0", (-1,)),
        ("(x**2)==4", (2,)),
        ("~(x!=7)", (7,)),
        ("(x-7)<0", tuple(range(7))),
    ],
)
def test_constants_cancellation_signed_difference_and_powers(
    expression: str,
    expected: tuple[int, ...],
) -> None:
    assert evaluate(expression, {"x": 10}) == (expected, 10)


@pytest.mark.parametrize(
    "expression,width,message",
    [
        ("x==0", 13, "twelve"),
        ("x!=0", 10, "32 satisfying"),
        ("x%3==0", 10, "power of two"),
        ("x%0==0", 10, "power of two"),
        ("x**9==0", 10, "0..8"),
        ("x**x==0", 10, "literal"),
        ("x**8==0", 12, "53 bits"),
        ("(x-2)%4==0", 10, "unsigned"),
        ("~x", 10, "Boolean"),
        ("x+1", 10, "zero or one"),
        ("(x/2)==0", 10, "unsupported"),
    ],
)
def test_invalid_or_oversized_predicates_refuse(
    expression: str,
    width: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        evaluate(expression, {"x": width})

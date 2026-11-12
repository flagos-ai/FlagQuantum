"""Destination-scaled XOR preserves exact dyadic floor and nonzero targets."""

import ast
from collections.abc import Callable, Mapping
from fractions import Fraction
from itertools import product
from typing import NoReturn

import pytest

from flagquantum.compiler._fixed_word import (
    exact_expression,
    fixed_xor_table,
    truth_cubes,
)

pytestmark = pytest.mark.unit


def expression(source: str) -> Callable[[Mapping[str, Fraction]], Fraction]:
    def constant(node: ast.expr) -> Fraction | None:
        return Fraction(str(node.value)) if isinstance(node, ast.Constant) else None

    def operator(node: ast.expr) -> str:
        if isinstance(node, ast.IfExp):
            return "select"
        if isinstance(node, ast.Compare):
            return "<"
        if isinstance(node, (ast.BinOp, ast.UnaryOp)):
            return {ast.Add: "+", ast.Mult: "*", ast.USub: "-"}.get(type(node.op), "")
        return ""

    def children(node: ast.expr) -> tuple[ast.expr, ...]:
        if isinstance(node, ast.IfExp):
            return node.test, node.body, node.orelse
        if isinstance(node, ast.Compare):
            return node.left, *node.comparators
        if isinstance(node, ast.BinOp):
            return node.left, node.right
        if isinstance(node, ast.UnaryOp):
            return (node.operand,)
        return ()

    def fail(node: ast.expr, message: str) -> NoReturn:
        raise ValueError(message)

    return exact_expression(
        ast.parse(source, mode="eval").body,
        name=lambda node: node.id if isinstance(node, ast.Name) else None,
        constant=constant,
        operator=operator,
        children=children,
        fail=fail,
    )


def test_conditional_boundary_and_both_branches_checked() -> None:
    evaluate = expression("-0.375 * x + 0.03125 if x < 0.5 else 0.625 * x")
    for x in range(16):
        value = Fraction(x, 16)
        assert evaluate({"x": value}) == (
            -Fraction(3, 8) * value + Fraction(1, 32)
            if x < 8
            else Fraction(5, 8) * value
        )
    with pytest.raises(ValueError, match="binary rationals"):
        expression("0.5 if x < 1 else 0.9")


def test_intermediate_size_refuses_before_destination_mask() -> None:
    evaluate = expression(f"x * {1 << 255}")
    with pytest.raises(ValueError, match="256 bits"):
        evaluate({"x": Fraction(2)})


@pytest.mark.parametrize(
    "source,diagnostic",
    [
        (str(1 << 256), "256 bits"),
        ("1 + " * 130 + "x", "512 nodes/128 depth"),
        ("0.5 if x else 0.25", "Boolean"),
        ("x / 2", "unsupported"),
    ],
)
def test_exact_expression_refuses_invalid_contract(
    source: str, diagnostic: str
) -> None:
    with pytest.raises(ValueError, match=diagnostic):
        expression(source)({"x": Fraction(2)})


@pytest.mark.parametrize(
    "slope,offset",
    [
        (Fraction(1), Fraction(0)),
        (Fraction(15, 16), Fraction(1, 32)),
        (Fraction(13, 16), Fraction(1, 16)),
        (Fraction(5, 8), Fraction(7, 32)),
        (Fraction(-3, 8), Fraction(1, 32)),
    ],
)
def test_all_inputs_nonzero_targets_and_inverse(
    slope: Fraction, offset: Fraction
) -> None:
    reads, table = fixed_xor_table(
        {"x": (tuple(range(4)), 4)},
        tuple(range(4, 8)),
        target_fraction=4,
        evaluate=lambda inputs: slope * inputs["x"] + offset,
    )
    assert reads == tuple(range(4))
    for x, target in product(range(16), repeat=2):
        value = (slope * Fraction(x, 16) + offset) * 16
        expected = (value.numerator // value.denominator) & 15
        assert target ^ table[x] == target ^ expected
        assert (target ^ table[x]) ^ table[x] == target


def test_guard_shared_bits_uses_one_read_domain() -> None:
    reads, table = fixed_xor_table(
        {"x": ((0, 1, 2, 3), 4)},
        (4, 5, 6, 7),
        target_fraction=4,
        evaluate=lambda inputs: inputs["x"],
        guard=((2, 3), 2),
    )
    assert reads == (0, 1, 2, 3)
    assert table == tuple(x if x >> 2 == 2 else 0 for x in range(16))


@pytest.mark.parametrize("width", range(5))
def test_every_small_boolean_table_partitions_exactly(width: int) -> None:
    for word in range(1 << (1 << width)):
        values = tuple(word >> i & 1 for i in range(1 << width))
        cubes = truth_cubes(values, width)
        for basis in range(1 << width):
            assert (
                sum(
                    all((basis >> bit & 1) == value for bit, value in cube)
                    for cube in cubes
                )
                == values[basis]
            )
        if width == 4 and word == 255:
            break


@pytest.mark.parametrize("kind", ["alias", "reads", "target", "scale", "guard"])
def test_invalid_layout_refuses(kind: str) -> None:
    with pytest.raises(ValueError):
        fixed_xor_table(
            {"x": (tuple(range(8 if kind == "reads" else 4)), 4)},
            (
                (0, 1, 2, 3)
                if kind == "alias"
                else tuple(range(8, 17 if kind == "target" else 12))
            ),
            target_fraction=9 if kind == "scale" else 4,
            evaluate=lambda inputs: inputs["x"],
            guard=((0, 1), 4) if kind == "guard" else None,
        )

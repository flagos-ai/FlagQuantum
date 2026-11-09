"""Independent syntax consumers share exact signed/rational word semantics."""

import ast
from collections.abc import Callable
from dataclasses import replace
from fractions import Fraction
from itertools import product
from typing import NoReturn

import pytest

from benchmarks.synthesis.verification import classical_apply, encode
from flagquantum.compiler._construction.lowering import schedule_arithmetic
from flagquantum.compiler._construction.numeric_expression import WordExpression
from flagquantum.compiler._hybrid.model import HybridProgram, SourceLocation
from flagquantum.ecosystem._qmod.adapter_model import _Expression
from flagquantum.ecosystem._qmod.syntax import Parser, tokenize

pytestmark = pytest.mark.unit


def _text(node: ast.expr) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Constant) and type(node.value) in {int, float}:
        return str(node.value)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        if node.func.id in {"min", "max"} and len(node.args) == 2 and not node.keywords:
            return node.func.id
    if isinstance(node, (ast.BinOp, ast.UnaryOp, ast.Compare)):
        operator = node.ops[0] if isinstance(node, ast.Compare) else node.op
        symbols = {
            ast.Add: "+",
            ast.Sub: "-",
            ast.Mult: "*",
            ast.Pow: "**",
            ast.BitAnd: "&",
            ast.BitOr: "|",
            ast.BitXor: "^",
            ast.Mod: "%",
            ast.USub: "-",
            ast.Invert: "~",
            ast.Eq: "==",
            ast.NotEq: "!=",
            ast.Lt: "<",
            ast.LtE: "<=",
            ast.Gt: ">",
            ast.GtE: ">=",
        }
        if type(operator) in symbols:
            return symbols[type(operator)]
    raise ValueError("test syntax consumer refuses this node")


def _children(node: ast.expr) -> tuple[ast.expr, ...]:
    if isinstance(node, ast.BinOp):
        return node.left, node.right
    if isinstance(node, ast.UnaryOp):
        return (node.operand,)
    if isinstance(node, ast.Call):
        return tuple(node.args)
    if isinstance(node, ast.Compare) and len(node.ops) == 1:
        return node.left, *node.comparators
    return ()


def _fail(node: ast.expr, code: str, message: str) -> NoReturn:
    raise ValueError(f"numeric:{node.lineno}:{node.col_offset + 1}:{code}: {message}")


def _python(
    expression: str, widths: dict[str, int], signed: set[str]
) -> WordExpression[ast.expr]:
    return WordExpression(
        ast.parse(expression, mode="eval").body,
        widths,
        "expression",
        signed,
        text=_text,
        children=_children,
        location=lambda node: SourceLocation(
            "numeric", node.lineno, node.col_offset + 1
        ),
        reject=_fail,
    )


def _source(expression: str, widths: dict[str, int], signed: set[str]) -> _Expression:
    parser = Parser(tokenize(expression, "numeric"))
    node = parser.expression(decimal=True, difference=True)
    parser.take("<eof>")
    return _Expression(node, widths, "expression", signed)


def _unlocated(graph: HybridProgram) -> HybridProgram:
    block = graph.body.blocks[0]
    operations = tuple(replace(op, location=None) for op in block.operations)
    return replace(
        graph,
        location=None,
        body=replace(graph.body, blocks=(replace(block, operations=operations),)),
    )


# Raw a spans its entire three-bit two's-complement domain when signed.
CASES: tuple[
    tuple[str, bool, int, bool, int, Callable[[int, int], Fraction | int]], ...
] = (
    ("a | b", True, 3, True, 0, lambda a, b: a | b),
    ("a ^ b", True, 3, True, 0, lambda a, b: a ^ b),
    ("a == b", True, 1, False, 0, lambda a, b: int(a == b)),
    ("a != b", True, 1, False, 0, lambda a, b: int(a != b)),
    ("-a", True, 4, True, 0, lambda a, b: -a),
    ("a - b", True, 4, True, 0, lambda a, b: a - b),
    (
        "max(3.5, a)",
        False,
        4,
        False,
        1,
        lambda a, b: Fraction(7, 2) if a < Fraction(7, 2) else a,
    ),
    (
        "min(3.5, a)",
        False,
        3,
        False,
        1,
        lambda a, b: Fraction(7, 2) if a > Fraction(7, 2) else a,
    ),
    ("3.5 * a", False, 6, False, 1, lambda a, b: Fraction(7, 2) * a),
    ("a - 3.5", False, 4, True, 1, lambda a, b: a - Fraction(7, 2)),
)


@pytest.mark.parametrize("expression,signed,width,result_signed,fraction,oracle", CASES)
def test_two_consumers_and_independent_full_domain_nonzero_xor(
    expression: str,
    signed: bool,
    width: int,
    result_signed: bool,
    fraction: int,
    oracle: Callable[[int, int], Fraction | int],
) -> None:
    formats = {"a": 3, "b": 2}
    names = {"a"} if signed else set()
    native = _python(expression, formats, names)
    source = _source(expression, formats, names)
    assert native.word_widths[native.node] == source.word_widths[source.node] == width
    assert (
        (native.node in native.signed_words)
        == (source.node in source.signed_words)
        == result_signed
    )
    assert (
        native.fractional_bits.get(native.node, 0)
        == source.fractional_bits.get(source.node, 0)
        == fraction
    )
    graph = native.function()
    assert _unlocated(graph) == _unlocated(source.function())
    inputs = {}
    allocated: list[tuple[int, ...]] = []
    cursor = 0
    for argument in native.arguments:
        bits = formats[native.names[len(allocated)]]
        view = tuple(range(cursor, cursor + bits))
        allocated.append(view)
        inputs[argument.id] = view
        cursor += bits
    output = tuple(range(cursor, cursor + width))
    cursor += width
    candidate = schedule_arithmetic(
        (graph,),
        inputs,
        output,
        entry=graph.name,
        n_qubits=128,
        workspace=tuple(range(cursor, 128)),
        workspace_state="fresh_zero",
    )
    maximum = (1 << width) - 1
    for words in product(*(range(1 << len(view)) for view in allocated)):
        values = dict(zip(native.names, words, strict=True))
        raw = values.get("a", 0)
        a = raw - 8 if signed and raw & 4 else raw
        answer = oracle(a, values.get("b", 0)) * (1 << fraction)
        assert int(answer) == answer
        encoded = int(answer) & maximum
        for target in {0, 1, maximum}:
            initial = encode((*words, target), (*allocated, output))
            expected = encode((*words, target ^ encoded), (*allocated, output))
            assert classical_apply(candidate.circuit, initial) == expected


@pytest.mark.parametrize(
    "expression,width,signed,fraction,value",
    [("4-5", 1, True, 0, -1), ("4-3.5", 1, False, 1, Fraction(1, 2))],
)
def test_constant_difference_formats_are_exact(
    expression: str, width: int, signed: bool, fraction: int, value: Fraction | int
) -> None:
    result = _python(expression, {}, set())
    assert result.word_widths[result.node] == width
    assert (result.node in result.signed_words) == signed
    assert result.fractional_bits[result.node] == fraction
    assert Fraction(result.integer_value(result.node), 1 << fraction) == value


@pytest.mark.parametrize(
    "expression,widths,signed,reason",
    [
        ("0.1*a", {"a": 2}, set(), "exactly binary"),
        ("a+b", {"a": 3, "b": 2}, {"a"}, "signed operands"),
        ("a%3", {"a": 3}, set(), "power-of-two"),
        ("a-b", {"a": 256, "b": 256}, set(), "supported word width"),
    ],
)
def test_source_independent_refusal_retains_node_location(
    expression: str, widths: dict[str, int], signed: set[str], reason: str
) -> None:
    with pytest.raises(ValueError, match="numeric:1:.*" + reason):
        _python(expression, widths, signed)
    with pytest.raises(ValueError, match=reason):
        _source(expression, widths, signed)

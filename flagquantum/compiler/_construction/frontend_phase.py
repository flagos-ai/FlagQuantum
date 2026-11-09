"""Capture native real phase arithmetic using shared Boolean-polynomial semantics."""

from __future__ import annotations

import ast
import math
from typing import TYPE_CHECKING, NoReturn

from .._phase_polynomial import Location, expand_phase
from .frontend_python import _error
from .frontend_views import locations

if TYPE_CHECKING:
    from .frontend_program import _Body


def capture_polynomial_phase(
    owner: _Body,
    expression: ast.expr,
    coefficient: float,
    statement: ast.Call,
    *,
    condition: ast.expr | None = None,
) -> None:
    """Read checked physical views, expand sparsely and emit existing phase effects."""

    def fail(node: ast.expr, code: str, message: str) -> NoReturn:
        _error(owner.source, node, code, message)

    def children(node: ast.expr) -> tuple[ast.expr, ...]:
        if isinstance(node, ast.BinOp):
            return node.left, node.right
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return (node.operand,)
        if isinstance(node, ast.Compare):
            return node.left, *node.comparators
        return ()

    def operator(node: ast.expr) -> str:
        if isinstance(node, ast.Compare):
            comparisons = {
                ast.Eq: "==",
                ast.NotEq: "!=",
                ast.Lt: "<",
                ast.LtE: "<=",
                ast.Gt: ">",
                ast.GtE: ">=",
            }
            return comparisons.get(type(node.ops[0]), "")
        if isinstance(node, ast.Constant) and type(node.value) is int:
            return str(node.value)
        operation = node.op if isinstance(node, (ast.BinOp, ast.UnaryOp)) else None
        for kind, symbol in (
            (ast.Add, "+"),
            (ast.Sub, "-"),
            (ast.Mult, "*"),
            (ast.Div, "/"),
            (ast.Pow, "**"),
            (ast.USub, "-"),
        ):
            if isinstance(operation, kind):
                return symbol
        return ""

    visible = owner.aliases if owner.aliases is not None else owner.ports

    def quantum_word(node: ast.expr) -> tuple[Location, ...] | None:
        base = node
        while isinstance(base, ast.Subscript):
            base = base.value
        if not isinstance(base, ast.Name) or base.id not in visible:
            return None
        if not isinstance(node, (ast.Name, ast.Subscript)):
            return None
        if isinstance(node, ast.Name) and node.id in owner.array_names:
            fail(node, "program.phase", "phase arrays require an element index")
        if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Slice):
            fail(node, "program.phase", "phase arrays require one scalar element")
        return locations(owner, node)

    def number(node: ast.expr) -> float:
        value = owner.phase_number(node)
        if value is None:
            fail(
                node,
                "program.phase",
                "use finite real arithmetic and declared quantum values",
            )
        return value

    # Source limits bound total nodes; this guard also bounds recursive expansion.
    pending = [(expression, 0)]
    if condition is not None:
        pending.append((condition, 0))
    while pending:
        node, depth = pending.pop()
        if depth >= 128:
            fail(node, "budget.phase", "native phase expression depth exceeds 128")
        if isinstance(node, ast.Compare) and (
            len(node.ops) != 1
            or not isinstance(
                node.ops[0], (ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE)
            )
        ):
            fail(
                node, "program.control", "phase control requires one integer comparison"
            )
        if isinstance(node, ast.BinOp) and not isinstance(
            node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow)
        ):
            fail(node, "program.phase", "use +, -, *, / and literal powers 0..8")
        pending.extend((child, depth + 1) for child in children(node))
    polynomial, reads = expand_phase(
        expression,
        quantum_word,
        number,
        operator_of=operator,
        operands_of=children,
        fail=fail,
        condition=condition,
    )
    # Algebraic cancellation must not hide uninitialized or illegal reads.
    for value, bit in reads:
        owner.emit("qreg.control_phase", statement, (value,), indices=(bit,), angle=0.0)
    for mask, weight in sorted(polynomial.items()):
        angle = coefficient * weight
        if not math.isfinite(angle) or abs(angle) > 2**53:
            fail(expression, "program.phase", "phase angle exceeds finite ±2**53 range")
        selected = tuple(
            location for i, location in enumerate(reads) if mask & (1 << i)
        )
        if selected:
            owner.emit(
                "qreg.control_phase",
                statement,
                tuple(value for value, _ in selected),
                indices=tuple(bit for _, bit in selected),
                angle=float(angle),
            )
        else:
            owner.emit("qreg.phase", statement, (), angle=float(angle))

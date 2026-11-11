"""Capture native real phase arithmetic using shared Boolean-polynomial semantics."""

from __future__ import annotations

import ast
import math
from typing import TYPE_CHECKING, NoReturn

from .._hybrid.model import Value
from .._hybrid.verifier import HybridVerificationError
from .._phase_factors import Cube
from .._phase_polynomial import Location, expand_phase
from .frontend_numeric import has_numeric_reads
from .frontend_phase_factors import (
    capture_sparse_predicate_phase,
    emit_factor,
    expand_factored_phase,
)
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

    numeric = has_numeric_reads(owner, expression)
    if numeric and (
        condition is not None
        or any(isinstance(node, ast.Compare) for node in ast.walk(expression))
    ):
        fail(
            expression,
            "program.numeric",
            "numeric comparisons and conditions are not admitted in phase expressions",
        )

    def numeric_format(node: ast.expr) -> tuple[bool, int]:
        ports = owner.alias_ports if owner.aliases is not None else owner.ports
        assert ports is not None
        if isinstance(node, ast.Name) and node.id in ports:
            port = ports[node.id]
            return port.signed, port.fractional_bits
        return False, 0

    def children(node: ast.expr) -> tuple[ast.expr, ...]:
        if isinstance(node, ast.BinOp):
            return node.left, node.right
        if isinstance(node, ast.UnaryOp) and isinstance(
            node.op, (ast.USub, ast.Invert)
        ):
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
            (ast.BitAnd, "&"),
            (ast.BitOr, "|"),
            (ast.BitXor, "^"),
            (ast.Mod, "%"),
            (ast.USub, "-"),
            (ast.Invert, "~"),
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
    pending: list[tuple[ast.expr, int, bool]] = [(expression, 0, False)]
    if condition is not None:
        pending.append((condition, 0, True))
    while pending:
        node, depth, is_condition = pending.pop()
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
        if isinstance(node, ast.BinOp):
            allowed: tuple[type[ast.operator], ...]
            if is_condition:
                allowed = (
                    ast.Add,
                    ast.Sub,
                    ast.Mult,
                    ast.Div,
                    ast.Pow,
                    ast.BitAnd,
                    ast.BitOr,
                    ast.BitXor,
                    ast.Mod,
                )
            else:
                allowed = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow)
            if not isinstance(node.op, allowed):
                fail(
                    node,
                    "program.phase",
                    "use +, -, *, / and literal powers 0..8"
                    + (
                        " with bounded &, |, ^ and % in predicates"
                        if is_condition
                        else ""
                    ),
                )
        if (
            isinstance(node, ast.UnaryOp)
            and isinstance(node.op, ast.Invert)
            and not is_condition
        ):
            fail(
                node,
                "program.phase",
                "bitwise inversion is only supported in predicates",
            )
        pending.extend((child, depth + 1, is_condition) for child in children(node))
    factors: tuple[tuple[Cube, Value], ...] = ()
    try:
        polynomial, reads = expand_phase(
            expression,
            quantum_word,
            number,
            operator_of=operator,
            operands_of=children,
            fail=fail,
            condition=condition,
            signed_word=lambda node: numeric_format(node)[0],
            fraction_bits=lambda node: numeric_format(node)[1],
        )
    except HybridVerificationError as error:
        budget_refusal = all(
            diagnostic.code == "construction.python.budget.phase"
            for diagnostic in error.diagnostics
        )
        if (
            condition is not None
            and budget_refusal
            and not has_numeric_reads(owner, condition)
        ):
            constant = owner.phase_number(expression)
            if constant is not None:
                angle = coefficient * constant
                if not math.isfinite(angle) or abs(angle) > 2**53:
                    fail(
                        expression,
                        "program.phase",
                        "phase angle exceeds finite ±2**53 range",
                    )
                if capture_sparse_predicate_phase(
                    owner,
                    condition,
                    float(angle),
                    statement,
                    quantum_word,
                    number,
                    operator,
                    children,
                    fail,
                ):
                    return
        if (
            numeric
            or condition is not None
            or any(isinstance(node, ast.Compare) for node in ast.walk(expression))
            or not all(
                diagnostic.code == "construction.python.budget.phase"
                for diagnostic in error.diagnostics
            )
        ):
            raise
        expanded = expand_factored_phase(
            owner, expression, quantum_word, number, operator, children, fail
        )
        if expanded is None:
            raise
        polynomial, reads, factors = expanded
    for factor, value in factors:
        emit_factor(owner, statement, factor, value)
    # Emitted monomials witness their reads; canceled reads still need a check.
    covered = 0
    for mask in polynomial:
        covered |= mask
    for index, (value, bit) in enumerate(reads):
        if not covered & (1 << index):
            owner.emit(
                "qreg.control_phase", statement, (value,), indices=(bit,), angle=0.0
            )
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

    for _, value in reversed(factors):
        owner.emit("qreg.uncompute", statement, (value,))
        owner.emit("qreg.release", statement, (value,))

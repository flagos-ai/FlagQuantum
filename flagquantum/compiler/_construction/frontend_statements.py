"""Specialize closed classical conditions without evaluating Python or qubits."""

from __future__ import annotations

import ast
from collections.abc import Sequence
from typing import TYPE_CHECKING

from .frontend_classical import classical_number
from .frontend_evolution import flush_evolutions
from .frontend_python import _error
from .frontend_storage import declare_hidden
from .frontend_views import capture_loop
from .program_types import adjoint

if TYPE_CHECKING:
    from .frontend_program import _Body


def _condition(owner: _Body, node: ast.expr, depth: int = 0) -> bool:
    if depth >= 32:
        _error(owner.source, node, "budget", "classical condition depth exceeds 32")
    if isinstance(node, ast.Constant) and type(node.value) is bool:
        return bool(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        return not _condition(owner, node.operand, depth + 1)
    if isinstance(node, ast.BoolOp) and isinstance(node.op, (ast.And, ast.Or)):
        # Validate every operand, including those beyond a short-circuit point.
        values = tuple(_condition(owner, value, depth + 1) for value in node.values)
        return all(values) if isinstance(node.op, ast.And) else any(values)
    if (
        isinstance(node, ast.Compare)
        and len(node.ops) == len(node.comparators) == 1
        and isinstance(
            node.ops[0], (ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE)
        )
    ):
        left = classical_number(owner, node.left)
        right = classical_number(owner, node.comparators[0])
        if left is not None and right is not None:
            operation = node.ops[0]
            if isinstance(operation, ast.Eq):
                return left == right
            if isinstance(operation, ast.NotEq):
                return left != right
            if isinstance(operation, ast.Lt):
                return left < right
            if isinstance(operation, ast.LtE):
                return left <= right
            if isinstance(operation, ast.Gt):
                return left > right
            return left >= right
    _error(
        owner.source,
        node,
        "program.condition",
        "if requires a finite classical comparison, bool literal or and/or/not; "
        "use control for a coherent quantum condition",
    )


def capture_conditional(owner: _Body, node: ast.If) -> None:
    """Capture only the selected classical branch under existing shared budgets."""
    if owner.branch_depth >= 32:
        _error(owner.source, node, "budget", "classical branch nesting exceeds 32")
    if any(
        isinstance(child, (ast.Return, ast.Break, ast.Continue))
        for child in ast.walk(node)
    ):
        _error(
            owner.source,
            node,
            "program.condition",
            "branch return, break and continue are not admitted",
        )
    selected = node.body if _condition(owner, node.test) else node.orelse
    owner.branch_depth += 1
    try:
        capture_statements(owner, selected, nested=True)
    finally:
        owner.branch_depth -= 1


def capture_statements(
    owner: _Body, body: Sequence[ast.stmt], *, nested: bool = False
) -> None:
    """Dispatch native syntax to domain-owned capture without executing Python."""
    statements = iter(body)
    for statement in statements:
        if isinstance(statement, ast.AnnAssign):
            if nested:
                _error(
                    owner.source,
                    statement,
                    "program.storage",
                    "hidden declarations require the unconditional program root",
                )
            declare_hidden(owner, statement)
        elif isinstance(statement, ast.For):
            capture_loop(owner, statement)
        elif isinstance(statement, ast.If):
            capture_conditional(owner, statement)
        elif isinstance(statement, ast.AugAssign):
            if owner.aliases is not None:
                _error(
                    owner.source,
                    statement,
                    "program.borrow",
                    "mutable helpers cannot initialize or compute Outputs",
                )
            owner.assign(statement)
        elif isinstance(statement, ast.With):
            flush_evolutions(owner)
            if owner.region_marker(statement) is adjoint:
                owner.adjoint_scope(statement)
            else:
                owner.within_apply(statement, next(statements, None))
        elif isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call):
            owner.call(statement.value)
        elif isinstance(statement, ast.Return) and (
            statement.value is None
            or isinstance(statement.value, ast.Constant)
            and statement.value.value is None
        ):
            if nested:
                _error(
                    owner.source,
                    statement,
                    "program.loop",
                    "return inside a static loop is not admitted",
                )
            continue
        elif (
            isinstance(statement, ast.Expr)
            and isinstance(statement.value, ast.Constant)
            and isinstance(statement.value.value, str)
        ):
            continue
        else:
            _error(
                owner.source,
                statement,
                "program.statement",
                "use captured operations and bounded static for loops",
            )

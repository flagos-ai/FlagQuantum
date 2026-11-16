"""Capture finite signed-clause phase predicates without truth-table expansion."""

from __future__ import annotations

import ast
import math
from typing import TYPE_CHECKING

from .._hybrid.model import Operation, Value, ValueId
from . import cnf_phase
from .cnf_phase import Clause, Location
from .frontend_phase import capture_polynomial_phase
from .frontend_python import _error
from .frontend_views import gate_target
from .register_profile import register_type

if TYPE_CHECKING:
    from .frontend_program import _Body


def _terms(expression: ast.expr, *, conjunction: bool) -> tuple[ast.expr, ...]:
    pending = [expression]
    result = []
    while pending:
        node = pending.pop()
        if isinstance(node, ast.BinOp) and isinstance(
            node.op, ast.BitAnd if conjunction else ast.BitOr
        ):
            pending.extend((node.right, node.left))
        elif isinstance(node, ast.BoolOp) and isinstance(
            node.op, ast.And if conjunction else ast.Or
        ):
            pending.extend(reversed(node.values))
        else:
            result.append(node)
    return tuple(result)


def _bitwise_condition(
    clauses: tuple[tuple[ast.expr, ...], ...], anchor: ast.expr
) -> ast.expr:
    """Translate the checked Boolean spelling to the existing phase syntax."""

    def combine(nodes: list[ast.expr], conjunction: bool) -> ast.expr:
        result = nodes[0]
        for following in nodes[1:]:
            result = ast.copy_location(
                ast.BinOp(
                    result, ast.BitAnd() if conjunction else ast.BitOr(), following
                ),
                anchor,
            )
        return result

    conditions = []
    for clause in clauses:
        terms = []
        for node in clause:
            if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
                node = ast.copy_location(ast.UnaryOp(ast.Invert(), node.operand), node)
            terms.append(node)
        conditions.append(combine(terms, False))
    return combine(conditions, True)


def capture_cnf_phase(
    owner: _Body, condition: ast.expr, body: ast.Call, constant: float | None
) -> bool:
    """Admit signed indexed CNF for control(predicate, lambda: phase(1, pi))."""
    if constant != 1 or len(body.args) != 2:
        return False
    angle = owner.phase_number(body.args[1])
    if angle not in (math.pi, -math.pi):
        return False
    raw = _terms(condition, conjunction=True)
    if len(raw) == 1:
        return False
    literals = tuple(_terms(clause, conjunction=False) for clause in raw)
    if any(
        not isinstance(
            (
                node.operand
                if isinstance(node, ast.UnaryOp)
                and isinstance(node.op, (ast.Invert, ast.Not))
                else node
            ),
            ast.Subscript,
        )
        for clause in literals
        for node in clause
    ):
        return False
    resolved: list[Clause] = []
    reads = []
    for clause in literals:
        terms = []
        for literal in clause:
            negative = isinstance(literal, ast.UnaryOp)
            node = literal.operand if isinstance(literal, ast.UnaryOp) else literal
            location = gate_target(owner, node)
            reads.append(location)
            terms.append((location, negative))
        resolved.append(tuple(terms))
    small = len(set(reads)) <= 12
    if small and not any(
        isinstance(node, (ast.BoolOp, ast.Not)) for node in ast.walk(condition)
    ):
        # Keep the established small-predicate candidate and its zero-work
        # implementations, including repeated Grover callbacks.
        return False
    if len(raw) > 80 or any(len(clause) > 3 for clause in literals):
        _error(
            owner.source,
            condition,
            "budget.control",
            "CNF phase admits at most eighty clauses and three literals per clause",
        )
    clauses = cnf_phase.canonical_cnf(tuple(resolved))
    if small:
        for value, bit in reads:
            owner.emit("qreg.gate", body, (value,), gate="i", index=bit)
        capture_polynomial_phase(
            owner,
            body.args[0],
            angle,
            body,
            condition=_bitwise_condition(literals, condition),
        )
        return True
    # Tautologies still read source bits; verification must reject uninitialized
    # or dropped storage before simplification can erase the offending read.
    for value, bit in reads:
        owner.emit("qreg.gate", body, (value,), gate="i", index=bit)
    if not clauses:
        owner.emit("qreg.phase", body, (), angle=math.pi)
        return True
    scratch = Value(ValueId(owner.name, owner.counter), register_type(len(clauses) + 1))
    owner.counter += 1
    owner.operations.append(
        Operation(
            "qreg.hidden",
            results=(scratch,),
            attributes={"name": f"$cnf{scratch.id.index}", "workspace": True},
            location=owner.source.location(body),
        )
    )
    owner.emit("qreg.allocate", body, (scratch,))

    def gate(location: Location, name: str) -> None:
        value, bit = location
        owner.emit("qreg.gate", body, (value,), gate=name, index=bit)

    def controlled_x(locations: tuple[Location, ...]) -> None:
        owner.emit(
            "qreg.control_x",
            body,
            tuple(value for value, _ in locations),
            indices=tuple(bit for _, bit in locations),
        )

    cnf_phase.emit_cnf_phase(clauses, scratch, gate=gate, controlled_x=controlled_x)
    owner.emit("qreg.drop", body, (scratch,))
    return True

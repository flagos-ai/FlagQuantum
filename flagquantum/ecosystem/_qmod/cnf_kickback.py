"""Lower a checked compute/phase/uncompute for signed clauses of a Boolean CNF."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from ...compiler._construction import cnf_phase
from ...compiler._construction.cnf_phase import Clause
from ...compiler._construction.register_profile import register_type
from ...compiler._hybrid.model import Operation, Value, ValueId
from .syntax import Expression, Statement, reject

if TYPE_CHECKING:
    from .context import _Adapt
    from .phase import Location


def _terms(expression: Expression, operator: str) -> list[Expression]:
    pending = [expression]
    result = []
    while pending:
        node = pending.pop()
        if node.token.text == operator:
            pending.extend(reversed(node.children))
        else:
            result.append(node)
    return result


def lower_cnf_kickback(owner: _Adapt, statements: tuple[Statement, ...]) -> bool:
    declare, allocate, scope, free = statements
    marker = declare.declaration
    if (
        marker is None
        or marker.kind not in {"qbit", "qnum"}
        or marker.width not in {None, 1}
        or marker.fractional_bits
        or scope.within is None
        or len(scope.within) != 2
        or scope.apply is None
        or len(scope.apply) != 1
    ):
        return False
    name = marker.name.text

    def matches(statement: Statement, gate: str, args: tuple[str, ...]) -> bool:
        return (
            statement == Statement(statement.token, statement.arguments)
            and statement.token.text == gate
            and tuple(arg.token.text for arg in statement.arguments) == args
            and not any(arg.children for arg in statement.arguments)
        )

    mark = scope.apply[0]
    if not (
        (
            matches(allocate, "allocate", ("1", name))
            or marker.kind == "qbit"
            and matches(allocate, "allocate", (name,))
        )
        and matches(scope.within[0], "X", (name,))
        and matches(scope.within[1], "H", (name,))
        and matches(free, "free", (name,))
        and mark == Statement(mark.token, mark.arguments)
        and mark.token.text == "^="
        and len(mark.arguments) == 2
        and not mark.arguments[0].children
        and mark.arguments[0].token.text == name
        and mark.arguments[1].token.text == "and"
    ):
        return False
    if name in owner.parameters or name in owner.constants:
        reject(marker.name, "name", "kickback marker must be a fresh local")

    token = mark.token
    raw_clauses = _terms(mark.arguments[1], "and")
    if len(raw_clauses) > 80:
        reject(token, "budget.control", "CNF phase exceeds eighty clauses")
    resolved_clauses: list[Clause] = []
    for clause in raw_clauses:
        literals = _terms(clause, "or")
        if len(literals) > 3:
            reject(clause.token, "control", "CNF clauses admit at most three literals")
        resolved: list[tuple[Location, bool]] = []
        for literal in literals:
            negative = literal.token.text == "==" and len(literal.children) == 2
            if negative:
                literal, zero = literal.children
                if zero.children or zero.token.text != "0":
                    reject(zero.token, "control", "CNF negation must compare to zero")
            if (
                literal.token.text != "["
                or len(literal.children) != 2
                or literal.children[0].children
                or literal.children[0].token.text == name
            ):
                reject(literal.token, "control", "CNF requires disjoint indexed bits")
            location = owner.gate_target(literal)
            # Even a tautological clause reads an initialized source.
            value, bit = location
            owner.effect_op("qreg.gate", token, (value,), gate="i", index=bit)
            resolved.append((location, negative))
        resolved_clauses.append(tuple(resolved))
    clauses = cnf_phase.canonical_cnf(tuple(resolved_clauses))
    if not clauses:
        owner.effect_op("qreg.phase", token, (), angle=math.pi)
        return True

    scratch = Value(
        ValueId(owner.scope, owner.counter), register_type(len(clauses) + 1)
    )
    owner.counter += 1
    owner.append_op(
        token,
        Operation(
            "qreg.hidden",
            results=(scratch,),
            attributes={"name": f"$cnf{scratch.id.index}", "workspace": True},
            location=token.location,
        ),
    )
    owner.effect_op("qreg.allocate", token, (scratch,))

    def gate(location: Location, name: str) -> None:
        value, bit = location
        owner.effect_op("qreg.gate", token, (value,), gate=name, index=bit)

    cnf_phase.emit_cnf_phase(
        clauses,
        scratch,
        gate=gate,
        controlled_x=lambda locations: owner.emit_indexed_x(token, locations),
    )
    owner.effect_op("qreg.drop", token, (scratch,))
    return True

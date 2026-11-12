"""Bounded clause compute/phase/uncompute shared by source frontends."""

from __future__ import annotations

from collections.abc import Callable

from .._hybrid.model import Value
from .register_profile import register_width

Location = tuple[Value, int]
Clause = tuple[tuple[Location, bool], ...]


def canonical_cnf(clauses: tuple[Clause, ...]) -> tuple[Clause, ...]:
    """Remove duplicate literals and tautologies after bounding the raw input."""
    if not 1 <= len(clauses) <= 80:
        raise ValueError("CNF phase requires one to eighty clauses")
    result = []
    for clause in clauses:
        if not 1 <= len(clause) <= 3:
            raise ValueError("CNF clauses admit at most three literals")
        resolved: dict[Location, bool] = {}
        tautology = False
        for location, negative in clause:
            value, bit = location
            if type(negative) is not bool or not 0 <= bit < register_width(value):
                raise ValueError("CNF literals require valid signed bit locations")
            if location in resolved and resolved[location] != negative:
                tautology = True
            resolved[location] = negative
        if not tautology:
            result.append(tuple(resolved.items()))
    return tuple(result)


def emit_cnf_phase(
    clauses: tuple[Clause, ...],
    scratch: Value,
    *,
    gate: Callable[[Location, str], None],
    controlled_x: Callable[[tuple[Location, ...]], None],
) -> None:
    """Apply a minus sign to satisfying words and restore every scratch bit."""
    if (
        not clauses
        or register_width(scratch) != len(clauses) + 1
        or any(value == scratch for clause in clauses for (value, _), _ in clause)
    ):
        raise ValueError("CNF needs disjoint clause flags and a fresh phase marker")

    def clause_xor(clause: Clause, target: Location) -> None:
        positive = tuple(location for location, negative in clause if not negative)
        for location in positive:
            gate(location, "x")
        controlled_x((*tuple(location for location, _ in clause), target))
        for location in reversed(positive):
            gate(location, "x")
        gate(target, "x")

    flags = tuple((scratch, i) for i in range(len(clauses)))
    marker = scratch, len(clauses)
    for clause, flag in zip(clauses, flags, strict=True):
        clause_xor(clause, flag)
    conjunction = (*flags, marker)
    controlled_x(conjunction)
    gate(marker, "z")
    controlled_x(conjunction)
    for clause, flag in zip(reversed(clauses), reversed(flags), strict=True):
        clause_xor(clause, flag)

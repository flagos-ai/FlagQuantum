"""Canonicalize Boolean feedback while retaining ordered quantum effects."""

import hashlib
import json
from collections.abc import Iterable
from typing import TypeAlias

# Square-free Boolean polynomials over GF(2). A term is a product of
# classical inputs and measurement events, and addition is symmetric difference.
Polynomial: TypeAlias = frozenset[frozenset[int]]
_ZERO: Polynomial = frozenset()
_ONE: Polynomial = frozenset({frozenset()})


def _variable(index: int) -> Polynomial:
    return frozenset({frozenset({index})})


def _and(left: Polynomial, right: Polynomial) -> Polynomial:
    terms: set[frozenset[int]] = set()
    for a in left:
        for b in right:
            term = a | b
            if term in terms:
                terms.remove(term)
            else:
                terms.add(term)
    return frozenset(terms)


def feedback_digest(records: Iterable[str], inputs: tuple[int, ...]) -> tuple[int, str]:
    """Hash the quantum effects with feedback expressed in immutable variables.

    A scope snapshots its condition, and inactive measurement leaves the
    previous slot value in place. Scratch starts in zero in these workloads.
    Known-zero reset instructions and descriptive marks are accounted for
    separately. Register declarations must have been removed by the caller.
    """
    slots = {f"b{index}": _variable(index) for index in inputs}
    event = max(inputs, default=-1) + 1
    scopes = [_ONE]
    effects: list[tuple[str, list[str], list[list[int]]]] = []
    for record in records:
        if record.startswith(("INFO_TAG ", "R ")):
            continue
        body, predicate, control = record.partition(" if ")
        opcode, *operands = body.split()
        guard = _and(scopes[-1], slots.get(control, _ZERO)) if predicate else scopes[-1]
        if opcode == "PUSH_CONDITION":
            scopes.append(guard)
            continue
        if opcode == "POP_CONDITION":
            scopes.pop()
            continue
        if opcode in ("BIT_STORE0", "BIT_INVERT", "HMR"):
            result = operands.pop()
            old = slots.get(result, _ZERO)
            if opcode == "HMR":
                value = _variable(event)
                event += 1
            else:
                value = _ZERO if opcode == "BIT_STORE0" else old ^ _ONE
            slots[result] = old ^ _and(guard, old ^ value)
        if opcode not in ("BIT_STORE0", "BIT_INVERT"):
            effects.append((opcode, operands, sorted(sorted(term) for term in guard)))
    if len(scopes) != 1:
        raise ValueError("Unbalanced KMX feedback scopes")
    data = json.dumps(effects, separators=(",", ":")).encode()
    return len(effects), hashlib.sha256(data).hexdigest()

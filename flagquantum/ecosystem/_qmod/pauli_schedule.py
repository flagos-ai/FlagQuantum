"""Bounded static repetition arithmetic and commuting Pauli coalescence."""

from __future__ import annotations

import math
from collections.abc import Callable
from operator import add, mul, sub, truediv
from typing import TYPE_CHECKING

from .syntax import Expression, Token, reject

if TYPE_CHECKING:
    from .adapter import _Adapt

Pauli = tuple[tuple[int, str], ...]
Term = tuple[Pauli, float]
_BINARY: dict[str, Callable[[float, float], float]] = {
    "+": add,
    "-": sub,
    "*": mul,
    "/": truediv,
}
_MAX_TERMS = 8192
_MAX_SCAN = 64


def ceiling_repetitions(owner: _Adapt, expression: Expression) -> int:
    """Evaluate the ceiling's real expression at specialization time."""

    def real(node: Expression) -> float:
        text = node.token.text
        if not node.children:
            return owner.classical_number(node)
        if text == "-" and len(node.children) == 1:
            return -real(node.children[0])
        if len(node.children) != 2 or text not in _BINARY.keys() | {"**"}:
            reject(node.token, "evolution", "static ceiling requires real arithmetic")
        left, right = (real(child) for child in node.children)
        if text == "**":
            if not 0 <= left <= 256 or not 0 <= right <= 8:
                reject(
                    node.token,
                    "evolution",
                    "static power exceeds its bounded real domain",
                )
            result = float(left**right)
        else:
            if text == "/" and right == 0:
                reject(node.token, "evolution", "static ceiling division by zero")
            result = _BINARY[text](left, right)
        if not math.isfinite(result) or abs(result) > 2**53:
            reject(
                node.token,
                "evolution",
                "static ceiling arithmetic exceeds finite ±2**53",
            )
        return result

    return math.ceil(real(expression))


def compact_commuting(
    token: Token, sequence: list[Term], repetitions: int
) -> list[Term] | None:
    """Merge equal Paulis across commuting terms without crossing barriers."""
    if len(sequence) * repetitions > _MAX_TERMS:
        reject(
            token,
            "budget.evolution",
            f"static Pauli schedule exceeds {_MAX_TERMS} terms",
        )
    if any(not math.isfinite(angle) for _, angle in sequence):
        reject(token, "angle", "static Pauli schedule requires finite angles")
    # Use the ordinary path outside the roundoff envelope for adding angles.
    if any(abs(angle) > 4096 for _, angle in sequence) or (
        math.fsum(abs(angle) for _, angle in sequence) * repetitions > 4096
    ):
        return None

    def masks(support: Pauli) -> tuple[int, int]:
        return (
            sum(1 << i for i, axis in support if axis in {"X", "Y"}),
            sum(1 << i for i, axis in support if axis in {"Y", "Z"}),
        )

    marked = [(support, masks(support), angle) for support, angle in sequence]
    output: list[tuple[Pauli, tuple[int, int], list[float]]] = []
    identity = []
    for _ in range(repetitions):
        for support, (x, z), angle in marked:
            if not support:
                identity.append(angle)
                continue
            merged = False
            lower = max(len(output) - _MAX_SCAN, 0)
            for i in range(len(output) - 1, lower - 1, -1):
                previous, (px, pz), angles = output[i]
                if previous == support:
                    angles.append(angle)
                    merged = True
                    break
                if ((x & pz) ^ (z & px)).bit_count() % 2:
                    break
            if not merged:
                output.append((support, (x, z), [angle]))
    result = [(support, math.fsum(angles)) for support, _, angles in output]
    if identity:
        result.insert(0, ((), math.fsum(identity)))
    return result

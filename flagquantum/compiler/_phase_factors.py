"""Preserve bounded Boolean cubes before real phase expansion duplicates them."""

from collections.abc import Callable
from typing import TypeVar

from ._phase_polynomial import Location

Cube = tuple[tuple[Location, bool], ...]
MAX_FACTORS = 8
_Node = TypeVar("_Node")


def preserve_factors(
    expression: _Node,
    quantum_word: Callable[[_Node], tuple[Location, ...] | None],
    *,
    operator_of: Callable[[_Node], str],
    operands_of: Callable[[_Node], tuple[_Node, ...]],
    replace_operands: Callable[[_Node, tuple[_Node, ...]], _Node],
    make_factor: Callable[[_Node, int], _Node],
) -> tuple[_Node, tuple[Cube, ...]]:
    """Share products of three to six distinct signed bit literals.

    Frontends supply syntax conversion and resolved logical locations. Compiler
    verification and candidate selection own the predicates' physical realization.
    """
    factors: list[Cube] = []

    def cube(node: _Node) -> list[tuple[Location, bool]] | None:
        children = operands_of(node)
        if operator_of(node) == "*":
            left, right = (cube(child) for child in children)
            return None if left is None or right is None else left + right
        positive = True
        operand = node
        if (
            operator_of(node) == "-"
            and len(children) == 2
            and not operands_of(children[0])
            and operator_of(children[0]) == "1"
        ):
            positive, operand = False, children[1]
        word = quantum_word(operand)
        return [(word[0], positive)] if word is not None and len(word) == 1 else None

    def visit(node: _Node) -> _Node:
        literals = cube(node)
        if literals is not None and 3 <= len(literals) <= 6:
            ordered = tuple(
                sorted(
                    set(literals), key=lambda item: (item[0][0].id, item[0][1], item[1])
                )
            )
            if len(ordered) >= 3 and len({location for location, _ in ordered}) == len(
                ordered
            ):
                if ordered not in factors and len(factors) < MAX_FACTORS:
                    factors.append(ordered)
                if ordered in factors:
                    return make_factor(node, factors.index(ordered))
        if literals is not None:
            return node
        return replace_operands(
            node, tuple(visit(child) for child in operands_of(node))
        )

    return visit(expression), tuple(factors)

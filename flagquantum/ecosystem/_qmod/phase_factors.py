"""Preserve bounded Boolean cubes before real phase expansion duplicates them."""

from collections.abc import Callable
from dataclasses import replace

from .phase import Location
from .syntax import Expression

Cube = tuple[tuple[Location, bool], ...]
MAX_FACTORS = 8


def preserve_factors(
    expression: Expression,
    quantum_word: Callable[[Expression], tuple[Location, ...] | None],
) -> tuple[Expression, tuple[Cube, ...]]:
    """Share products of three to six distinct signed bit literals.

    This is a syntax-to-existing-arithmetic translation, not a numerical oracle.
    Compiler verification and candidate selection own their physical realization.
    """
    factors: list[Cube] = []

    def cube(node: Expression) -> list[tuple[Location, bool]] | None:
        if node.token.text == "*":
            left, right = (cube(child) for child in node.children)
            return None if left is None or right is None else left + right
        positive = True
        operand = node
        if (
            node.token.text == "-"
            and len(node.children) == 2
            and not node.children[0].children
            and node.children[0].token.text == "1"
        ):
            positive, operand = False, node.children[1]
        word = quantum_word(operand)
        return [(word[0], positive)] if word is not None and len(word) == 1 else None

    def visit(node: Expression) -> Expression:
        literals = cube(node)
        if literals is not None and 3 <= len(literals) <= 6:
            ordered = tuple(
                sorted(
                    set(literals),
                    key=lambda item: (item[0][0].id, item[0][1], item[1]),
                )
            )
            if len(ordered) >= 3 and len({location for location, _ in ordered}) == len(
                ordered
            ):
                if ordered not in factors and len(factors) < MAX_FACTORS:
                    factors.append(ordered)
                if ordered in factors:
                    return Expression(
                        replace(
                            node.token, text=f"$phase_factor{factors.index(ordered)}"
                        )
                    )
        if literals is not None:
            return node
        children = tuple(visit(child) for child in node.children)
        return replace(node, children=children)

    return visit(expression), tuple(factors)

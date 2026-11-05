"""Bounded real polynomials over physical Boolean locations for phase effects."""

import math
from collections.abc import Callable

from ...compiler._hybrid.model import Value
from .syntax import Expression, reject

Location = tuple[Value, int]
Polynomial = dict[int, float]
MAX_PHASE_INPUT_BITS = 24
MAX_PHASE_DEGREE = 8
MAX_PHASE_TERMS = 256
MAX_PHASE_PRODUCTS = 20_000


def expand_phase(
    expression: Expression,
    quantum_word: Callable[[Expression], tuple[Location, ...] | None],
    number: Callable[[Expression], float],
) -> tuple[Polynomial, tuple[Location, ...]]:
    """Expand real arithmetic with b*b=b, retaining every encountered input bit."""
    locations: list[Location] = []
    products = 0

    def checked(polynomial: Polynomial, node: Expression) -> Polynomial:
        if len(polynomial) > MAX_PHASE_TERMS:
            reject(node.token, "budget.phase", "phase exceeds 256 monomials")
        if any(mask.bit_count() > MAX_PHASE_DEGREE for mask in polynomial):
            reject(node.token, "budget.phase", "phase monomial exceeds degree eight")
        if any(
            not math.isfinite(value) or abs(value) > 2**53
            for value in polynomial.values()
        ):
            reject(node.token, "phase", "phase coefficient exceeds finite ±2**53 range")
        return {mask: value for mask, value in polynomial.items() if value != 0}

    def multiply(left: Polynomial, right: Polynomial, node: Expression) -> Polynomial:
        nonlocal products
        products += len(left) * len(right)
        if products > MAX_PHASE_PRODUCTS:
            reject(node.token, "budget.phase", "phase exceeds 20000 monomial products")
        result: Polynomial = {}
        for first, first_value in left.items():
            for second, second_value in right.items():
                mask = first | second
                result[mask] = result.get(mask, 0.0) + first_value * second_value
        return checked(result, node)

    def visit(node: Expression) -> Polynomial:
        word = quantum_word(node)
        if word is not None:
            result: Polynomial = {}
            for bit, location in enumerate(word):
                if location not in locations:
                    locations.append(location)
                    if len(locations) > MAX_PHASE_INPUT_BITS:
                        reject(
                            node.token,
                            "budget.phase",
                            "phase exceeds 24 quantum input bits",
                        )
                mask = 1 << locations.index(location)
                result[mask] = result.get(mask, 0.0) + 2.0**bit
            return result
        text = node.token.text
        if text == "-" and len(node.children) == 1:
            return {mask: -value for mask, value in visit(node.children[0]).items()}
        if text in {"+", "-", "*", "/", "**"}:
            left = visit(node.children[0])
            if text == "**":
                result = {0: 1.0}
                for _ in range(int(node.children[1].token.text)):
                    result = multiply(result, left, node)
                return result
            right = visit(node.children[1])
            if text == "*":
                return multiply(left, right, node)
            if text == "/":
                if any(mask != 0 for mask in right) or not right.get(0):
                    reject(
                        node.token,
                        "phase",
                        "phase divisor must be a nonzero classical real",
                    )
                return checked(
                    {mask: value / right[0] for mask, value in left.items()}, node
                )
            result = dict(left)
            sign = 1 if text == "+" else -1
            for mask, value in right.items():
                result[mask] = result.get(mask, 0.0) + sign * value
            return checked(result, node)
        return checked({0: number(node)}, node)

    polynomial = visit(expression)
    return polynomial, tuple(locations)

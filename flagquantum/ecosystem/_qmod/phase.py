"""Bounded real polynomials over physical Boolean locations for phase effects."""

import math
from collections.abc import Callable
from operator import and_, eq, ge, gt, le, lt, ne, or_, xor

from ...compiler._hybrid.model import Value
from .syntax import Expression, reject

Location = tuple[Value, int]
Polynomial = dict[int, float]
MAX_PHASE_INPUT_BITS = 24
MAX_PHASE_DEGREE = 8
MAX_PHASE_TERMS = 256
MAX_PHASE_PRODUCTS = 20_000
MAX_PHASE_PREDICATE_BITS = 8


def expand_phase(
    expression: Expression,
    quantum_word: Callable[[Expression], tuple[Location, ...] | None],
    number: Callable[[Expression], float],
    *,
    signed_word: Callable[[Expression], bool] | None = None,
    condition: Expression | None = None,
    disjoint_condition_target: Location | None = None,
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

    def multiply(
        left: Polynomial, right: Polynomial, node: Expression, *, integer: bool = False
    ) -> Polynomial:
        nonlocal products
        products += len(left) * len(right)
        if products > MAX_PHASE_PRODUCTS:
            reject(node.token, "budget.phase", "phase exceeds 20000 monomial products")
        result: Polynomial = {}
        for first, first_value in left.items():
            for second, second_value in right.items():
                mask = first | second
                if integer:
                    result[mask] = int(result.get(mask, 0)) + int(first_value) * int(
                        second_value
                    )
                else:
                    result[mask] = result.get(mask, 0.0) + first_value * second_value
        return checked(result, node)

    def visit(node: Expression, *, integer: bool = False) -> Polynomial:
        word = quantum_word(node)
        if word is not None:
            result: Polynomial = {}
            signed = signed_word is not None and signed_word(node)
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
                weight = 1 << bit
                if signed and bit == len(word) - 1:
                    weight = -weight
                result[mask] = (
                    int(result.get(mask, 0)) + weight
                    if integer
                    else result.get(mask, 0.0) + float(weight)
                )
            return result
        text = node.token.text
        if integer and text in {"%", "&", "|", "^", "min", "max"}:
            left, right = (visit(child, integer=True) for child in node.children)
            if len(locations) > MAX_PHASE_PREDICATE_BITS:
                reject(
                    node.token,
                    "budget.phase",
                    "phase predicate exceeds eight input bits",
                )
            if text == "%":
                divisor = node.children[1]
                if (
                    divisor.children
                    or not divisor.token.text.isdecimal()
                    or int(divisor.token.text) < 1
                    or int(divisor.token.text) & (int(divisor.token.text) - 1)
                ):
                    reject(
                        node.token,
                        "modulo",
                        "predicate modulo requires a positive power-of-two literal divisor",
                    )
            values = []
            for basis in range(1 << len(locations)):
                a, b = integer_value(left, basis), integer_value(right, basis)
                if text not in {"min", "max"} and (a < 0 or b < 0):
                    reject(
                        node.token,
                        "type",
                        "predicate bitwise and modulo operands must be unsigned",
                    )
                if text == "min":
                    evaluated = min(a, b)
                elif text == "max":
                    evaluated = max(a, b)
                elif text == "%":
                    evaluated = a % b
                else:
                    evaluated = {"&": and_, "|": or_, "^": xor}[text](a, b)
                values.append(evaluated)
            for bit in range(len(locations)):
                for basis in range(len(values)):
                    if basis & (1 << bit):
                        values[basis] -= values[basis ^ (1 << bit)]
            return checked(
                {mask: float(value) for mask, value in enumerate(values)}, node
            )
        if text == "-" and len(node.children) == 1:
            return {
                mask: -value
                for mask, value in visit(node.children[0], integer=integer).items()
            }
        if text in {"+", "-", "*", "/", "**"}:
            left = visit(node.children[0], integer=integer)
            if text == "**":
                exponent = node.children[1]
                if (
                    not exponent.children
                    and exponent.token.text.isdecimal()
                    and 0 <= int(exponent.token.text) <= 8
                ):
                    degree = int(exponent.token.text)
                elif not any(left):
                    value = number(node)
                    if integer and not value.is_integer():
                        reject(
                            node.token,
                            "control",
                            "phase predicates require integer arithmetic",
                        )
                    return checked({0: int(value) if integer else value}, node)
                else:
                    reject(
                        exponent.token,
                        "phase",
                        "quantum power requires an integer literal exponent in 0..8",
                    )
                result = {0: 1 if integer else 1.0}
                for _ in range(degree):
                    result = multiply(result, left, node, integer=integer)
                return result
            right = visit(node.children[1], integer=integer)
            if text == "*":
                return multiply(left, right, node, integer=integer)
            if text == "/":
                if integer:
                    reject(
                        node.token,
                        "control",
                        "phase predicates require integer arithmetic without division",
                    )
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
                result[mask] = (
                    (int(result.get(mask, 0)) + sign * int(value))
                    if integer
                    else result.get(mask, 0.0) + sign * value
                )
            return checked(result, node)
        value = number(node)
        if integer and not value.is_integer():
            reject(node.token, "control", "phase predicates require integer arithmetic")
        return checked({0: int(value) if integer else value}, node)

    def integer_value(polynomial: Polynomial, basis: int) -> int:
        return sum(
            int(value) for mask, value in polynomial.items() if mask & basis == mask
        )

    def predicate(node: Expression) -> Callable[[int], bool]:
        comparisons: dict[str, Callable[[int, int], bool]] = {
            "==": eq,
            "!=": ne,
            "<": lt,
            "<=": le,
            ">": gt,
            ">=": ge,
        }
        text = node.token.text
        if not node.children and text in {"0", "1"}:
            return lambda basis: text == "1"
        if text in comparisons:
            left, right = (visit(child, integer=True) for child in node.children)
            comparison = comparisons[text]
            return lambda basis: comparison(
                integer_value(left, basis), integer_value(right, basis)
            )
        if text in {"and", "&", "|", "^"}:
            left_predicate, right_predicate = (
                predicate(child) for child in node.children
            )
            if text in {"and", "&"}:
                return lambda basis: left_predicate(basis) and right_predicate(basis)
            if text == "|":
                return lambda basis: left_predicate(basis) or right_predicate(basis)
            return lambda basis: left_predicate(basis) != right_predicate(basis)
        if text == "~":
            operand = predicate(node.children[0])
            return lambda basis: not operand(basis)
        locations_for_word = quantum_word(node)
        if locations_for_word is None or len(locations_for_word) != 1:
            reject(
                node.token,
                "control",
                "phase predicate requires a comparison or scalar bit",
            )
        word = visit(node, integer=True)
        return lambda basis: bool(integer_value(word, basis))

    guard: Polynomial | None = None
    if condition is not None:
        evaluate = predicate(condition)
        if disjoint_condition_target in locations:
            reject(
                condition.token,
                "predicate.alias",
                "predicate cannot read its XOR target",
            )
        if len(locations) > MAX_PHASE_PREDICATE_BITS:
            reject(
                condition.token,
                "budget.phase",
                "phase predicate exceeds eight input bits",
            )
        coefficients = [int(evaluate(basis)) for basis in range(1 << len(locations))]
        for bit in range(len(locations)):
            for basis in range(len(coefficients)):
                if basis & (1 << bit):
                    coefficients[basis] -= coefficients[basis ^ (1 << bit)]
        guard = checked(
            {mask: float(value) for mask, value in enumerate(coefficients)}, condition
        )
    polynomial = visit(expression)
    if guard is not None:
        polynomial = multiply(guard, polynomial, expression)
    return polynomial, tuple(locations)

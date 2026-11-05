"""Bounded source decisions, disjoint cubes and existing clean Boolean factors."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from operator import eq, ge, gt, le, lt, ne
from typing import TYPE_CHECKING

from ...compiler._construction.register_profile import register_type
from ...compiler._hybrid.model import Value, ValueId
from .phase import Location, expand_phase
from .phase_factors import Cube
from .syntax import Expression, Token, reject

if TYPE_CHECKING:
    from .adapter import _Adapt

MAX_DECISION_BITS = 12
MAX_DECISION_STEPS = 262144
MAX_DECISION_NODES = 4096
MAX_DECISION_CUBES = 128
LOCAL_DECISION_BITS = 8


@dataclass(frozen=True)
class Decision:
    locations: tuple[Location, ...]
    cubes: tuple[Cube, ...]
    nodes: int
    evaluator_steps: int


def decompose_predicate(
    expression: Expression,
    quantum_word: Callable[[Expression], tuple[Location, ...] | None],
    number: Callable[[Expression], float],
) -> Decision:
    """Reduce bounded ordered cofactors; accepting paths are pairwise disjoint."""
    locations: list[Location] = []
    steps = 0

    def counted(
        evaluate: Callable[[int], int], node: Expression
    ) -> Callable[[int], int]:
        def call(basis: int) -> int:
            nonlocal steps
            steps += 1
            if steps > MAX_DECISION_STEPS:
                reject(
                    node.token,
                    "budget.decision",
                    "decision evaluator exceeds 262144 steps",
                )
            value = evaluate(basis)
            if abs(value).bit_length() > 256:
                reject(
                    node.token, "budget.decision", "decision integer exceeds 256 bits"
                )
            return value

        return call

    def compile_value(node: Expression) -> Callable[[int], int]:
        word = quantum_word(node)
        if word is not None:
            indices = []
            for location in word:
                if location not in locations:
                    locations.append(location)
                indices.append(locations.index(location))
                if len(locations) > MAX_DECISION_BITS:
                    reject(
                        node.token,
                        "budget.decision",
                        "decision predicate exceeds twelve input bits",
                    )
            positions = tuple(indices)
            return counted(
                lambda basis: sum(
                    ((basis >> source) & 1) << bit
                    for bit, source in enumerate(positions)
                ),
                node,
            )
        text = node.token.text
        if not node.children:
            value = number(node)
            if not math.isfinite(value) or not value.is_integer():
                reject(
                    node.token,
                    "decision.type",
                    "decision constants require exact integers",
                )
            return counted(lambda basis: int(value), node)
        if text in {"-", "~"} and len(node.children) == 1:
            operand = compile_value(node.children[0])

            def unary(basis: int) -> int:
                value = operand(basis)
                if text == "-":
                    return -value
                if value not in (0, 1):
                    reject(
                        node.token,
                        "decision.type",
                        "decision complement requires a Boolean operand",
                    )
                return 1 - value

            return counted(unary, node)
        if len(node.children) != 2 or text not in {
            "+",
            "-",
            "*",
            "**",
            "%",
            "&",
            "|",
            "^",
            "==",
            "!=",
            "<",
            "<=",
            ">",
            ">=",
        }:
            reject(
                node.token, "decision.type", "unsupported bounded decision arithmetic"
            )
        if text in {"%", "**"}:
            literal = node.children[1]
            if literal.children or not literal.token.text.isdecimal():
                reject(
                    node.token,
                    "decision.type",
                    "decision powers/modulo require a literal",
                )
            value = int(literal.token.text)
            if (text == "%" and (value < 1 or value & (value - 1))) or (
                text == "**" and not 1 <= value <= 8
            ):
                reject(
                    node.token,
                    "decision.type",
                    "decision modulo needs a power of two; powers need 1..8",
                )
        left, right = (compile_value(child) for child in node.children)

        def binary(basis: int) -> int:
            a, b = left(basis), right(basis)
            if text in {"%", "&", "|", "^"} and (a < 0 or b < 0):
                reject(
                    node.token,
                    "decision.type",
                    "decision bitwise/modulo operands must be unsigned",
                )
            if text == "+":
                return a + b
            if text == "-":
                return a - b
            if text == "*":
                return a * b
            if text == "**":
                return int(a**b)
            if text == "%":
                return a % b
            if text == "&":
                return a & b
            if text == "|":
                return a | b
            if text == "^":
                return a ^ b
            return int(
                {"==": eq, "!=": ne, "<": lt, "<=": le, ">": gt, ">=": ge}[text](a, b)
            )

        return counted(binary, node)

    evaluate = compile_value(expression)
    width = len(locations)
    unique: dict[tuple[int, int, int], int] = {}
    nodes: list[tuple[int, int, int]] = [(-1, 0, 0), (-1, 1, 1)]

    def node(bit: int, low: int, high: int) -> int:
        if low == high:
            return low
        key = bit, low, high
        if key not in unique:
            if len(unique) >= MAX_DECISION_NODES:
                reject(
                    expression.token,
                    "budget.decision",
                    "decision graph exceeds 4096 nodes",
                )
            unique[key] = len(nodes)
            nodes.append(key)
        return unique[key]

    def leaf_tree(bit: int, values: tuple[int, ...]) -> int:
        if len(values) == 1:
            return values[0]
        return node(
            bit, leaf_tree(bit + 1, values[::2]), leaf_tree(bit + 1, values[1::2])
        )

    def cofactor(bit: int, prefix: int) -> int:
        if width - bit > LOCAL_DECISION_BITS:
            return node(
                bit, cofactor(bit + 1, prefix), cofactor(bit + 1, prefix | (1 << bit))
            )
        values = tuple(
            evaluate(prefix | (suffix << bit)) for suffix in range(1 << (width - bit))
        )
        if any(value not in (0, 1) for value in values):
            reject(
                expression.token,
                "decision.type",
                "decision predicate must return one Boolean bit",
            )
        return leaf_tree(bit, values)

    root = cofactor(0, 0)
    cubes: list[Cube] = []

    def paths(index: int, prefix: Cube) -> None:
        if index == 0:
            return
        if index == 1:
            if len(cubes) >= MAX_DECISION_CUBES:
                reject(
                    expression.token,
                    "budget.decision",
                    "decision graph exceeds 128 accepting paths",
                )
            cubes.append(prefix)
            return
        bit, low, high = nodes[index]
        paths(low, (*prefix, (locations[bit], False)))
        paths(high, (*prefix, (locations[bit], True)))

    paths(root, ())
    return Decision(tuple(locations), tuple(cubes), len(unique), steps)


def lower_decision_phase(
    owner: _Adapt,
    token: Token,
    expression: Expression,
    coefficient: Expression,
    condition: Expression,
    quantum_word: Callable[[Expression], tuple[Location, ...] | None],
) -> None:
    """Reuse owned Boolean compute/phase/uncompute and clean-tail scheduling."""
    decision = decompose_predicate(condition, quantum_word, owner.classical_number)
    polynomial, locations = expand_phase(
        expression, quantum_word, owner.classical_number
    )
    angle = owner.classical_number(coefficient)
    for value, bit in tuple(dict.fromkeys((*decision.locations, *locations))):
        owner.effect_op(
            "qreg.control_phase", token, (value,), indices=(bit,), angle=0.0
        )
    for cube in decision.cubes:
        flags = []
        for start in range(0, len(cube), 6):
            value = Value(ValueId(owner.scope, owner.counter), register_type(1))
            owner.counter += 1
            owner.phase_factor(token, cube[start : start + 6], value)
            flags.append(value)
        for mask, weight in sorted(polynomial.items()):
            selected = tuple((flag, 0) for flag in flags) + tuple(
                location for bit, location in enumerate(locations) if mask & (1 << bit)
            )
            radians = angle * weight
            if len(selected) > 8 or not math.isfinite(radians) or abs(radians) > 2**53:
                reject(
                    token,
                    "budget.decision",
                    "decision phase exceeds existing indexed width or finite angle budget",
                )
            if selected:
                owner.effect_op(
                    "qreg.control_phase",
                    token,
                    tuple(v for v, _ in selected),
                    indices=tuple(i for _, i in selected),
                    angle=radians,
                )
            else:
                owner.effect_op("qreg.phase", token, (), angle=radians)
        for flag in reversed(flags):
            owner.effect_op("qreg.uncompute", token, (flag,))
            owner.effect_op("qreg.release", token, (flag,))

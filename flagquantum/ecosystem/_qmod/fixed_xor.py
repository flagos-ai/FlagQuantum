"""Lower a bounded exact arithmetic word XOR using the destination bit scale."""

from __future__ import annotations

from fractions import Fraction
from typing import TYPE_CHECKING

from .angles import REAL_LITERAL
from .computed_rotation import exact_expression
from .masked_selection import emit_truth_xor
from .syntax import Expression, Token, reject

if TYPE_CHECKING:
    from .adapter import _Adapt
    from .phase import Location


def lower_fixed_xor(
    owner: _Adapt,
    token: Token,
    destination: Expression,
    expression: Expression,
    guard: Expression | None = None,
) -> bool:
    target = owner.parameters.get(destination.token.text)
    if destination.children or target is None or target.kind != "qphase":
        return False
    targets = owner.array_view(destination, allow_phase=True)
    if len(targets) > 8:
        reject(
            destination.token,
            "budget.precision",
            "fixed-point XOR target needs at most eight bits",
        )

    constants: dict[Expression, Fraction] = {}
    operands: dict[str, tuple[tuple[Location, ...], int]] = {}

    def collect(node: Expression) -> None:
        text = node.token.text
        if (
            not node.children
            and (parameter := owner.parameters.get(text)) is not None
            and parameter.kind in {"qnum", "qphase"}
        ):
            if text in owner.signed or parameter.fractional_bits:
                reject(
                    node.token,
                    "precision",
                    "bounded fixed XOR requires unsigned operands",
                )
            view = owner.array_view(node, allow_phase=parameter.kind == "qphase")
            operands[text] = view, len(view) if parameter.kind == "qphase" else 0
            return
        if not node.children and REAL_LITERAL.fullmatch(text):
            value = Fraction(text)
        elif (
            text == "["
            and node.children[0].token.text in owner.classical_arrays
            or not node.children
            and text in owner.classical_values
        ):
            value = Fraction(str(owner.classical_number(node)))
        else:
            for child in node.children:
                collect(child)
            return
        if value.denominator & (value.denominator - 1):
            reject(
                node.token,
                "precision",
                "fixed XOR coefficients must be exact binary rationals",
            )
        if max(value.numerator.bit_length(), value.denominator.bit_length()) > 256:
            reject(node.token, "precision", "fixed XOR coefficient exceeds 256 bits")
        constants[node] = value

    collect(expression)
    locations = tuple(
        dict.fromkeys(loc for view, _ in operands.values() for loc in view)
    )
    selected: tuple[Location, ...] = ()
    expected = 0
    if guard is not None:
        if guard.token.text != "==" or len(guard.children) != 2:
            reject(
                guard.token,
                "precision",
                "bounded fixed XOR requires one static equality guard",
            )
        selected = owner.array_view(guard.children[0])
        expected = owner.static_integer(guard.children[1])
        if not 0 <= expected < 1 << len(selected):
            reject(guard.token, "width", "fixed XOR guard value does not fit")
        locations = tuple(dict.fromkeys((*locations, *selected)))
    if len(locations) > 7:
        reject(
            token, "budget.precision", "fixed-point XOR needs at most seven read bits"
        )
    if set(locations).intersection(targets):
        reject(token, "alias", "fixed-point XOR reads must be disjoint from the target")

    evaluate = exact_expression(expression, set(operands), constants)
    positions = {location: bit for bit, location in enumerate(locations)}

    def word(basis: int, view: tuple[Location, ...]) -> int:
        return sum(
            (basis >> positions[location] & 1) << bit
            for bit, location in enumerate(view)
        )

    table = []
    for basis in range(1 << len(locations)):
        if selected and word(basis, selected) != expected:
            table.append(0)
            continue
        inputs = {
            name: Fraction(word(basis, view), 1 << scale)
            for name, (view, scale) in operands.items()
        }
        exact = evaluate(inputs) * (1 << len(targets))
        table.append((exact.numerator // exact.denominator) & ((1 << len(targets)) - 1))

    checked = set()
    for value, bit in (*locations, *targets):
        if value not in checked:
            owner.effect_op(
                "qreg.control_phase", token, (value,), indices=(bit,), angle=0.0
            )
            checked.add(value)
    for bit, location in enumerate(targets):
        emit_truth_xor(
            owner, token, locations, location, [word >> bit & 1 for word in table]
        )
    return True

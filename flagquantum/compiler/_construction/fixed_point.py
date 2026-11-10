"""Bounded fixed-point arithmetic shared by source frontends.

The closed profile preserves the existing eight-bit fractional arithmetic,
twenty-bit literal staging, twenty-bit words and five independent branch bits.
Source adapters retain literal recognition, ownership and profile admission.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from fractions import Fraction
from itertools import product
from typing import Generic, NoReturn, TypeVar

FRACTION_BITS = 8
LITERAL_BITS = 20
MAX_BRANCH_BITS = 5
BRANCH_ROWS = 1 << MAX_BRANCH_BITS
_Node = TypeVar("_Node")


@dataclass(frozen=True)
class FixedPointScalar:
    width: int
    signed: bool
    fraction: int
    low: Fraction
    high: Fraction
    values: tuple[Fraction, ...]


def truncate_fixed_point(value: Fraction, bits: int) -> Fraction:
    scale = 1 << bits
    return Fraction((value * scale).__floor__(), scale)


def fixed_point_number(
    fail: Callable[[str, str], NoReturn], value: Fraction
) -> FixedPointScalar:
    literal_scale = 1 << LITERAL_BITS
    native = Fraction(int(value * literal_scale), literal_scale)
    quantized = truncate_fixed_point(native, FRACTION_BITS)
    fraction = quantized.denominator.bit_length() - 1
    return fixed_point_scalar(
        fail, quantized, quantized, fraction, (quantized,) * BRANCH_ROWS
    )


def fixed_point_scalar(
    fail: Callable[[str, str], NoReturn],
    low: Fraction,
    high: Fraction,
    fraction: int,
    values: tuple[Fraction, ...],
) -> FixedPointScalar:
    if len(values) != BRANCH_ROWS:
        fail("budget.precision", "quantized domain requires thirty-two branch rows")
    if low > high or fraction < 0:
        fail("precision", "invalid quantized interval or fractional scale")
    fraction = min(fraction, FRACTION_BITS)
    scale = 1 << fraction
    lower, upper = int(low * scale), int(high * scale)
    signed = lower < 0
    if signed:
        width = max((~lower).bit_length(), max(upper, 0).bit_length()) + 1
    else:
        width = upper.bit_length()
    width = max(width, fraction, 1)
    if width > 20:
        fail("budget.precision", "quantized output format exceeds twenty bits")
    if any(
        value * scale != int(value * scale) or value < low or value > high
        for value in values
    ):
        fail("precision", "quantized result does not fit the inferred format")
    return FixedPointScalar(width, signed, fraction, low, high, values)


class FixedPointBranches:
    """Track prepared selectors and prove a retained discard environment.

    Scalar rows always refer to this one joint selector domain, including when
    values are passed through helpers. Word width never creates independent
    branches. Adapters own lexical binding and must keep their scalars in the
    same domain; this object neither allocates qubits nor erases dropped data.
    """

    def __init__(self) -> None:
        self.bits = 0
        self._discarded: list[FixedPointScalar] = []

    def uniform(self, fail: Callable[[str, str], NoReturn]) -> FixedPointScalar:
        if self.bits == MAX_BRANCH_BITS:
            fail(
                "budget.precision",
                "quantized preparation exceeds five independent basis bits",
            )
        value = fixed_point_scalar(
            fail,
            Fraction(0),
            Fraction(1),
            0,
            tuple(Fraction(row >> self.bits & 1) for row in range(BRANCH_ROWS)),
        )
        self.bits += 1
        return value

    def discard(self, value: FixedPointScalar) -> None:
        self._discarded.append(value)

    def verify_discarded(self, fail: Callable[[str, str], NoReturn]) -> None:
        rows = 1 << self.bits
        labels = {
            tuple(value.values[row] for value in self._discarded) for row in range(rows)
        }
        if len(labels) != rows:
            fail(
                "ownership",
                "dropped numeric state must distinguish every prepared branch",
            )


class FixedPointExpression(Generic[_Node]):
    """Evaluate admitted expressions without depending on a source AST.

    Adapters provide source accessors and source-located rejection. Literal
    callbacks return unquantized finite Fractions; arithmetic applies the
    explicit closed quantization profile. No source functions are executed.
    """

    def __init__(
        self,
        *,
        text: Callable[[_Node], str],
        children: Callable[[_Node], tuple[_Node, ...]],
        literal: Callable[[_Node], Fraction | None],
        reject: Callable[[_Node, str, str], NoReturn],
    ) -> None:
        self._text, self._children = text, children
        self._literal, self._reject = literal, reject

    def evaluate(
        self, node: _Node, values: Mapping[str, FixedPointScalar]
    ) -> FixedPointScalar:
        literal = self._literal(node)
        if literal is not None:
            return fixed_point_number(
                lambda code, message: self._reject(node, code, message), literal
            )
        name, args = self._text(node), self._children(node)
        if not args and name in values:
            return values[name]
        if name == "-" and len(args) == 1:
            operand = self.evaluate(args[0], values)
            return fixed_point_scalar(
                lambda code, message: self._reject(node, code, message),
                -operand.high,
                -operand.low,
                operand.fraction,
                tuple(-value for value in operand.values),
            )
        if name == "**" and len(args) == 2:
            base = self.evaluate(args[0], values)
            degree = self._literal(args[1])
            if degree is None or degree.denominator != 1 or not 1 <= degree <= 4:
                self._reject(
                    node,
                    "precision",
                    "quantized power requires a literal degree in 1..4",
                )
            exponent = int(degree)
            fraction = min(base.fraction * exponent, FRACTION_BITS)
            edges = [base.low**exponent, base.high**exponent]
            if exponent % 2 == 0 and base.low < 0 < base.high:
                edges.append(Fraction())
            return fixed_point_scalar(
                lambda code, message: self._reject(node, code, message),
                truncate_fixed_point(min(edges), fraction),
                truncate_fixed_point(max(edges), fraction),
                fraction,
                tuple(
                    truncate_fixed_point(value**exponent, fraction)
                    for value in base.values
                ),
            )
        if name in {"+", "-", "*"} and len(args) == 2:
            left, right = (self.evaluate(child, values) for child in args)
            pairs = tuple(zip(left.values, right.values, strict=True))
            fraction = max(left.fraction, right.fraction)
            if name == "+":
                return fixed_point_scalar(
                    lambda code, message: self._reject(node, code, message),
                    left.low + right.low,
                    left.high + right.high,
                    fraction,
                    tuple(a + b for a, b in pairs),
                )
            if name == "-":
                return fixed_point_scalar(
                    lambda code, message: self._reject(node, code, message),
                    left.low - right.high,
                    left.high - right.low,
                    fraction,
                    tuple(a - b for a, b in pairs),
                )
            fraction = min(left.fraction + right.fraction, FRACTION_BITS)
            bounds = [
                a * b
                for a, b in product((left.low, left.high), (right.low, right.high))
            ]
            return fixed_point_scalar(
                lambda code, message: self._reject(node, code, message),
                truncate_fixed_point(min(bounds), fraction),
                truncate_fixed_point(max(bounds), fraction),
                fraction,
                tuple(truncate_fixed_point(a * b, fraction) for a, b in pairs),
            )
        self._reject(
            node, "precision", "unsupported lexical quantized numeric expression"
        )

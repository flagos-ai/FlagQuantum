# SPDX-License-Identifier: Apache-2.0
# Adapted from locked Kickmix fixed_precision_angle_128.cc/.h.
# https://github.com/quantumlib/Qualtran/blob/9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453/kickmix/src/kickmix/util/fixed_precision_angle_128.cc

"""Exact compiler-owned phases in the source's 128-bit full-turn ring."""

import re
from dataclasses import dataclass
from fractions import Fraction
from math import floor, isfinite, ldexp

_WORD_MASK = (1 << 64) - 1
_ANGLE_MASK = (1 << 128) - 1


@dataclass(frozen=True)
class PhaseAngle128:
    """An immutable angle: 2*pi*(low/2**128 + high/2**64).

    Arithmetic wraps modulo a full turn. Factories and decimal rounding
    retain the locked source algorithm. Numerical conversion is an explicit
    FlagQuantum boundary and never replaces the exact stored words.
    """

    low: int = 0
    high: int = 0

    def __post_init__(self) -> None:
        if any(
            type(word) is not int or not 0 <= word <= _WORD_MASK
            for word in (self.low, self.high)
        ):
            raise ValueError("Phase words must be unsigned 64-bit integers")

    @property
    def units(self) -> int:
        return self.low | self.high << 64

    @classmethod
    def _from_units(cls, units: int) -> "PhaseAngle128":
        return cls(units & _WORD_MASK, (units >> 64) & _WORD_MASK)

    def __bool__(self) -> bool:
        return bool(self.low or self.high)

    def clifford_hierarchy_level(self) -> int:
        value = self.units
        return 0 if not value else 129 - (value & -value).bit_length()

    @classmethod
    def from_power_of_two_half_turns(cls, exponent: int) -> "PhaseAngle128":
        if type(exponent) is not int:
            raise ValueError("Phase exponent must be an integer")
        return (
            cls()
            if exponent > 0 or exponent <= -128
            else cls._from_units(1 << (exponent + 127))
        )

    @classmethod
    def from_half_turns_exact_double(cls, value: float) -> "PhaseAngle128":
        """Mirror the source conversion, including its subnormal underflow edge."""
        if not isfinite(value):
            raise ValueError("Angle cannot be NaN or infinite")
        value *= 0.5
        negative = value < 0
        absolute = abs(value)
        fraction = absolute - floor(absolute)
        high_float = ldexp(fraction, 64)
        high = floor(high_float)
        low_float = ldexp(high_float - high, 64)
        low = floor(low_float)
        if low_float != low:
            raise ValueError("Double has bits below the 128-bit full-turn grid")
        angle = cls(low, high)
        return -angle if negative else angle

    @classmethod
    def from_rounded_decimal_half_turns(cls, text: str) -> "PhaseAngle128":
        """Keep the source's 128-digit sticky prefix and ties-to-even rounding."""
        if re.fullmatch(r"[-+]?[0-9]+\.[0-9]+", text) is None:
            raise ValueError(
                "Decimal phase requires signed digits, a dot and fractional digits"
            )
        negative = text.startswith("-")
        unsigned = text[1:] if text[0] in "+-" else text
        integer, fraction = unsigned.split(".")
        odd = (ord(integer[-1]) - ord("0")) & 1
        fraction = fraction.rstrip("0")
        if not fraction:
            return cls._from_units(odd << 127)
        sticky = len(fraction) > 128
        # This integer packs the source's eight little-endian 64-bit limbs.
        # A digit step performs their same high-to-low long division by ten.
        accumulator = (5 << 512) // 10 if sticky else 0
        for digit in reversed(fraction[:128]):
            accumulator = (((ord(digit) - ord("0")) << 512) | accumulator) // 10
        units = (odd << 127) | (accumulator >> 385)
        half_bit = (accumulator >> 384) & 1
        sticky_low = bool(accumulator & ((1 << 384) - 1))
        if half_bit and (sticky_low or units & 1):
            units += 1
        angle = cls._from_units(units)
        return -angle if negative else angle

    def to_decimal_half_turns(self) -> str:
        value = self.units
        integer = value >> 127
        remainder = (value << 1) & _ANGLE_MASK
        if not remainder:
            return f"{integer}.0"
        digits = []
        while remainder:
            digit, remainder = divmod(remainder * 10, 1 << 128)
            digits.append(str(digit))
        return f"{integer}." + "".join(digits)

    def __str__(self) -> str:
        return self.to_decimal_half_turns() + "π"

    def __add__(self, other: "PhaseAngle128") -> "PhaseAngle128":
        return self._from_units(self.units + other.units)

    def __neg__(self) -> "PhaseAngle128":
        return self._from_units(-self.units)

    def __mul__(self, factor: int) -> "PhaseAngle128":
        if type(factor) is not int or not 0 <= factor <= _WORD_MASK:
            raise ValueError("Phase multiplier must be an unsigned 64-bit integer")
        return self._from_units(self.units * factor)

    @classmethod
    def from_integer_half_turns(cls, value: int) -> "PhaseAngle128":
        return cls._from_units((value & 1) << 127)

    def plus_integer_half_turns(self, value: int) -> "PhaseAngle128":
        return self._from_units(self.units ^ ((value & 1) << 127))

    def rotated_half_turn(self) -> "PhaseAngle128":
        return self._from_units(self.units ^ (1 << 127))

    def closer_to_half_turn_than_zero(self) -> bool:
        return self.high >> 62 in (1, 2)

    def is_multiple_of_45_degrees(self) -> bool:
        return self.units % (1 << 125) == 0

    def is_multiple_of_90_degrees(self) -> bool:
        return self.units % (1 << 126) == 0

    def is_multiple_of_180_degrees(self) -> bool:
        return self.units % (1 << 127) == 0

    def half_turns(self) -> Fraction:
        """Return the stored value exactly, before a numerical conversion."""
        return Fraction(self.units, 1 << 127)

    def signed_half_turns_float(self) -> float:
        """Convert the representative in [-1, 1) at a numerical boundary."""
        units = self.units
        if units >= 1 << 127:
            units -= 1 << 128
        return units / (1 << 127)

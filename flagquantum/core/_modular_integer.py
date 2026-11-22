# SPDX-License-Identifier: Apache-2.0
# Adapted from locked Kickmix util/mod_int.cc/.h at
# 9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453.

"""Immutable modular scalar snapshots retaining source word operation order."""

from __future__ import annotations

from dataclasses import dataclass

from ._fixed_width_integer import FixedWidthInteger
from ._fixed_width_modular import double_modulo, halve_modulo, remainder_into
from ._word_operations import WORD_MASK


@dataclass(frozen=True, eq=False)
class ModularInteger:
    """Own a raw value and share an immutable modulus without eager reduction."""

    value: FixedWidthInteger
    modulus: FixedWidthInteger

    @classmethod
    def zero(cls, modulus: FixedWidthInteger) -> ModularInteger:
        return cls(FixedWidthInteger(modulus.width), modulus)

    @classmethod
    def from_signed(cls, value: int, modulus: FixedWidthInteger) -> ModularInteger:
        word = FixedWidthInteger.from_integer(modulus.width, value, signed=True)
        if value < 0:
            return cls(word.negate_into(), modulus).negate()
        return cls(word, modulus)

    @classmethod
    def from_text(cls, text: str, modulus: FixedWidthInteger) -> ModularInteger:
        reduced = remainder_into(FixedWidthInteger.from_text(text), modulus)
        return cls(FixedWidthInteger(modulus.width).xor_word(reduced), modulus)

    @classmethod
    def from_word(
        cls,
        value: FixedWidthInteger,
        modulus: FixedWidthInteger,
        *,
        moved: bool = False,
    ) -> ModularInteger:
        if not moved and (value.width < modulus.width or value < modulus):
            return cls(FixedWidthInteger(modulus.width).xor_word(value), modulus)
        # The source rvalue constructor owns a copy before mutating remainder.
        result = FixedWidthInteger(value.width, value.value)
        if result >= modulus:
            result = remainder_into(result, modulus)
        if result.word_count != modulus.word_count:
            result = FixedWidthInteger(modulus.width).xor_word(result)
        return cls(FixedWidthInteger(modulus.width, result.value), modulus)

    def verify_invariants(self) -> None:
        if self.modulus.width != self.modulus.bits_in_use:
            raise ValueError("Modulus must fill its declared width")
        if self.value >= self.modulus:
            raise ValueError("Modular value must be smaller than its modulus")
        if self.value.width != self.modulus.width:
            raise ValueError("Modular value and modulus widths must match")

    def __bool__(self) -> bool:
        return bool(self.value)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ModularInteger):
            return NotImplemented
        return self.value == other.value

    def __hash__(self) -> int:
        return hash(self.value)

    def increment(self) -> ModularInteger:
        result = self.value.increment()[0]
        if result == self.modulus:
            result = FixedWidthInteger(self.value.width)
        return ModularInteger(result, self.modulus)

    def add(self, other: ModularInteger | int) -> ModularInteger:
        if isinstance(other, int):
            FixedWidthInteger.from_integer(0, other, signed=True)
            result, carry = self.value.add_scalar_carry(other & WORD_MASK)
        else:
            result, carry = self.value.add_word(other.value)
        if carry or result >= self.modulus:
            result = result.subtract_word(self.modulus)[0]
        return ModularInteger(result, self.modulus)

    def subtract(self, other: ModularInteger) -> ModularInteger:
        result, borrow = self.value.subtract_word(other.value)
        if borrow:
            result = result.add_word(self.modulus)[0]
        return ModularInteger(result, self.modulus)

    def negate(self) -> ModularInteger:
        result = self.value
        if result:
            result = result.xor_word(-1).add_word(self.modulus, True)[0]
        return ModularInteger(result, self.modulus)

    def double(self) -> ModularInteger:
        return ModularInteger(double_modulo(self.value, self.modulus), self.modulus)

    def halve(self) -> ModularInteger:
        return ModularInteger(halve_modulo(self.value, self.modulus), self.modulus)

    def multiply(self, other: ModularInteger | int) -> ModularInteger:
        left = self.value
        if isinstance(other, int):
            FixedWidthInteger.from_integer(0, other, signed=True)
            if other == -(1 << 63):
                raise ValueError("Source signed minimum multiplication overflows")
            if other < 0:
                left = self.negate().value
            product = FixedWidthInteger(self.modulus.width + 64).accumulate_product(
                left, abs(other)
            )
        else:
            product = FixedWidthInteger(self.modulus.width * 2).accumulate_product(
                left, other.value
            )
        result = remainder_into(product, self.modulus)
        capacity_mask = (1 << (64 * self.value.word_count)) - 1
        return ModularInteger(
            FixedWidthInteger(self.value.width, result.value & capacity_mask),
            self.modulus,
        )

    def squared(self) -> ModularInteger:
        product = FixedWidthInteger(self.modulus.width * 2).accumulate_product(
            self.value, self.value
        )
        result = remainder_into(product, self.modulus)
        word = FixedWidthInteger(self.modulus.width, result.value)
        return self.from_word(word, self.modulus, moved=True)

    def divide(
        self, other: ModularInteger | int, *, moved: bool = False
    ) -> ModularInteger:
        if isinstance(other, int):
            other = self.from_signed(other, self.modulus)
        if not self.modulus.bit(0):
            raise ValueError("Source division requires an odd modulus")
        left, right = other.value, self.modulus
        left_cofactor = (
            ModularInteger(FixedWidthInteger(), self.modulus)
            if moved and other is self
            else self
        )
        right_cofactor = self.zero(self.modulus)
        if not left.bit(0):
            left, right = right, left
            left_cofactor, right_cofactor = right_cofactor, left_cofactor
        while right:
            if right.bit(0):
                if right < left:
                    left, right = right, left
                    left_cofactor, right_cofactor = right_cofactor, left_cofactor
                right = right.subtract_word(left)[0]
                right_cofactor = right_cofactor.subtract(left_cofactor)
            right = right.shift_right(1)
            right_cofactor = right_cofactor.halve()
        if left != 1:
            raise ValueError("Modular divisor is not coprime with the modulus")
        return left_cofactor

    def inverse(self) -> ModularInteger:
        return ModularInteger(
            FixedWidthInteger(self.modulus.width).xor_word(1), self.modulus
        ).divide(self)

    def shift_left(self, amount: int, *, signed: bool = True) -> ModularInteger:
        FixedWidthInteger.from_integer(0, amount, signed=signed)
        result = self
        for _ in range(abs(amount)):
            result = result.double() if amount > 0 else result.halve()
        return result

    def shift_right(self, amount: int, *, signed: bool = True) -> ModularInteger:
        FixedWidthInteger.from_integer(0, amount, signed=signed)
        result = self
        for _ in range(abs(amount)):
            result = result.halve() if amount > 0 else result.double()
        return result

    def __str__(self) -> str:
        return f"{self.value.decimal()} (mod {self.modulus.decimal()})"

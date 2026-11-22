# SPDX-License-Identifier: Apache-2.0
# Adapted from locked Kickmix util/fixed_width_int.cc/.h.
# Revision 9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453.

"""Declared-width scalar integers with explicit source storage and alias rules."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from struct import pack, unpack
from typing import Literal

from ._fixed_width_products import accumulate_product
from ._unsigned_integer_text import grouped_digits, parse_integer_literal, write_decimal
from ._word_operations import WORD_MASK, add_carry, require_word


@dataclass(frozen=True, eq=False)
class FixedWidthInteger:
    """Own unsigned raw storage, including source-created hanging bits.

    Factory construction clips to width. Direct value construction represents
    raw allocated words, whose capacity is the next multiple of 64 bits.
    All updates return an immutable successor. Bit iteration is LSB-first.
    """

    width: int = 0
    value: int = 0

    def __post_init__(self) -> None:
        if type(self.width) is not int or self.width < 0:
            raise ValueError("Integer width must be a nonnegative integer")
        if type(self.value) is not int or not 0 <= self.value < 1 << (
            64 * self.word_count
        ):
            raise ValueError("Raw integer value must fit its allocated word storage")

    @property
    def word_count(self) -> int:
        return (self.width + 63) // 64

    @property
    def bits_in_use(self) -> int:
        return self.value.bit_length()

    @property
    def mask(self) -> int:
        return (1 << self.width) - 1

    @classmethod
    def from_integer(
        cls, width: int, value: int, *, signed: bool = False
    ) -> FixedWidthInteger:
        if signed:
            if type(value) is not int or not -(1 << 63) <= value < 1 << 63:
                raise ValueError("Signed scalar must fit 64 bits")
        else:
            require_word(value)
        return cls(width, value & ((1 << width) - 1))

    @classmethod
    def from_text(
        cls,
        text: str,
        width: int | None = None,
        *,
        radix: Literal[2, 10, 16] | None = None,
    ) -> FixedWidthInteger:
        value, declared = parse_integer_literal(text, width, radix=radix)
        return cls(declared, value)

    def _fit(self, value: int) -> FixedWidthInteger:
        return FixedWidthInteger(self.width, value & self.mask)

    def clip(self) -> tuple[FixedWidthInteger, bool]:
        return self._fit(self.value), bool(self.value & ~self.mask)

    def resize_clear(self, width: int) -> FixedWidthInteger:
        return FixedWidthInteger(width)

    def resize_shrink(self, width: int) -> FixedWidthInteger:
        if not 0 <= width <= self.width:
            raise ValueError("Shrinking cannot increase declared width")
        return FixedWidthInteger(width, self.value & ((1 << width) - 1))

    def write_from(self, other: FixedWidthInteger) -> FixedWidthInteger:
        return self._fit(0 if other is self else other.value)

    def assign_integer(self, value: int, *, signed: bool = False) -> FixedWidthInteger:
        if not self.word_count and not signed:
            raise ValueError("Source unsigned assignment requires allocated storage")
        return self.from_integer(self.width, value, signed=signed)

    def __bool__(self) -> bool:
        return bool(self.value)

    def __int__(self) -> int:
        if self.bits_in_use > 64:
            raise ValueError("Integer is too large for an unsigned 64-bit scalar")
        return self.value

    def __iter__(self) -> Iterator[bool]:
        return (bool((self.value >> bit) & 1) for bit in range(self.width))

    def bit(self, index: int) -> bool:
        if type(index) is not int or index < 0:
            raise ValueError("Bit index must be nonnegative")
        return (
            bool((self.value >> index) & 1) if index < 64 * self.word_count else False
        )

    def reference_bit(self, index: int) -> bool:
        if not 0 <= index < self.width:
            raise ValueError("Bit reference must lie inside declared width")
        return self.bit(index)

    def set_bit(self, index: int, value: bool = True) -> FixedWidthInteger:
        if type(index) is not int or index < 0:
            raise ValueError("Bit index must be nonnegative")
        if index >= 64 * self.word_count:
            return self
        updated = self.value | (1 << index) if value else self.value & ~(1 << index)
        return FixedWidthInteger(self.width, updated)

    def flip_bit(self, index: int) -> FixedWidthInteger:
        return self.set_bit(index, not self.bit(index))

    def read_shifted_word(self, offset: int) -> int:
        if type(offset) is not int or not -(1 << 63) <= offset <= WORD_MASK:
            raise ValueError(
                "Shifted word offset must fit the source integer overloads"
            )
        if offset >= 1 << 63:
            offset -= 1 << 64
        if offset <= -64 or offset >= 64 * self.word_count:
            return 0
        return (
            self.value >> offset if offset >= 0 else self.value << -offset
        ) & WORD_MASK

    def increment(self, offset: int = 0) -> tuple[FixedWidthInteger, bool]:
        if type(offset) is not int or offset < 0:
            raise ValueError("Increment bit offset must be nonnegative")
        if offset >= self.width:
            return self, True
        total = self.value + (1 << offset)
        return self._fit(total), bool(total >> self.width)

    def decrement(self, offset: int = 0) -> tuple[FixedWidthInteger, bool]:
        if type(offset) is not int or offset < 0:
            raise ValueError("Decrement bit offset must be nonnegative")
        if offset >= self.width:
            return self, True
        total = self.value - (1 << offset)
        return self._fit(total), total < 0 or bool(total & ~self.mask)

    def add_word(
        self, other: FixedWidthInteger, carry: bool = False
    ) -> tuple[FixedWidthInteger, bool]:
        common = 64 * min(self.word_count, other.word_count)
        low_mask = (1 << common) - 1
        total = (self.value & low_mask) + (other.value & low_mask) + carry
        raw = FixedWidthInteger(
            self.width, (self.value & ~low_mask) | (total & low_mask)
        )
        if total >> common:
            return raw.increment(common)
        return raw.clip()

    def add_scalar_carry(
        self, other: int, carry: bool = False
    ) -> tuple[FixedWidthInteger, bool]:
        if not self.word_count:
            raise ValueError("Source scalar carry requires allocated storage")
        low, overflow = add_carry(carry, self.value & WORD_MASK, other)
        raw = FixedWidthInteger(self.width, (self.value & ~WORD_MASK) | low)
        if overflow:
            return raw.increment(0)
        return raw.clip()

    def subtract_word(self, other: FixedWidthInteger) -> tuple[FixedWidthInteger, bool]:
        common = 64 * min(self.word_count, other.word_count)
        low_mask = (1 << common) - 1
        difference = (self.value & low_mask) - (other.value & low_mask)
        raw = FixedWidthInteger(
            self.width, (self.value & ~low_mask) | (difference & low_mask)
        )
        borrow = False
        if difference < 0:
            raw, borrow = raw.decrement(common)
        return raw.clip()[0], borrow

    def add_scalar(
        self, other: int, *, signed: bool = False, subtract: bool = False
    ) -> FixedWidthInteger:
        # Validate the selected C++ scalar overload before applying its word rule.
        self.from_integer(0, other, signed=signed)
        if not self.word_count:
            return self
        old = self.value & WORD_MASK
        unsigned = other & WORD_MASK
        low = (old - unsigned if subtract else old + unsigned) & WORD_MASK
        raw = FixedWidthInteger(self.width, (self.value & ~WORD_MASK) | low)
        if subtract:
            if other >= 0 and low > old:
                raw = raw.decrement(64)[0]
            elif other < 0 and low < old:
                raw = raw.increment(64)[0]
        elif other >= 0 and low < unsigned:
            raw = raw.increment(64)[0]
        elif other < 0 and low > unsigned:
            raw = raw.decrement(64)[0]
        return raw.clip()[0]

    def xor_word(self, other: FixedWidthInteger | int) -> FixedWidthInteger:
        if isinstance(other, FixedWidthInteger):
            common = 64 * min(self.word_count, other.word_count)
            operand = other.value & ((1 << common) - 1)
        else:
            if type(other) is not int or not -(1 << 31) <= other < 1 << 31:
                raise ValueError("Scalar XOR mask must fit a signed 32-bit integer")
            operand = other
        return self._fit(self.value ^ operand)

    def and_word(self, other: FixedWidthInteger) -> FixedWidthInteger:
        common_mask = (1 << (64 * min(self.word_count, other.word_count))) - 1
        return self._fit(
            (self.value & ~common_mask) | (self.value & other.value & common_mask)
        )

    def or_word(self, other: FixedWidthInteger) -> FixedWidthInteger:
        return self._fit(self.value | other.value)

    def _shift_words(self, amount: int, *, left: bool) -> FixedWidthInteger:
        require_word(amount)
        raw = self.value
        indices = reversed(range(self.word_count)) if left else range(self.word_count)
        for index in indices:
            offset = (index * 64 - amount if left else index * 64 + amount) & WORD_MASK
            if offset >= 1 << 63:
                offset -= 1 << 64
            word = FixedWidthInteger(self.width, raw).read_shifted_word(offset)
            position = index * 64
            raw = (raw & ~(WORD_MASK << position)) | (word << position)
        return self._fit(raw)

    def shift_left(self, amount: int) -> FixedWidthInteger:
        if amount < 0:
            return self.shift_right(-amount)
        return self._shift_words(amount, left=True)

    def shift_right(self, amount: int, *, signed: bool = False) -> FixedWidthInteger:
        if amount < 0:
            return self.shift_left(-amount)
        result = self._shift_words(amount, left=False)
        if signed and self.width and self.bit(self.width - 1):
            fill = min(amount, self.width)
            result = result._fit(
                result.value | (((1 << fill) - 1) << (self.width - fill))
            )
        return result

    def multiply_into(self, other: FixedWidthInteger | int) -> FixedWidthInteger:
        if isinstance(other, int):
            require_word(other)
            return self._fit(self.value * other)
        result = self
        for bit in reversed(range(self.width)):
            if result.bit(bit):
                result = result.flip_bit(bit)
                source = result if other is self else other
                result = result.add_shifted(source, bit)
        return result

    def _accumulate_shifted(
        self, other: FixedWidthInteger, shift: int = 0, *, subtract: bool = False
    ) -> tuple[FixedWidthInteger, bool]:
        if type(shift) is not int or shift < 0:
            raise ValueError("Shifted addition requires a nonnegative shift")
        # Sequential reads matter when the operand aliases its updated target.
        result = self.value
        carry = False
        for index in range(shift // 64, self.word_count):
            source = FixedWidthInteger(self.width, result) if other is self else other
            right = source.read_shifted_word(index * 64 - shift)
            left = (result >> (index * 64)) & WORD_MASK
            total = left - right - carry if subtract else left + right + carry
            carry = total < 0 if subtract else bool(total >> 64)
            position = 64 * index
            result = (result & ~(WORD_MASK << position)) | (
                (total & WORD_MASK) << position
            )
        return self._fit(result), carry

    def add_shifted(
        self, other: FixedWidthInteger, shift: int = 0
    ) -> FixedWidthInteger:
        return self._accumulate_shifted(other, shift)[0]

    def subtract_shifted(
        self, other: FixedWidthInteger, shift: int = 0
    ) -> tuple[FixedWidthInteger, bool]:
        return self._accumulate_shifted(other, shift, subtract=True)

    def accumulate_product(
        self,
        left: FixedWidthInteger | int,
        right: FixedWidthInteger | int,
        *,
        shift: int = 0,
        subtract: bool = False,
    ) -> FixedWidthInteger:
        if type(shift) is not int or shift < 0:
            raise ValueError("Product shift must be nonnegative")
        if isinstance(right, int):
            left, right = right, left
        if isinstance(right, int):
            right = self.from_integer(64, right)
        if isinstance(left, int):
            require_word(left)
        elif shift or subtract:
            raise ValueError("Shifted/subtracted product requires a scalar factor")
        value = accumulate_product(
            self.value,
            self.width,
            left.value if isinstance(left, FixedWidthInteger) else left,
            left.width if isinstance(left, FixedWidthInteger) else None,
            right.value,
            right.width,
            left_alias=left is self,
            right_alias=right is self,
            shift=shift,
            subtract=subtract,
        )
        return FixedWidthInteger(self.width, value)

    def times_five(self) -> tuple[FixedWidthInteger, bool]:
        total = self.value * 5
        return self._fit(total), bool(total >> self.width)

    def divide_five(self) -> FixedWidthInteger:
        return FixedWidthInteger(self.width, self.value // 5)

    def __hash__(self) -> int:
        return hash(self.value)

    def __eq__(self, other: object) -> bool:
        return self.value == (
            other.value if isinstance(other, FixedWidthInteger) else other
        )

    def __lt__(self, other: FixedWidthInteger | int) -> bool:
        return self.value < (
            other.value if isinstance(other, FixedWidthInteger) else other
        )

    def __le__(self, other: FixedWidthInteger | int) -> bool:
        return self.value <= (
            other.value if isinstance(other, FixedWidthInteger) else other
        )

    def __ge__(self, other: FixedWidthInteger | int) -> bool:
        return not self < other

    def __gt__(self, other: FixedWidthInteger | int) -> bool:
        return not self <= other

    def __add__(self, other: FixedWidthInteger | int) -> FixedWidthInteger:
        if isinstance(other, int):
            other = self.from_integer(64, other & WORD_MASK)
        width = self.width + other.width + 1
        left = FixedWidthInteger(width).xor_word(self)
        return left.add_word(other)[0]

    def __sub__(self, other: FixedWidthInteger | int) -> FixedWidthInteger:
        if isinstance(other, int):
            other = self.from_integer(64, other & WORD_MASK)
        width = self.width + other.width + 2
        return FixedWidthInteger(width).xor_word(self).subtract_word(other)[0]

    def __neg__(self) -> FixedWidthInteger:
        width = self.width + 1
        return FixedWidthInteger(width).subtract_word(self)[0]

    def __mul__(self, other: FixedWidthInteger | int) -> FixedWidthInteger:
        if isinstance(other, int):
            other = self.from_integer(64, other & WORD_MASK)
        width = self.width + other.width
        return FixedWidthInteger(width).xor_word(self).multiply_into(other)

    def __lshift__(self, amount: int) -> FixedWidthInteger:
        width = self.width + max(0, amount)
        return FixedWidthInteger(width, self.value & ((1 << width) - 1)).shift_left(
            amount
        )

    def __rshift__(self, amount: int) -> FixedWidthInteger:
        return self.shift_right(amount)

    def __mod__(self, modulus: FixedWidthInteger) -> FixedWidthInteger:
        from ._fixed_width_modular import remainder_into

        result = remainder_into(FixedWidthInteger(self.width, self.value), modulus)
        return FixedWidthInteger(modulus.width, result.value & modulus.mask)

    def negate_into(self) -> FixedWidthInteger:
        return self._fit(-self.value)

    def binary(self) -> str:
        return grouped_digits(self.value, self.width, 2, 8)

    def hexadecimal(self) -> str:
        return grouped_digits(self.value, self.width, 16, 8)

    def decimal(self) -> str:
        if self.value <= self.mask:
            return write_decimal(self.value)
        # The source clips during its first decimal division, after extracting
        # the low decimal digit from all allocated storage.
        digit = self.value % 10
        remaining = self.add_scalar(digit, subtract=True).shift_right(1).divide_five()
        prefix = write_decimal(remaining.value) if remaining else ""
        return prefix + str(digit)

    def __str__(self) -> str:
        digits = grouped_digits(self.value, self.width, 16, 16)
        return f"0x{digits} [num_bits={self.width}] (decimal={self.decimal()})"

    def approximate_mantissa(self) -> float:
        if not self.value:
            return 0.0
        leading = self.read_shifted_word(self.bits_in_use - 53) & ((1 << 52) - 1)
        return float(unpack("d", pack("Q", leading | (1023 << 52)))[0])

    def __float__(self) -> float:
        if not self.value:
            return 0.0
        if self.bits_in_use >= 1025:
            raise ValueError("Integer exceeds the source binary64 exponent range")
        leading = self.read_shifted_word(self.bits_in_use - 53) & ((1 << 52) - 1)
        return float(
            unpack("d", pack("Q", leading | ((self.bits_in_use + 1022) << 52)))[0]
        )

    def __imod__(self, modulus: FixedWidthInteger) -> FixedWidthInteger:
        from ._fixed_width_modular import remainder_into

        return remainder_into(self, modulus)

    def __iadd__(self, other: FixedWidthInteger | int) -> FixedWidthInteger:
        return (
            self.add_scalar(other, signed=other < 0)
            if isinstance(other, int)
            else self.add_word(other)[0]
        )

    def __isub__(self, other: FixedWidthInteger | int) -> FixedWidthInteger:
        return (
            self.add_scalar(other, signed=other < 0, subtract=True)
            if isinstance(other, int)
            else self.subtract_word(other)[0]
        )

    def __imul__(self, other: FixedWidthInteger | int) -> FixedWidthInteger:
        return self.multiply_into(other)

    def __ilshift__(self, amount: int) -> FixedWidthInteger:
        return self.shift_left(amount)

    def __irshift__(self, amount: int) -> FixedWidthInteger:
        return self.shift_right(amount)

    def __ixor__(self, other: FixedWidthInteger | int) -> FixedWidthInteger:
        return self.xor_word(other)

    def __iand__(self, other: FixedWidthInteger) -> FixedWidthInteger:
        return self.and_word(other)

    def __ior__(self, other: FixedWidthInteger) -> FixedWidthInteger:
        return self.or_word(other)

    def __radd__(self, other: int) -> FixedWidthInteger:
        return self.from_integer(64, other & WORD_MASK) + self

    def __rsub__(self, other: int) -> FixedWidthInteger:
        return self.from_integer(64, other & WORD_MASK) - self

    def __rmul__(self, other: int) -> FixedWidthInteger:
        return self.from_integer(64, other & WORD_MASK) * self

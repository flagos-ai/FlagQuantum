# SPDX-License-Identifier: Apache-2.0
# Adapted from locked Kickmix util/fixed_width_int.cc at
# 9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453.

"""Modular value algorithms with source cofactor and clipping order."""

from math import floor

from ._fixed_width_integer import FixedWidthInteger


def add_modulo(
    target: FixedWidthInteger,
    other: FixedWidthInteger,
    modulus: FixedWidthInteger,
    *,
    subtract: bool = False,
) -> FixedWidthInteger:
    if target >= modulus:
        raise ValueError("Modular accumulation requires a reduced target")
    reduce_other = other >= modulus if subtract else other > modulus
    if reduce_other:
        return add_modulo(target, other % modulus, modulus, subtract=subtract)
    result, overflow = (
        target.subtract_word(other) if subtract else target.add_word(other)
    )
    if subtract:
        return result.add_word(modulus)[0] if overflow else result
    return result.subtract_word(modulus)[0] if overflow or result >= modulus else result


def double_modulo(
    target: FixedWidthInteger, modulus: FixedWidthInteger
) -> FixedWidthInteger:
    result, carry = target.add_word(target)
    current_modulus = result if modulus is target else modulus
    return (
        result.subtract_word(current_modulus)[0]
        if carry or result >= current_modulus
        else result
    )


def halve_modulo(
    target: FixedWidthInteger, modulus: FixedWidthInteger
) -> FixedWidthInteger:
    if not modulus.bit(0):
        raise ValueError("Modular halving requires an odd modulus")
    result, carry = target.add_word(modulus) if target.bit(0) else (target, False)
    result = result.shift_right(1)
    current_modulus = result if modulus is target else modulus
    if carry and current_modulus:
        result = result.set_bit(current_modulus.bits_in_use - 1)
    return result


def negate_modulo(
    target: FixedWidthInteger, modulus: FixedWidthInteger
) -> FixedWidthInteger:
    result = target.negate_into()
    current_modulus = result if modulus is target else modulus
    result = result.add_word(current_modulus)[0]
    current_modulus = result if modulus is target else modulus
    return result.xor_word(current_modulus) if result == current_modulus else result


def remainder_into(
    target: FixedWidthInteger, modulus: FixedWidthInteger
) -> FixedWidthInteger:
    if not modulus:
        raise ValueError("Integer division by zero")
    if modulus is target:
        raise ValueError(
            "Source mutating remainder aliases its modulus and cannot terminate"
        )
    result, negative = target, False
    divisor_bits = modulus.bits_in_use
    divisor_mantissa = modulus.approximate_mantissa()
    while result.bits_in_use > divisor_bits:
        shift = result.bits_in_use - divisor_bits - 53
        ratio = result.approximate_mantissa() / divisor_mantissa * (2.0**53)
        if shift < 0:
            ratio *= 2.0**shift
            shift = 0
        result = result.accumulate_product(
            modulus, floor(ratio), shift=shift, subtract=True
        )
        if result.bit(result.width - 1):
            negative = not negative
            result = result.negate_into()
    while result >= modulus:
        result = result.subtract_word(modulus)[0]
    if negative and result:
        result = result.xor_word(-1).add_word(modulus, True)[0]
    return result


def is_coprime(target: FixedWidthInteger, other: FixedWidthInteger) -> bool:
    width = max(target.width, other.width)
    left = FixedWidthInteger(width).xor_word(target)
    right = FixedWidthInteger(width).xor_word(other)
    if not left.bit(0):
        left, right = right, left
    if not left.bit(0):
        return False
    while right:
        if not right.bit(0):
            right = right.shift_right(1)
        elif right > left:
            right = right.subtract_word(left)[0].shift_right(1)
        else:
            left, right = right, left
            right = right.subtract_word(left)[0].shift_right(1)
    return left == 1


def inverse_modulo(
    target: FixedWidthInteger, modulus: FixedWidthInteger
) -> FixedWidthInteger:
    if not modulus.bit(0):
        raise ValueError("Source inverse requires an odd modulus")
    if target >= modulus:
        return inverse_modulo(target % modulus, modulus)
    width = modulus.width
    left = FixedWidthInteger(width).xor_word(target)
    right = FixedWidthInteger(width).xor_word(modulus)
    left_cofactor, right_cofactor = FixedWidthInteger(width, 1), FixedWidthInteger(
        width
    )
    if not left.bit(0):
        left, right = right, left
        left_cofactor, right_cofactor = right_cofactor, left_cofactor
    while right:
        if not right.bit(0):
            right = right.shift_right(1)
            right_cofactor = halve_modulo(right_cofactor, modulus)
        elif right > left:
            right = right.subtract_word(left)[0].shift_right(1)
            right_cofactor = halve_modulo(
                add_modulo(right_cofactor, left_cofactor, modulus, subtract=True),
                modulus,
            )
        else:
            left, right = right, left
            right = right.subtract_word(left)[0].shift_right(1)
            left_cofactor, right_cofactor = right_cofactor, left_cofactor
            right_cofactor = halve_modulo(
                add_modulo(right_cofactor, left_cofactor, modulus, subtract=True),
                modulus,
            )
    return left_cofactor

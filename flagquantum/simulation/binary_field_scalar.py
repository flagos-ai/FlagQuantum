# SPDX-License-Identifier: Apache-2.0
# Adapted from locked Kickmix GF2Field Python scalar bindings.
# Revision 9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453.

"""Python scalar semantics over the shared neutral binary field arithmetic."""

from ..core._binary_field import BinaryField


def binary_field_operand(value: object) -> int:
    """Admit the Python int argument domain, including source boolean scalars."""
    if not isinstance(value, int):
        raise TypeError("Binary field scalar arguments require Python integers")
    result = int(value)
    if result < 0 or result.bit_length() > 1024:
        raise ValueError("Binary polynomial must fit 1024 nonnegative coefficient bits")
    return result


def binary_field_sum(field: BinaryField, left: int, right: int) -> int:
    """Reduce both Python operands before the original scalar XOR operation."""
    return field.add(
        field.mod(binary_field_operand(left)), field.mod(binary_field_operand(right))
    )


def binary_field_power(field: BinaryField, value: int, exponent: int) -> int:
    """Retain arbitrary-width signed exponents and the source zero inverse."""
    if not isinstance(exponent, int):
        raise TypeError("Binary field exponent must be a Python integer")
    exponent = int(exponent)
    value = field.mod(binary_field_operand(value))
    if exponent < 0:
        value = field.inverse(value)
        exponent = -exponent
    return field.power(value, exponent)

# SPDX-License-Identifier: Apache-2.0
# Adapted from locked Kickmix FixedWidthInt and big_int_to_str.
# https://github.com/quantumlib/Qualtran/tree/9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453/kickmix/src/kickmix

"""Fixed-width unsigned literals and their command report digits."""

from typing import Literal

from ...core._unsigned_integer_text import parse_integer_literal, write_decimal


def parse_unsigned_word(value: str) -> tuple[int, int]:
    """Return the value and declared width of a source unsigned literal."""
    try:
        return parse_integer_literal(value)
    except ValueError as error:
        raise ValueError(f"Invalid unsigned word {value!r}") from error


def format_unsigned_word(value: int, width: int, radix: Literal[2, 10, 16]) -> str:
    """Render a fitted unsigned lane word using the locked presentation."""
    if value < 0 or value.bit_length() > width:
        raise ValueError("Unsigned word does not fit the declared width")
    if radix == 10:
        return write_decimal(value)
    if not width:
        return ""
    if radix == 16:
        return format(value, f"0{(width + 3) // 4}X")
    return format(value, f"0{width}b")

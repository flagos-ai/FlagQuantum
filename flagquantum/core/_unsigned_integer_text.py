# SPDX-License-Identifier: Apache-2.0
# Adapted from locked Kickmix util/fixed_width_int.cc and big_int_to_str.
# Revision 9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453.

"""Unsigned integer lexical rules independent of Compiler report formatting."""

import re
from typing import Literal

_DECIMAL_CHUNK = 9
_DECIMAL_BASE = 10**_DECIMAL_CHUNK


def read_decimal(digits: str) -> int:
    try:
        return int(digits)
    except ValueError:
        value = 0
        for i in range(0, len(digits), _DECIMAL_CHUNK):
            chunk = digits[i : i + _DECIMAL_CHUNK]
            value = value * 10 ** len(chunk) + int(chunk)
        return value


def write_decimal(value: int) -> str:
    try:
        return str(value)
    except ValueError:
        chunks = []
        while value >= _DECIMAL_BASE:
            value, low = divmod(value, _DECIMAL_BASE)
            chunks.append(f"{low:0{_DECIMAL_CHUNK}}")
        return str(value) + "".join(reversed(chunks))


def parse_integer_literal(
    text: str, width: int | None = None, *, radix: Literal[2, 10, 16] | None = None
) -> tuple[int, int]:
    """Retain declared digit widths and explicit decimal truncation.

    Generic parsing trims ASCII spaces and requires nonempty text. Explicit
    radix parsing retains the source factories' empty-digit behavior.
    """
    if width is not None and (type(width) is not int or width < 0):
        raise ValueError("Declared integer width must be nonnegative")
    if radix is None:
        text = text.strip(" ")
        if not text:
            raise ValueError("Empty text cannot be parsed as an integer")
        radix = 2 if text.startswith("0b") else 16 if text.startswith("0x") else 10
    prefix = {2: "0b", 16: "0x"}.get(radix)
    if prefix is not None and text.startswith(prefix):
        text = text[2:]
    pattern = {2: r"[01_]*", 16: r"[0-9a-fA-F_]*", 10: r"[0-9_]*"}[radix]
    if re.fullmatch(pattern, text) is None:
        raise ValueError("Invalid unsigned integer digits")
    digits = text.replace("_", "")
    value = (
        read_decimal(digits) if radix == 10 and digits else int(digits or "0", radix)
    )
    actual = value.bit_length()
    if radix == 2:
        actual = len(digits)
    elif radix == 16 and digits:
        actual = 4 * (len(digits) - 1) + max(1, int(digits[0], 16).bit_length())
    if width is not None:
        if radix != 10 and actual > width:
            raise ValueError("Literal digit width exceeds the declared integer width")
        return value & ((1 << width) - 1), width
    return value, actual


def grouped_digits(value: int, width: int, radix: Literal[2, 16], group: int) -> str:
    """Render the declared lane digits with source group boundaries."""
    count = width if radix == 2 else (width + 3) // 4
    if not count:
        return ""
    specifier = "b" if radix == 2 else "X"
    digits = format(value, f"0{count}{specifier}")[-count:]
    first = len(digits) % group or group
    groups = [digits[:first]]
    groups.extend(digits[k : k + group] for k in range(first, len(digits), group))
    return "_".join(groups)

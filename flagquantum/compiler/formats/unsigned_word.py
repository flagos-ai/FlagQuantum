# SPDX-License-Identifier: Apache-2.0
# Adapted from locked Kickmix FixedWidthInt and big_int_to_str.
# https://github.com/quantumlib/Qualtran/tree/9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453/kickmix/src/kickmix

"""Fixed-width unsigned literals and their command report digits."""

import re
from typing import Literal

_DECIMAL_CHUNK = 9
_DECIMAL_BASE = 10**_DECIMAL_CHUNK


def _read_decimal(digits: str) -> int:
    try:
        return int(digits)
    except ValueError:
        # Python can limit whole decimal conversions. The format admits
        # arbitrarily wide registers, so convert bounded chunks instead.
        value = 0
        for i in range(0, len(digits), _DECIMAL_CHUNK):
            chunk = digits[i : i + _DECIMAL_CHUNK]
            value = value * 10 ** len(chunk) + int(chunk)
        return value


def _write_decimal(value: int) -> str:
    try:
        return str(value)
    except ValueError:
        chunks = []
        while value >= _DECIMAL_BASE:
            value, low = divmod(value, _DECIMAL_BASE)
            chunks.append(f"{low:0{_DECIMAL_CHUNK}}")
        return str(value) + "".join(reversed(chunks))


def parse_unsigned_word(value: str) -> tuple[int, int]:
    """Return the value and declared width of a source unsigned literal."""
    text = value.strip(" ")
    if text.startswith("0b"):
        digits, base, pattern = text[2:], 2, r"[01_]*"
    elif text.startswith("0x"):
        digits, base, pattern = text[2:], 16, r"[0-9a-fA-F_]*"
    else:
        digits, base, pattern = text, 10, r"[0-9_]+"
    if not re.fullmatch(pattern, digits):
        raise ValueError(f"Invalid unsigned word {value!r}")
    digits = digits.replace("_", "")
    if not digits:
        return 0, 0
    if base == 2:
        return int(digits, 2), len(digits)
    if base == 16:
        return int(digits, 16), 4 * (len(digits) - 1) + max(
            1, int(digits[0], 16).bit_length()
        )
    number = _read_decimal(digits)
    return number, number.bit_length()


def format_unsigned_word(value: int, width: int, radix: Literal[2, 10, 16]) -> str:
    """Render a fitted unsigned lane word using the locked presentation."""
    if value < 0 or value.bit_length() > width:
        raise ValueError("Unsigned word does not fit the declared width")
    if radix == 10:
        return _write_decimal(value)
    if not width:
        return ""
    if radix == 16:
        return format(value, f"0{(width + 3) // 4}X")
    return format(value, f"0{width}b")

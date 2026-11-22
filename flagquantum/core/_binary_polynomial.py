# SPDX-License-Identifier: Apache-2.0
# Adapted from locked Kickmix util/gf2_poly.cc and gf2_poly.h.
# Revision 9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453.
# Integer values replace inline words and mutable methods. No public export.

"""Internal binary polynomial value algebra for metadata and synthesis.

Bit k is the coefficient of x**k. Plain integers retain value ownership across
domains. Storage-facing operations use the source 1024-bit capacity. Remainder
also serves existing unbounded synthesis callers and therefore has no width cap.
"""

import re

_CAPACITY = 1024
_MASK = (1 << _CAPACITY) - 1
_WHITESPACE = " \t\n\r"


def truncate(value: int, width: int) -> int:
    """Project a nonnegative value onto the requested source storage width."""
    if value < 0 or width < 0:
        raise ValueError("Polynomial values and widths must be nonnegative")
    return value & ((1 << min(width, _CAPACITY)) - 1)


def xor_shifted(value: int, other: int, shift: int) -> int:
    """Return the source bounded shifted XOR, including self aliases."""
    if shift < 0:
        raise ValueError("Polynomial shift must be nonnegative")
    if shift >= _CAPACITY:
        return value
    return value ^ ((other << shift) & _MASK)


def parse(text: str) -> int:
    """Read the source algebraic, hexadecimal or binary polynomial grammar."""
    text = text.strip(_WHITESPACE)
    if text[:2].lower() in ("0x", "0b") and "+" not in text:
        digits = text[2:].replace("_", "")
        hexadecimal = text[1].lower() == "x"
        pattern = r"[0-9a-fA-F]+" if hexadecimal else r"[01]+"
        if not re.fullmatch(pattern, digits):
            raise ValueError("Expected hexadecimal or binary polynomial digits")
        # FixedWidthInt counts declared digits, including leading zeroes.
        width = len(digits)
        if hexadecimal:
            width = 4 * (len(digits) - 1) + max(1, int(digits[0], 16).bit_length())
        if width > _CAPACITY:
            raise ValueError("Polynomial literal exceeds 1024 coefficient bits")
        return int(digits, 16 if hexadecimal else 2)
    result = 0
    for term in text.split("+"):
        term = term.strip(_WHITESPACE)
        if term == "0":
            continue
        if term == "1":
            exponent = 0
        else:
            match = re.fullmatch(
                r"[xX][ \t\n\r]*(?:(?:\^|\*\*)[ \t\n\r]*([0-9]+))?", term
            )
            if match is None:
                raise ValueError(f"Invalid polynomial term {term!r}")
            digits = match[1]
            # Bound before integer conversion, including arbitrarily many zeroes.
            significant = digits.lstrip("0") if digits is not None else "1"
            if len(significant) > 4:
                raise ValueError("Polynomial exponent exceeds 1023")
            exponent = int(significant or "0")
        if exponent >= _CAPACITY:
            raise ValueError("Polynomial exponent exceeds 1023")
        coefficient = 1 << exponent
        if result & coefficient:
            raise ValueError(f"Repeated polynomial term {term!r}")
        result ^= coefficient
    return result


def multiply(left: int, right: int) -> int:
    """Execute the source four-bit comb product within 1024 coefficient bits."""
    if left < 0 or right < 0:
        raise ValueError("Polynomial values must be nonnegative")
    if not left or not right:
        return 0
    if left.bit_length() + right.bit_length() - 2 >= _CAPACITY:
        raise ValueError("Polynomial product exceeds 1024 coefficient bits")
    if left.bit_length() < right.bit_length():
        left, right = right, left
    table = [0] * 16
    table[1] = left
    for index in range(2, 16, 2):
        table[index] = (table[index // 2] << 1) & _MASK
        table[index + 1] = table[index] ^ left
    words = (right.bit_length() + 63) // 64
    result = 0
    for nibble in reversed(range(16)):
        for word in range(words):
            index = (right >> (word * 64 + nibble * 4)) & 15
            if index:
                result = xor_shifted(result, table[index], word * 64)
        if nibble:
            result = (result << 4) & _MASK
    return result


def remainder(value: int, modulus: int) -> int:
    """Reduce a polynomial by source shift-XOR without restricting caller width."""
    if value < 0 or modulus <= 0:
        raise ValueError(
            "Polynomial reduction requires nonnegative value and positive modulus"
        )
    while value.bit_length() >= modulus.bit_length():
        value ^= modulus << (value.bit_length() - modulus.bit_length())
    return value


def gcd(left: int, right: int) -> int:
    """Return the source Euclidean polynomial greatest common divisor."""
    if left < 0 or right < 0:
        raise ValueError("Polynomial values must be nonnegative")
    while right:
        left, right = right, remainder(left, right)
    return left


def invert_mod(value: int, modulus: int) -> int:
    """Return the source bounded extended-Euclidean inverse, or zero for nonunits."""
    r0, r1 = remainder(value, modulus), modulus
    t0, t1 = 1, 0
    while r1:
        while r0 and r0.bit_length() >= r1.bit_length():
            shift = r0.bit_length() - r1.bit_length()
            r0 = xor_shifted(r0, r1, shift)
            t0 = xor_shifted(t0, t1, shift)
        r0, r1 = r1, r0
        t0, t1 = t1, t0
    return remainder(t0, modulus) if r0 == 1 else 0


def algebraic(value: int, max_terms: int = _CAPACITY) -> str:
    """Render descending powers with the source highest-four/lowest-two elision."""
    terms = [
        "1" if degree == 0 else "x" if degree == 1 else f"x^{degree}"
        for degree in reversed(range(value.bit_length()))
        if value >> degree & 1
    ]
    if len(terms) > max_terms >= 6:
        terms = terms[:4] + ["..."] + terms[-2:]
    return " + ".join(terms) or "0"

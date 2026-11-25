"""Exact, bounded classical integer bindings; no real-to-integer coercion."""

from collections.abc import Sequence

from .syntax import Token, reject


def checked_integer(value: object, token: Token) -> int:
    if (
        type(value) is not int
        or not isinstance(value, int)
        or abs(value).bit_length() > 256
    ):
        reject(
            token,
            "binding",
            "classical int requires an exact non-Boolean integer of at most 256 bits",
        )
    return value


def integer_operation(token: Token, left: int, right: int) -> int:
    operator = token.text
    if operator == "+":
        result = left + right
    elif operator == "-":
        result = left - right
    elif operator == "*":
        result = left * right
    elif operator == "%":
        if right <= 0:
            reject(token, "expression", "static modulo requires a positive divisor")
        result = left % right
    else:
        assert operator == "**"
        if not 0 <= right <= 255:
            reject(token, "budget.number", "static exponent must be in 0..255")
        result = left**right
    if abs(result).bit_length() > 256:
        reject(token, "budget.number", "static integer exceeds 256 bits")
    return result


def checked_integer_array(
    value: object, token: Token, width: int | None = None
) -> tuple[int, ...]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or not 1 <= len(value) <= 256
        or (width is not None and len(value) != width)
    ):
        reject(
            token,
            "binding",
            "classical int array requires 1..256 elements and its declared length",
        )
    return tuple(checked_integer(item, token) for item in value)

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

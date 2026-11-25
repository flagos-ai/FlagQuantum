"""Exact unsigned range helpers; bounds are inclusive and widths never truncate."""


def unsigned_width(upper_bound: int) -> int:
    """Return the minimal positive width representing [0, upper_bound]."""
    if isinstance(upper_bound, bool) or not isinstance(upper_bound, int):
        raise TypeError("upper_bound must be a Python integer")
    if upper_bound < 0:
        raise ValueError("upper_bound must be nonnegative")
    return max(1, upper_bound.bit_length())


def bitwise_width(
    operator: str, widths: tuple[int, ...], signed: tuple[bool, ...]
) -> int:
    """Compute the encoding width for zero- or sign-extended bitwise operands."""
    if operator == "~":
        return widths[0]
    if operator == "&" and not all(signed):
        return min(
            width for width, sign in zip(widths, signed, strict=True) if not sign
        )
    signed_result = any(signed)
    return max(
        width + int(signed_result and not sign)
        for width, sign in zip(widths, signed, strict=True)
    )

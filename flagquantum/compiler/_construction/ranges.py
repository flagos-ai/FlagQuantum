"""Exact unsigned range helpers; bounds are inclusive and widths never truncate."""


def unsigned_width(upper_bound: int) -> int:
    """Return the minimal positive width representing [0, upper_bound]."""
    if isinstance(upper_bound, bool) or not isinstance(upper_bound, int):
        raise TypeError("upper_bound must be a Python integer")
    if upper_bound < 0:
        raise ValueError("upper_bound must be nonnegative")
    return max(1, upper_bound.bit_length())

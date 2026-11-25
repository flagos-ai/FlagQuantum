"""Independent affine-square costs expressed as one- and two-bit factors."""

import math

from examples.synthesis.factor_reference import CostFactor


def order_register_factors(
    factors: list[CostFactor], widths: tuple[int, ...]
) -> list[CostFactor]:
    """Place fields in declaration order and preserve each field's low-bit index."""
    if not widths or any(type(width) is not int or width < 1 for width in widths):
        raise ValueError("Register widths must be positive integers")
    total = sum(widths)
    if total > 24 or any(
        type(bit) is not int or not 0 <= bit < total
        for bits, _ in factors
        for bit in bits
    ):
        raise ValueError("Factor locations must fit at most twenty-four register bits")
    mapping: list[int] = []
    offset = 0
    for width in widths:
        mapping.extend(total - offset - width + bit for bit in range(width))
        offset += width
    return [(tuple(mapping[bit] for bit in bits), table) for bits, table in factors]


def affine_square_factors(
    bits: tuple[int, ...],
    coefficients: tuple[float, ...],
    offset: float,
    weight: float,
) -> list[CostFactor]:
    """Represent an affine square using idempotent Boolean indicator variables.

    Each emitted truth table spans at most two bits. This avoids enlarging the
    existing six-bit enumerated residual reference or building a full-state
    diagonal. It is example-side Circuit reference code, not Compiler lowering.
    """
    if (
        not 1 <= len(bits) <= 24
        or len(bits) != len(coefficients)
        or len(set(bits)) != len(bits)
        or any(type(bit) is not int or not 0 <= bit < 24 for bit in bits)
        or not all(math.isfinite(value) for value in (*coefficients, offset, weight))
    ):
        raise ValueError(
            "Use distinct logical bits and matching finite affine coefficients"
        )
    factors: list[CostFactor] = [((), (weight * offset**2,))]
    for index, (bit, coefficient) in enumerate(zip(bits, coefficients, strict=True)):
        factors.append(
            ((bit,), (0.0, weight * (coefficient**2 + 2 * offset * coefficient)))
        )
        for other, other_coefficient in zip(
            bits[index + 1 :], coefficients[index + 1 :], strict=True
        ):
            factors.append(
                (
                    (bit, other),
                    (0.0, 0.0, 0.0, 2 * weight * coefficient * other_coefficient),
                )
            )
    return factors

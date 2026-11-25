"""Independent Circuit factors derive freshly from the original constraints."""

from itertools import combinations

import flagquantum as fq
from examples.synthesis.factor_reference import (
    CostFactor,
    factored_cost_circuit,
    squared_residual,
)


def reference_circuit(
    parameters: tuple[float, ...], coefficients: tuple[float, ...]
) -> fq.Circuit:
    # Named array views list most-significant bits first; x[0] is its low bit.
    x = tuple(3 + i for i in range(13))
    factors: list[CostFactor] = [
        ((x[i],), (0.0, value)) for i, value in enumerate(coefficients)
    ]
    for group in ((1, 3, 4), (0, 5), (6, 9)):
        factors.append(
            squared_residual(
                tuple(x[i] for i in group), (1.0,) * len(group), -1.0, -3.0
            )
        )
    terms = tuple(zip((*x, 0, 1, 2), (1.0,) * 13 + (1.0, 2.0, 4.0), strict=True))
    # Expand only the original affine capacity residual, never a synthesis IR.
    factors.append(((), (-3.0 * 49,)))
    for bit, weight in terms:
        factors.append(((bit,), (0.0, -3.0 * (weight * weight - 14 * weight))))
    for (left, a), (right, b) in combinations(terms, 2):
        factors.append(((left, right), (0.0, 0.0, 0.0, -6.0 * a * b)))
    interleaved = tuple(
        value for i in range(4) for value in (parameters[i], parameters[4 + i])
    )
    return factored_cost_circuit(16, tuple(factors), interleaved)

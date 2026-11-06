# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 grid Hamiltonian inverse_sqrt.py.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/chemistry/trotter/grid_ham/inverse_sqrt.py

"""Coefficient tables and declared costs for grid inverse-square-root stages."""

from ...compiler._component_trace.resource_model import (
    ComponentCost,
    CostCall,
    GateCost,
)
from ...compiler.registers.arithmetic_resource_model import source_add_cost

_LOW = (
    0.9999413248911988,
    0.4960989191590354,
    0.33261112772430493,
    0.14876762006038398,
)
_HIGH = (
    0.8164851520538522,
    0.27136515484240234,
    0.12756148214815175,
    0.04475302857915384,
)


def grid_inverse_sqrt_qrom_data(
    selection_width: int, coefficient_width: int
) -> tuple[tuple[int, ...], ...]:
    """Return four coefficient banks in original address order, with zero at zero."""
    if type(selection_width) is not int or selection_width < 2:
        raise ValueError(
            "Grid inverse-square-root QROM requires at least two address bits"
        )
    if type(coefficient_width) is not int or not 1 <= coefficient_width <= 63:
        raise ValueError(
            "Grid inverse-square-root QROM requires 1 to 63 coefficient bits"
        )

    def word(value: float) -> int:
        numerator, denominator = value.as_integer_ratio()
        return (numerator << coefficient_width) // denominator

    banks = []
    for low, high in zip(_LOW, _HIGH, strict=True):
        values = [0, word(low), word(high), word(low / 2**0.5)]
        for exponent in range(2, selection_width):
            repeat = 1 << (exponent - 1)
            scale = 2 ** (exponent / 2)
            values.extend([word(low / scale)] * repeat)
            values.extend([word(high / scale)] * repeat)
        banks.append(tuple(values))
    return tuple(banks)


def _real_product_cost(width: int) -> ComponentCost:
    return ComponentCost(
        "MultiplyTwoReals", 3 * width, GateCost(toffoli=width**2 - width - 1)
    )


def source_grid_inverse_sqrt_polynomial_cost(
    distance_width: int, coefficient_width: int, output_width: int
) -> ComponentCost:
    """Count the three source real products and three signed additions."""
    if any(
        type(width) is not int or width < 1
        for width in (distance_width, coefficient_width, output_width)
    ):
        raise ValueError(
            "Grid inverse-square-root polynomial widths must be positive integers"
        )
    return ComponentCost(
        "PolynmomialEvaluationInverseSquareRoot",
        distance_width + 4 * coefficient_width + output_width,
        calls=(
            CostCall(_real_product_cost(coefficient_width), 3),
            CostCall(source_add_cost(coefficient_width, coefficient_width), 3),
        ),
    )


def source_grid_inverse_sqrt_newton_cost(
    distance_width: int, polynomial_width: int, target_width: int
) -> ComponentCost:
    """Count the locked square, scale, two real products and one signed add."""
    if any(
        type(width) is not int or width < 1
        for width in (distance_width, polynomial_width, target_width)
    ):
        raise ValueError(
            "Grid inverse-square-root Newton widths must be positive integers"
        )
    square = ComponentCost(
        "SquareRealNumber",
        3 * polynomial_width,
        GateCost(toffoli=polynomial_width**2 // 2 - 4),
    )
    scale = ComponentCost(
        "ScaleIntByReal",
        2 * polynomial_width + distance_width,
        GateCost(
            toffoli=polynomial_width * (2 * distance_width - 1) - distance_width**2
        ),
    )
    return ComponentCost(
        "NewtonRaphsonApproxInverseSquareRoot",
        distance_width + polynomial_width + target_width,
        calls=(
            CostCall(square),
            CostCall(scale),
            CostCall(_real_product_cost(target_width), 2),
            CostCall(source_add_cost(target_width, target_width)),
        ),
    )

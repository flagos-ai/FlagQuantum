# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 grid Hamiltonian inverse_sqrt.py.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/chemistry/trotter/grid_ham/inverse_sqrt.py

"""Coefficient tables and declared costs for grid inverse-square-root stages."""

from typing import cast

from ...compiler._component_trace.resource_model import (
    ComponentCost,
    CostCall,
    GateCost,
    estimate_component_resources,
)
from ...compiler.registers.arithmetic.arithmetic_resource_model import source_add_cost
from ...compiler.registers.fixed_point.grid_inverse_sqrt_signature import (
    grid_inverse_sqrt_signature,
)
from ...compiler.registers.resources.symbolic_resource_model import (
    SymbolicResourceFormula,
    source_add_resource_formula,
)
from ...core.ir.program.signatures import Signature
from ...core.value_math import SymbolicInt, smax
from ...core.value_parameters import is_symbolic_value

_LOW_DECIMAL = (
    "0.99994132489119882162",
    "0.49609891915903542303",
    "0.33261112772430493331",
    "0.14876762006038398086",
)
_HIGH_DECIMAL = (
    "0.81648515205385221995",
    "0.27136515484240234115",
    "0.12756148214815175348",
    "0.044753028579153842218",
)
_LOW = tuple(map(float, _LOW_DECIMAL))
_HIGH = tuple(map(float, _HIGH_DECIMAL))


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


def source_grid_inverse_sqrt_polynomial_signature(
    distance_width: int,
    coefficient_width: int,
    output_width: int,
) -> Signature:
    """Declare the locked polynomial resource model's ordered through ports."""
    if any(
        type(width) is not int or width < 1
        for width in (distance_width, coefficient_width, output_width)
    ):
        raise ValueError(
            "Grid inverse-square-root polynomial widths must be positive integers"
        )
    return grid_inverse_sqrt_signature(
        distance_width, coefficient_width, output_width, newton=False
    )


def source_grid_inverse_sqrt_polynomial_cost(
    distance_width: int,
    coefficient_width: int,
    output_width: int,
    *,
    legacy_shims: bool = False,
) -> ComponentCost:
    """Count the three source real products and three signed additions."""
    signature = source_grid_inverse_sqrt_polynomial_signature(
        distance_width, coefficient_width, output_width
    )
    return ComponentCost(
        "PolynmomialEvaluationInverseSquareRoot",
        int(signature.n_qubits()),
        calls=(
            CostCall(_real_product_cost(coefficient_width), 3),
            CostCall(
                source_add_cost(
                    coefficient_width, coefficient_width, legacy_shims=legacy_shims
                ),
                3,
            ),
        ),
    )


def source_grid_inverse_sqrt_newton_signature(
    distance_width: int,
    polynomial_width: int,
    target_width: int,
) -> Signature:
    """Declare the locked Newton resource model's ordered through ports."""
    if any(
        type(width) is not int or width < 1
        for width in (distance_width, polynomial_width, target_width)
    ):
        raise ValueError(
            "Grid inverse-square-root Newton widths must be positive integers"
        )
    return grid_inverse_sqrt_signature(
        distance_width, polynomial_width, target_width, newton=True
    )


def source_grid_inverse_sqrt_newton_cost(
    distance_width: int,
    polynomial_width: int,
    target_width: int,
    *,
    legacy_shims: bool = False,
) -> ComponentCost:
    """Count the locked square, scale, two real products and one signed add."""
    signature = source_grid_inverse_sqrt_newton_signature(
        distance_width, polynomial_width, target_width
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
        int(signature.n_qubits()),
        calls=(
            CostCall(square),
            CostCall(scale),
            CostCall(_real_product_cost(target_width), 2),
            CostCall(
                source_add_cost(target_width, target_width, legacy_shims=legacy_shims)
            ),
        ),
    )


def source_grid_inverse_sqrt_resource_formula(
    distance_width: SymbolicInt,
    polynomial_width: SymbolicInt,
    target_width: SymbolicInt,
    *,
    newton: bool,
    legacy_shims: bool = False,
) -> SymbolicResourceFormula:
    """Retain concrete or symbolic source model categories, independently of gates."""
    if newton and not is_symbolic_value(polynomial_width) and polynomial_width < 3:
        raise ValueError(
            "bitsize must be at least 3 for SquareRealNumber bloq to make sense."
        )
    if all(
        type(width) is int and width > 0
        for width in (distance_width, polynomial_width, target_width)
    ):
        model = (
            source_grid_inverse_sqrt_newton_cost
            if newton
            else source_grid_inverse_sqrt_polynomial_cost
        )
        report = estimate_component_resources(
            model(
                cast(int, distance_width),
                cast(int, polynomial_width),
                cast(int, target_width),
                legacy_shims=legacy_shims,
            )
        )
        gates = report.gates
        return SymbolicResourceFormula(
            gates.and_bloq,
            gates.clifford,
            gates.measurement,
            report.peak_logical_qubits,
            gates.t,
            gates.toffoli,
            gates.cswap,
            gates.rotation,
        )
    # Symbolic roots lack a physical decomposition upstream. The same direct
    # child formulas and maximum-width fallback apply; no schedule is inferred.
    width = target_width if newton else polynomial_width
    add = source_add_resource_formula(width, width, legacy_shims=legacy_shims)
    repetitions = 1 if newton else 3
    root_width = grid_inverse_sqrt_signature(
        distance_width, polynomial_width, target_width, newton=newton
    ).n_qubits()
    product = width**2 - width - 1
    if newton:
        count = (
            polynomial_width**2 // 2
            - 4
            + polynomial_width * (2 * distance_width - 1)
            - distance_width**2
            + 2 * product
        )
        peak = smax(
            root_width,
            3,
            2 * polynomial_width,
            3 * polynomial_width,
            polynomial_width + distance_width,
            2 * polynomial_width + distance_width,
            3 * target_width,
            add.logical_qubits,
        )
    else:
        count = 3 * product
        peak = smax(root_width, 3, 3 * polynomial_width, add.logical_qubits)
    return SymbolicResourceFormula(
        add.and_bloq * repetitions,
        add.clifford * repetitions,
        add.measurement * repetitions,
        cast(SymbolicInt, peak),
        t=add.t * repetitions,
        toffoli=count + add.toffoli * repetitions,
        cswap=add.cswap * repetitions,
        rotation=add.rotation * repetitions,
    )

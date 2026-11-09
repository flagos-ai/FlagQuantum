# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from the Qualtran v0.7.0 single-factorization PREPARE and SELECT cost graphs.
# https://github.com/quantumlib/Qualtran/tree/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/chemistry/sf

"""Declared PREPARE and SELECT costs for single-factorized chemistry."""

from ...compiler._component_trace.resource_model import (
    ComponentCost,
    CostCall,
    GateCost,
)
from ...compiler.registers.comparison.less_equal import (
    source_equal_width_less_equal_cost,
)
from ...compiler.registers.preparation.uniform_preparation import (
    source_uniform_preparation_cost,
)
from ...compiler.registers.sorting.contiguous_index import source_contiguous_index_cost
from .chemistry_qroam_model import (
    source_chemistry_qroam_cost,
    source_chemistry_qroam_two_axis_cost,
)

_TOFFOLI = ComponentCost("Toffoli", 3, GateCost(toffoli=1))


def _spatial_width(spin_orbitals: int) -> int:
    if type(spin_orbitals) is not int or spin_orbitals < 2 or spin_orbitals % 2:
        raise ValueError(
            "Single factorization requires a positive even spin-orbital count"
        )
    return (spin_orbitals // 2 - 1).bit_length()


def _controlled_swap_cost(width: int) -> ComponentCost:
    return ComponentCost("CSwap", 2 * width + 1, GateCost(cswap=width))


def source_single_factorization_select_cost(
    spin_orbitals: int, *, additional_control: bool = False
) -> ComponentCost:
    """Keep the locked SELECT width and Toffoli formula for either control flag."""
    width = _spatial_width(spin_orbitals)
    if type(additional_control) is not bool:
        raise ValueError("Single factorization SELECT control flag must be boolean")
    return ComponentCost(
        "SelectSingleFactorization",
        2 * width + 3,
        calls=(CostCall(_TOFFOLI, 2 * (spin_orbitals - 2)),),
    )


def source_single_factorization_inner_prepare_cost(
    auxiliaries: int,
    spin_orbitals: int,
    alias_bits: int,
    rotation_bits: int = 8,
    first_block: int = 1,
    second_block: int = 1,
    *,
    adjoint: bool = False,
) -> ComponentCost:
    """Model the original triangular PREPARE, including both QROAM axes."""
    n = _spatial_width(spin_orbitals)
    if (
        type(auxiliaries) is not int
        or auxiliaries < 0
        or type(rotation_bits) is not int
        or rotation_bits < 1
    ):
        raise ValueError(
            "Single factorization requires nonnegative auxiliaries and positive rotation precision"
        )
    if n == 0:
        raise ValueError(
            "Single factorization inner PREPARE requires at least two spatial orbitals"
        )
    triangular = spin_orbitals**2 // 8 + spin_orbitals // 4
    return ComponentCost(
        "Adjoint" if adjoint else "InnerPrepareSingleFactorization",
        auxiliaries.bit_length() + 2 * n + 1,
        calls=(
            CostCall(_TOFFOLI, 6 * n + 2 * rotation_bits - 7),
            CostCall(source_contiguous_index_cost(n, 2 * n)),
            CostCall(
                source_chemistry_qroam_two_axis_cost(
                    auxiliaries + 1,
                    triangular,
                    first_block,
                    second_block,
                    2 * n + alias_bits + 2,
                    adjoint=adjoint,
                )
            ),
            CostCall(source_equal_width_less_equal_cost(alias_bits)),
            CostCall(_controlled_swap_cost(2 * n)),
        ),
    )


def source_single_factorization_outer_prepare_cost(
    auxiliaries: int, alias_bits: int, *, adjoint: bool = False
) -> ComponentCost:
    """Count uniform, QROAM, comparison and swap on the outer alias register."""
    if type(auxiliaries) is not int or auxiliaries < 0:
        raise ValueError("Single factorization requires nonnegative auxiliaries")
    rows = auxiliaries + 1
    lookup_address = rows.bit_length()
    return ComponentCost(
        "Adjoint" if adjoint else "OuterPrepareSingleFactorization",
        auxiliaries.bit_length() + 3,
        calls=(
            CostCall(source_uniform_preparation_cost(rows)),
            CostCall(
                source_chemistry_qroam_cost(
                    rows, lookup_address + alias_bits + 2, adjoint=adjoint
                )
            ),
            CostCall(source_equal_width_less_equal_cost(alias_bits)),
            CostCall(_controlled_swap_cost(lookup_address + 1)),
        ),
    )

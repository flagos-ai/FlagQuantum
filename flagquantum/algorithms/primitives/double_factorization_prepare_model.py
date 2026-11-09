# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 double-factorization PREPARE.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/chemistry/df/prepare.py

"""Declared PREPARE and indexed data costs for double-factorized chemistry."""

from dataclasses import replace

from ...compiler._component_trace.resource_model import (
    ComponentCost,
    CostCall,
    GateCost,
)
from ...compiler.registers.arithmetic.arithmetic_resource_model import source_add_cost
from ...compiler.registers.preparation.uniform_preparation import (
    source_uniform_preparation_cost,
)
from .chemistry_qroam_model import source_chemistry_qroam_cost

_TOFFOLI = ComponentCost("Toffoli", 3, GateCost(toffoli=1))


def _df_address_widths(spin_orbitals: int, eigenvalues: int) -> tuple[int, int, int]:
    if type(spin_orbitals) is not int or spin_orbitals < 2 or spin_orbitals % 2:
        raise ValueError(
            "Double factorization requires a positive even spin-orbital count"
        )
    if type(eigenvalues) is not int or eigenvalues < 0:
        raise ValueError("Double factorization requires a nonnegative eigenvalue count")
    spatial = spin_orbitals // 2
    extent = eigenvalues + spatial
    return spatial, (spatial - 1).bit_length(), (extent - 1).bit_length()


def _require_positive_precision(*widths: int) -> None:
    if any(type(width) is not int or width < 1 for width in widths):
        raise ValueError("Double factorization requires positive bit precision")


def _auxiliary_width(auxiliaries: int) -> int:
    if type(auxiliaries) is not int or auxiliaries < 0:
        raise ValueError("Double factorization requires a nonnegative auxiliary count")
    return auxiliaries.bit_length()


def source_double_factorization_inner_prepare_cost(
    spin_orbitals: int,
    eigenvalues: int,
    rotation_bits: int,
    alias_bits: int,
    *,
    adjoint: bool = False,
) -> ComponentCost:
    """Count rank preparation, offset addition and the declared alias lookup."""
    spatial, rank, offset = _df_address_widths(spin_orbitals, eigenvalues)
    _require_positive_precision(rotation_bits, alias_bits)
    toffolis = 8 * rank + 2 * rotation_bits + alias_bits - 6
    if toffolis < 0:
        raise ValueError("The locked inner PREPARE formula gives a negative gate count")
    addition = source_add_cost(offset, offset)
    return ComponentCost(
        "Adjoint" if adjoint else "InnerPrepareDoubleFactorization",
        2 * rank + offset + rotation_bits + 1,
        calls=(
            CostCall(_TOFFOLI, toffolis),
            CostCall(replace(addition, name="Adjoint") if adjoint else addition),
            CostCall(
                source_chemistry_qroam_cost(
                    eigenvalues + spatial, rank + alias_bits + 2, adjoint=adjoint
                )
            ),
        ),
    )


def source_double_factorization_outer_prepare_cost(
    auxiliaries: int, alias_bits: int, *, adjoint: bool = False
) -> ComponentCost:
    """Count the locked outer alias graph, including its adjoint lookup cost."""
    width = _auxiliary_width(auxiliaries)
    _require_positive_precision(alias_bits)
    uniform = source_uniform_preparation_cost(auxiliaries + 1)
    return ComponentCost(
        "Adjoint" if adjoint else "OuterPrepareDoubleFactorization",
        width + 1,
        calls=(
            CostCall(replace(uniform, name="Adjoint") if adjoint else uniform),
            CostCall(
                source_chemistry_qroam_cost(
                    auxiliaries + 1, width + alias_bits, adjoint=adjoint
                )
            ),
            CostCall(_TOFFOLI, alias_bits + width),
        ),
    )


def source_double_factorization_indexed_data_cost(
    auxiliaries: int,
    spin_orbitals: int,
    eigenvalues: int,
    rotation_bits: int = 8,
    *,
    adjoint: bool = False,
) -> ComponentCost:
    """Count the lookup of rank, offset, rotation data and the nonzero flag."""
    width = _auxiliary_width(auxiliaries)
    _, rank, offset = _df_address_widths(spin_orbitals, eigenvalues)
    _require_positive_precision(rotation_bits)
    target = rank + offset + rotation_bits + 1
    return ComponentCost(
        "Adjoint" if adjoint else "OutputIndexedData",
        width + target,
        calls=(
            CostCall(
                source_chemistry_qroam_cost(auxiliaries + 1, target, adjoint=adjoint)
            ),
        ),
    )

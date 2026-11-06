# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 ProgRotGateArray.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/chemistry/df/select_bloq.py

"""Declared basis-rotation cost for double-factorized chemistry."""

from ...compiler._component_trace.resource_model import ComponentCost, CostCall
from .chemistry_qroam_model import source_chemistry_qroam_cost
from .double_factorization_prepare_model import _TOFFOLI, _df_address_widths


def source_double_factorization_programmed_rotation_cost(
    spin_orbitals: int,
    eigenvalues: int,
    rotation_bits: int,
    *,
    adjoint: bool = False,
) -> ComponentCost:
    """Count the source offset and rotations with the directional QROAM."""
    spatial, rank, offset = _df_address_widths(spin_orbitals, eigenvalues)
    if type(rotation_bits) is not int or rotation_bits < 2:
        raise ValueError("DF programmed rotation requires at least two bits")
    count = offset - 1 + spin_orbitals * (rotation_bits - 2)
    if count < 0:
        raise ValueError("The locked rotation formula gives a negative gate count")
    target = spatial * rotation_bits
    return ComponentCost(
        "Adjoint" if adjoint else "ProgRotGateArray",
        offset + rank + target + 1 + spin_orbitals,
        calls=(
            CostCall(_TOFFOLI, count),
            CostCall(
                source_chemistry_qroam_cost(
                    eigenvalues + spatial, target, adjoint=adjoint
                )
            ),
        ),
    )

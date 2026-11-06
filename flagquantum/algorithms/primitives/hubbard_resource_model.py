# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from the call graphs in Qualtran v0.7.0 Hubbard hopping and interaction.
# https://github.com/quantumlib/Qualtran/tree/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/chemistry/trotter/hubbard

"""Declared logical costs for the locked square-lattice Hubbard terms."""

from math import isfinite

from ...compiler._component_trace.resource_classification import classify_resource_gate
from ...compiler._component_trace.resource_generalization import RotationCall
from ...compiler._component_trace.resource_model import (
    ComponentCost,
    CostCall,
    GateCost,
)
from ...compiler.registers.hamming_weight_phasing import (
    source_hamming_weight_phasing_cost,
)
from .fermionic_fourier import source_two_bit_fermionic_fourier_cost


def _source_rz_cost(angle: float) -> ComponentCost:
    if not isfinite(angle):
        raise ValueError("Hubbard rotation angle must be finite")
    family = classify_resource_gate(RotationCall("rz", angle))
    return ComponentCost(
        "Rz",
        1,
        GateCost(
            t=int(family == "t"),
            clifford=int(family == "clifford"),
            rotation=int(family == "rotation"),
        ),
    )


def source_hubbard_interaction_cost(
    length: int, angle: float, hubb_u: float, *, hamming_weight: bool = False
) -> ComponentCost:
    """Count one Rz per site or two batches of Hamming-weight phasing."""
    if type(length) is not int or length < 1:
        raise ValueError("Hubbard lattice length must be a positive integer")
    width = 2 * length
    exponent = angle * hubb_u
    if hamming_weight:
        phase = source_hamming_weight_phasing_cost(length**2 // 2, exponent)
        return ComponentCost("InteractionHWP", width, calls=(CostCall(phase, 2),))
    return ComponentCost(
        "Interaction", width, calls=(CostCall(_source_rz_cost(exponent), length**2),)
    )


def source_hubbard_hopping_plaquette_cost(kappa: float) -> ComponentCost:
    """Count four two-bit Fourier calls and two equal-angle Rz calls."""
    return ComponentCost(
        "HoppingPlaquette",
        4,
        calls=(
            CostCall(source_two_bit_fermionic_fourier_cost(0, 1), 4),
            CostCall(_source_rz_cost(kappa), 2),
        ),
    )


def source_hubbard_hopping_tile_cost(
    length: int,
    angle: float,
    tau: float = 1,
    *,
    pink: bool = True,
    hamming_weight: bool = False,
) -> ComponentCost:
    """Count the locked plaquette or batched phasing graph for a single color."""
    if type(length) is not int or length < 2 or length % 2:
        raise ValueError("Hubbard hopping requires a positive even lattice length")
    width = 2 * length
    color = "pink" if pink else "gold"
    kappa = tau * angle
    plaquettes = length**2 // 2
    if hamming_weight:
        return ComponentCost(
            f"HoppingTileHWP[{color}]",
            width,
            calls=(
                CostCall(source_two_bit_fermionic_fourier_cost(0, 1), 4 * plaquettes),
                CostCall(source_hamming_weight_phasing_cost(plaquettes, kappa), 2),
            ),
        )
    return ComponentCost(
        f"HoppingTile[{color}]",
        width,
        calls=(CostCall(source_hubbard_hopping_plaquette_cost(kappa), plaquettes),),
    )

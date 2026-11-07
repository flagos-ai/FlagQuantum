# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 SelectTHC and THCRotations.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/chemistry/thc/select_bloq.py

"""Declared THC resource models before table creation or gate lowering."""

from math import ceil

from ....compiler._component_trace.resource_model import (
    ComponentCost,
    CostCall,
    GateCost,
)


def thc_rotations_cost(
    num_mu: int,
    num_spin_orb: int,
    num_bits_theta: int,
    *,
    kr1: int = 1,
    kr2: int = 1,
    two_body_only: bool = False,
    adjoint: bool = False,
) -> ComponentCost:
    """Build the locked lookup and addition estimate for one THC rotation leaf."""
    if any(
        type(value) is not int or value < 1
        for value in (num_mu, num_spin_orb, kr1, kr2)
    ):
        raise ValueError(
            "THC dimensions and erasure block sizes must be positive integers"
        )
    if num_spin_orb % 2 or type(num_bits_theta) is not int or num_bits_theta < 2:
        raise ValueError(
            "THC requires paired spin orbitals and at least two angle bits"
        )
    orbitals = num_spin_orb // 2
    width = 1 + num_bits_theta + num_mu.bit_length() + orbitals
    if adjoint:
        if two_body_only:
            lookup_cost = ceil(num_mu / kr1) + ceil(num_spin_orb / (2 * kr1)) + kr1
        else:
            lookup_cost = ceil(num_mu / kr2) + kr2
    else:
        lookup_cost = num_mu + orbitals - 2
        if two_body_only:
            lookup_cost -= orbitals
    rotation_cost = num_spin_orb * (num_bits_theta - 2)
    return ComponentCost(
        f"THCRotations(two_body_only={two_body_only}, adjoint={adjoint})",
        width,
        GateCost(toffoli=rotation_cost + lookup_cost),
    )


def select_thc_cost(
    num_mu: int,
    num_spin_orb: int,
    num_bits_theta: int,
    keep_bitsize: int,
    *,
    kr1: int = 1,
    kr2: int = 1,
) -> ComponentCost:
    """Build the locked THC SELECT model from dimensions and erasure block sizes.

    The model includes the source signature, including the alias-sampling
    keep register and rotation flag, plus one shared angle register. The
    source rotation leaf declares Toffoli counts for lookup and phase-gradient
    addition. It has no executable decomposition or measurement count.
    """
    if type(keep_bitsize) is not int or keep_bitsize < 1:
        raise ValueError(
            "THC dimensions and erasure block sizes must be positive integers"
        )
    calls = (
        CostCall(
            thc_rotations_cost(
                num_mu,
                num_spin_orb,
                num_bits_theta,
                kr1=kr1,
                kr2=kr2,
                two_body_only=two_body_only,
                adjoint=adjoint,
            )
        )
        for two_body_only in (False, True)
        for adjoint in (False, True)
    )
    rotate_mu, unrotate_mu, rotate_nu, unrotate_nu = calls
    orbitals = num_spin_orb // 2
    address_width = num_mu.bit_length()

    # Four spin swaps and one address swap. Equal widths can occur in
    # several call sites and are counted separately here.
    spin_swap = ComponentCost(
        "CSwap(system)", 1 + 2 * orbitals, GateCost(cswap=orbitals)
    )
    address_swap = ComponentCost(
        "CSwap(address)", 1 + 2 * address_width, GateCost(cswap=address_width)
    )
    spin_flag_swap = ComponentCost("CSwap(spin flags)", 3, GateCost(cswap=1))
    reflect_one = ComponentCost("ApplyControlledZs(1)", 1 + orbitals)
    # The two-control reflection's QubitCount includes its AND work bit.
    reflect_two_ports = 2 + orbitals
    reflect_two = ComponentCost(
        "ApplyControlledZs(1,0)", reflect_two_ports + 1, GateCost(toffoli=1)
    )
    # Five selection flags, one rotation flag, two addresses and both systems.
    signature_width = 6 + keep_bitsize + 2 * address_width + num_spin_orb
    live_width = signature_width + num_bits_theta
    return ComponentCost(
        "SelectTHC",
        live_width,
        GateCost(clifford=1),
        (
            CostCall(spin_swap),
            rotate_mu,
            CostCall(reflect_one),
            unrotate_mu,
            CostCall(spin_swap),
            CostCall(spin_flag_swap),
            CostCall(address_swap),
            CostCall(spin_swap),
            rotate_nu,
            CostCall(reflect_two, bystander_qubits=live_width - reflect_two_ports),
            unrotate_nu,
            CostCall(spin_swap),
        ),
    )

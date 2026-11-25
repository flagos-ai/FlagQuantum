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
from ....compiler.registers.layout.thc_signatures import (
    thc_rotation_signature as _rotation_signature,
)
from ....compiler.registers.layout.thc_signatures import (
    thc_selection_signature as _selection_signature,
)
from ....core.ir.program.signatures import Signature
from ..chemistry_number_phase import ChemistryNumberPhase


def _spatial_orbitals(num_mu: int, num_spin_orb: int) -> int:
    if any(type(value) is not int or value < 1 for value in (num_mu, num_spin_orb)):
        raise ValueError("THC dimensions must be positive integers")
    if num_spin_orb % 2:
        raise ValueError("THC requires paired spin orbitals")
    return num_spin_orb // 2


def thc_rotations_signature(
    num_mu: int, num_spin_orb: int, num_bits_theta: int
) -> Signature:
    """Describe the source rotation leaf's shared data and through ports."""
    _spatial_orbitals(num_mu, num_spin_orb)
    if type(num_bits_theta) is not int or num_bits_theta < 2:
        raise ValueError("THC requires at least two angle bits")
    return _rotation_signature(num_mu, num_spin_orb, num_bits_theta)


def select_thc_signature(
    num_mu: int,
    num_spin_orb: int,
    keep_bitsize: int,
    *,
    control_value: int | None = None,
) -> Signature:
    """Describe the source selection order and its optional pass-through bit."""
    _spatial_orbitals(num_mu, num_spin_orb)
    if type(keep_bitsize) is not int or keep_bitsize < 1:
        raise ValueError("THC keep width must be a positive integer")
    if control_value is not None and (
        type(control_value) is not int or control_value not in (0, 1)
    ):
        raise ValueError("THC SELECT control value must be zero, one or None")
    return _selection_signature(
        num_mu, num_spin_orb, keep_bitsize, control_value=control_value
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
    signature = thc_rotations_signature(num_mu, num_spin_orb, num_bits_theta)
    if any(type(value) is not int or value < 1 for value in (kr1, kr2)):
        raise ValueError("THC erasure block sizes must be positive integers")
    orbitals = num_spin_orb // 2
    width = int(signature.n_qubits())
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
    control_value: int | None = None,
) -> ComponentCost:
    """Build the locked THC SELECT model from dimensions and erasure block sizes.

    The model includes the source signature, including the alias-sampling
    keep register and rotation flag, plus one shared angle register. The
    source rotation leaf declares Toffoli counts for lookup and phase-gradient
    addition. It has no executable decomposition or measurement count.
    """
    signature = select_thc_signature(
        num_mu, num_spin_orb, keep_bitsize, control_value=control_value
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
    reflect_one = ChemistryNumberPhase((1,), orbitals).resource_model(
        "ApplyControlledZs(1)"
    )
    # The two-control reflection's QubitCount includes its AND work bit.
    reflect_two_ports = 2 + orbitals
    reflect_two = ChemistryNumberPhase((1, 0), orbitals).resource_model(
        "ApplyControlledZs(1,0)"
    )
    live_width = int(signature.n_qubits()) + num_bits_theta
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

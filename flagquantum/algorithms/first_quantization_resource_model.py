# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 first-quantized SELECT and PREPARE resource declarations.
# https://github.com/quantumlib/Qualtran/tree/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/chemistry/pbc/first_quantization

"""Declared costs for first-quantized plane-wave SELECT and PREPARE."""

from dataclasses import replace
from math import ceil

from ..compiler._component_trace.resource_model import ComponentCost, CostCall, GateCost
from ..compiler.registers.arithmetic_resource_model import source_add_cost
from ..compiler.registers.uniform_preparation import source_uniform_preparation_cost
from .first_quantization import source_multiplexed_cswap_3d_cost
from .first_quantization_nu_resource_model import source_prepare_nu_cost


def source_select_t_cost(num_bits_p: int) -> ComponentCost:
    """Count the pinned kinetic SELECT declaration."""
    _require_positive(num_bits_p)
    return ComponentCost(
        "SelectTFirstQuantization",
        5 + 5 * num_bits_p,
        GateCost(toffoli=5 * (num_bits_p - 1) + 2),
    )


def source_apply_nuclear_phase_cost(
    num_bits_p: int, num_bits_nuc_pos: int
) -> ComponentCost:
    """Keep both branches of the source nuclear phase estimate."""
    _require_positive(num_bits_p, num_bits_nuc_pos)
    p, nuc = num_bits_p, num_bits_nuc_pos
    toffoli = 3 * (2 * p * nuc - p * (p + 1) - 1) if nuc > p else 3 * nuc * (nuc - 1)
    return ComponentCost(
        "ApplyNuclearPhase", 2 * nuc + 3 * p, GateCost(toffoli=toffoli)
    )


def source_select_uv_cost(
    num_bits_p: int, num_atoms: int, num_bits_nuc_pos: int
) -> ComponentCost:
    """Count the pinned potential SELECT call graph with fallback peak width."""
    _require_positive(num_bits_p, num_atoms, num_bits_nuc_pos)
    if num_bits_p < 2:
        raise ValueError(
            "Source SELECT UV qubit count requires at least two momentum bits"
        )
    p = num_bits_p
    atom_width = (num_atoms - 1).bit_length()
    return ComponentCost(
        "SelectUVFirstQuantization",
        2 + atom_width + num_bits_nuc_pos + 3 * (p + 1) + 6 * p,
        GateCost(toffoli=6 * (p + 1)),
        (
            CostCall(_signed_conversion_cost(p), repetitions=6),
            CostCall(source_add_cost(p + 1, p + 1), repetitions=6),
            CostCall(_signed_conversion_cost(p + 2), repetitions=6),
            CostCall(source_apply_nuclear_phase_cost(p, num_bits_nuc_pos)),
        ),
    )


def source_select_first_quantization_cost(
    num_bits_p: int,
    eta: int,
    num_atoms: int,
    *,
    m_param: int = 2**8,
    num_bits_nuc_pos: int = 16,
) -> ComponentCost:
    """Build the source SELECT cost graph without constructing an executable SELECT."""
    _require_positive(num_bits_p, eta, num_atoms, m_param, num_bits_nuc_pos)
    swap = source_multiplexed_cswap_3d_cost(eta, num_bits_p)
    kinetic = source_select_t_cost(num_bits_p)
    potential = source_select_uv_cost(num_bits_p, num_atoms, num_bits_nuc_pos)
    p = num_bits_p
    ports = (
        4  # T/U/V flags, i != j and plus.
        + 2 * (eta - 1).bit_length()
        + 3  # Momentum coordinate selector w.
        + 3 * p  # r, s and mu.
        + 3 * (p + 1)  # Momentum transfer.
        + (m_param - 1).bit_length()
        + (num_atoms - 1).bit_length()
        + 3 * p * eta
    )
    one_momentum = ports + 3 * p
    both_momenta = ports + 6 * p
    nuclear_position = both_momenta + num_bits_nuc_pos
    return ComponentCost(
        "SelectFirstQuantization",
        ports,
        calls=(
            CostCall(swap, bystander_qubits=one_momentum - swap.qubits),
            # QubitCount schedules T before the independent second exchange.
            CostCall(kinetic, bystander_qubits=one_momentum - kinetic.qubits),
            CostCall(swap, bystander_qubits=both_momenta - swap.qubits),
            CostCall(potential, bystander_qubits=nuclear_position - potential.qubits),
            CostCall(swap, bystander_qubits=both_momenta - swap.qubits),
            CostCall(swap, bystander_qubits=one_momentum - swap.qubits),
        ),
    )


def _signed_conversion_cost(width: int) -> ComponentCost:
    return ComponentCost(
        "SignedIntegerToTwosComplement", width, GateCost(toffoli=width - 2)
    )


def _require_positive(*dimensions: int) -> None:
    if any(type(dimension) is not int or dimension < 1 for dimension in dimensions):
        raise ValueError("First-quantized dimensions must be positive integers")


def source_prepare_power_two_cost(bitsize: int) -> ComponentCost:
    """Keep the pinned unary power-of-two preparation's declared cost."""
    _require_positive(bitsize)
    return ComponentCost(
        "PreparePowerTwoState",
        max(bitsize, 3),
        GateCost(toffoli=bitsize - 2),
    )


def source_prepare_ij_cost(eta: int, num_bits_rot_aa: int = 8) -> ComponentCost:
    """Count the source distinct-particle preparation declaration."""
    _require_positive(eta, num_bits_rot_aa)
    width = (eta - 1).bit_length()
    return ComponentCost(
        "UniformSuperpostionIJFirstQuantization",
        max(2 * width, 3),
        GateCost(toffoli=7 * width + 4 * num_bits_rot_aa - 18),
    )


def source_prepare_t_cost(
    num_bits_p: int, *, decomposed: bool = False
) -> ComponentCost:
    """Report the pinned declared cost or the cost of its actual composite.

    The declared cost prices coordinate preparation at thirteen Toffoli.
    The composite instead calls uniform preparation on three values. Both
    reports use the composite's peak width.
    """
    power_two = source_prepare_power_two_cost(num_bits_p)
    ports = 2 + 2 * num_bits_p
    # QubitCount follows the composite even when QECGatesCost uses the
    # declared Toffolis. The w preparation holds four additional clean wires.
    if decomposed:
        coordinate_call = CostCall(
            source_uniform_preparation_cost(3),
            bystander_qubits=ports - 2,
        )
        width = ports
    else:
        coordinate_call = CostCall(
            ComponentCost("Toffoli", 3, GateCost(toffoli=1)), repetitions=13
        )
        width = ports + 4
    return ComponentCost(
        "PrepareTFirstQuantization",
        width,
        calls=(
            coordinate_call,
            CostCall(power_two, repetitions=2, bystander_qubits=ports - num_bits_p),
        ),
    )


def source_prepare_zeta_cost(
    num_atoms: int, lambda_zeta: int, *, adjoint: bool = False
) -> ComponentCost:
    """Estimate preparation of the nuclear index, or its source adjoint.

    The single big-endian index is a through register. The source has a cost
    declaration without a gate decomposition or work-register contract.
    """
    _require_positive(num_atoms)
    if type(lambda_zeta) is not int or lambda_zeta < 0:
        raise ValueError("Total nuclear charge must be a nonnegative integer")
    width = (num_atoms - 1).bit_length()
    # Keep the source's floating square-root rounding in the adjoint model.
    toffoli = ceil(lambda_zeta**0.5) if adjoint else lambda_zeta
    return ComponentCost("PrepareZetaState", max(width, 3), GateCost(toffoli=toffoli))


def source_prepare_uv_cost(
    num_bits_p: int,
    num_atoms: int,
    m_param: int,
    lambda_zeta: int,
    *,
    adjoint: bool = False,
) -> ComponentCost:
    """Combine the source momentum-transfer and nuclear-index cost graphs.

    Both child calls pass through disjoint register groups. The index is
    live during momentum preparation and vice versa. The source adjoint
    reverses the calls and uses each child's adjoint resource declaration.
    """
    nu = source_prepare_nu_cost(num_bits_p, m_param, adjoint=adjoint)
    zeta = source_prepare_zeta_cost(num_atoms, lambda_zeta, adjoint=adjoint)
    nu_ports = 4 * num_bits_p + (m_param - 1).bit_length() + 4
    index_width = (num_atoms - 1).bit_length()
    calls = (
        CostCall(nu, bystander_qubits=index_width),
        CostCall(zeta, bystander_qubits=nu_ports),
    )
    return ComponentCost(
        "PrepareUVFirstQuantization†" if adjoint else "PrepareUVFirstQuantization",
        nu_ports + index_width,
        calls=calls[::-1] if adjoint else calls,
    )


def source_prepare_tuv_cost(
    num_bits_t: int, eta: int, lambda_zeta: int, num_bits_rot_aa: int = 8
) -> ComponentCost:
    """Count the pinned term-selection preparation on two through flags."""
    _require_positive(num_bits_t, eta, num_bits_rot_aa)
    if type(lambda_zeta) is not int or lambda_zeta < 0:
        raise ValueError("Total nuclear charge must be a nonnegative integer")
    width = (eta + 2 * lambda_zeta - 1).bit_length()
    return ComponentCost(
        "PrepareTUVSuperpositions",
        3,
        GateCost(toffoli=num_bits_t + 4 * width + 2 * num_bits_rot_aa - 12),
    )


def source_prepare_first_quantization_cost(
    num_bits_p: int,
    eta: int,
    num_atoms: int,
    lambda_zeta: int,
    *,
    m_param: int = 2**8,
    num_bits_t: int = 16,
    num_bits_rot_aa: int = 8,
    adjoint: bool = False,
) -> ComponentCost:
    """Estimate the source top-level PREPARE and its live through registers."""
    term = source_prepare_tuv_cost(num_bits_t, eta, lambda_zeta, num_bits_rot_aa)
    if eta < 2:
        raise ValueError("Source PREPARE requires at least two particles")
    particles = source_prepare_ij_cost(eta, num_bits_rot_aa)
    kinetic = source_prepare_t_cost(num_bits_p)
    potential = source_prepare_uv_cost(
        num_bits_p, num_atoms, m_param, lambda_zeta, adjoint=adjoint
    )
    # The groups are disjoint. plus_t is wired straight through the composite.
    particle_ports = 2 * (eta - 1).bit_length()
    kinetic_ports = 2 + 2 * num_bits_p
    ports = 3 + particle_ports + kinetic_ports + potential.qubits
    if adjoint:
        # These three source adjoints retain their forward cost declarations.
        term = replace(term, name=f"{term.name}†")
        particles = replace(particles, name=f"{particles.name}†")
        kinetic = replace(kinetic, name=f"{kinetic.name}†")
    calls = (
        CostCall(term, bystander_qubits=ports - 2),
        CostCall(particles, bystander_qubits=ports - particle_ports),
        CostCall(kinetic, bystander_qubits=ports - kinetic_ports),
        CostCall(potential, bystander_qubits=ports - potential.qubits),
    )
    return ComponentCost(
        "PrepareFirstQuantization†" if adjoint else "PrepareFirstQuantization",
        ports,
        calls=calls[::-1] if adjoint else calls,
    )

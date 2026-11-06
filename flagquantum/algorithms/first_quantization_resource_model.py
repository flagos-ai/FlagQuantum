# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 first-quantized SELECT and PREPARE resource declarations.
# https://github.com/quantumlib/Qualtran/tree/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/chemistry/pbc/first_quantization

"""Declared costs for first-quantized plane-wave SELECT and PREPARE."""

from ..compiler._component_trace.resource_model import ComponentCost, CostCall, GateCost
from ..compiler.registers.arithmetic_resource_model import source_add_cost
from .first_quantization import source_multiplexed_cswap_3d_cost


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
            ComponentCost(
                "PrepareUniformSuperposition(3)",
                6,
                GateCost(and_bloq=5, clifford=35, rotation=2, measurement=5),
            ),
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

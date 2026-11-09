# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 first-quantized projectile cost declarations.
# https://github.com/quantumlib/Qualtran/tree/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/chemistry/pbc/first_quantization/projectile

"""Resource models for first-quantized chemistry with a quantum projectile."""

from ..compiler._component_trace.resource_model import (
    ComponentCost,
    CostCall,
    GateCost,
    estimate_component_resources,
)
from ..compiler.registers.arithmetic.arithmetic_resource_model import source_add_cost
from .first_quantization import (
    source_controlled_multiplexed_cswap_3d_cost,
    source_multiplexed_cswap_3d_cost,
)
from .first_quantization_nu_resource_model import (
    source_projectile_prepare_nu_adjoint_gate_cost,
    source_projectile_prepare_nu_cost,
)
from .first_quantization_resource_model import (
    _require_positive,
    _signed_conversion_cost,
    source_apply_nuclear_phase_cost,
    source_prepare_ij_cost,
    source_prepare_zeta_cost,
)


def source_projectile_power_two_cost(
    bitsize_n: int, bitsize_p: int, *, adjoint: bool = False
) -> ComponentCost:
    """Estimate the source unary preparation on an n-bit through register."""
    _require_positive(bitsize_n, bitsize_p)
    if bitsize_n < bitsize_p:
        raise ValueError(
            "Projectile momentum width must be at least the electron width"
        )
    toffoli = bitsize_n - 2
    if not adjoint:
        toffoli += bitsize_n - bitsize_p
    return ComponentCost(
        "PreparePowerTwoStateWithProj", max(3, bitsize_n), GateCost(toffoli=toffoli)
    )


def source_projectile_prepare_t_cost(
    num_bits_p: int, num_bits_n: int, *, adjoint: bool = False
) -> ComponentCost:
    """Price the two coordinate words and two unary words in the source graph."""
    unary = source_projectile_power_two_cost(num_bits_n, num_bits_p, adjoint=adjoint)
    # The coordinate inequalities have no adjoint Toffoli charge.
    coordinate_toffolis = 15 if adjoint else 31
    return ComponentCost(
        "PrepareTFirstQuantizationWithProj",
        4 + 2 * num_bits_n,
        calls=(
            CostCall(
                ComponentCost("Toffoli", 3, GateCost(toffoli=1)),
                repetitions=coordinate_toffolis,
            ),
            CostCall(unary, repetitions=2),
        ),
    )


def source_projectile_select_t_cost(num_bits_n: int) -> ComponentCost:
    """Count the source kinetic selector with two three-bit coordinate words."""
    _require_positive(num_bits_n)
    return ComponentCost(
        "SelectTFirstQuantizationWithProj",
        9 + 5 * num_bits_n,
        GateCost(toffoli=5 * (num_bits_n - 1) + 3),
    )


def source_projectile_select_uv_cost(
    num_bits_p: int, num_bits_n: int, num_atoms: int, num_bits_nuc_pos: int
) -> ComponentCost:
    """Count the potential selector on three unequal-width p and q axes."""
    _require_positive(num_bits_p, num_bits_n, num_atoms, num_bits_nuc_pos)
    if min(num_bits_p, num_bits_n) < 2:
        raise ValueError(
            "Source projectile SELECT UV requires at least two momentum bits"
        )
    p, n = num_bits_p, num_bits_n
    # Keep the independent electronic and projectile arithmetic families.
    # The call-graph query combines structurally equal children.
    calls = (
        CostCall(_signed_conversion_cost(p), repetitions=3),
        CostCall(_signed_conversion_cost(n), repetitions=3),
        CostCall(source_add_cost(p + 1, p + 1), repetitions=3),
        CostCall(source_add_cost(n + 1, n + 1), repetitions=3),
        CostCall(
            ComponentCost("Toffoli", 3, GateCost(toffoli=1)),
            repetitions=3 * (p + n + 2),
        ),
        CostCall(_signed_conversion_cost(p + 2), repetitions=3),
        CostCall(_signed_conversion_cost(n + 2), repetitions=3),
        CostCall(source_apply_nuclear_phase_cost(n, num_bits_nuc_pos)),
    )
    return ComponentCost(
        "SelectUVFirstQuantizationWithProj",
        2 + (num_atoms - 1).bit_length() + num_bits_nuc_pos + 3 * (n + 1) + 3 * (p + n),
        calls=calls,
    )


def source_projectile_prepare_uv_cost(
    num_bits_p: int, num_bits_n: int, num_atoms: int, m_param: int, lambda_zeta: int
) -> ComponentCost:
    """Model forward momentum transfer and nuclear-index preparation."""
    nu = source_projectile_prepare_nu_cost(num_bits_p, num_bits_n, m_param)
    zeta = source_prepare_zeta_cost(num_atoms, lambda_zeta)
    nu_ports = 4 * num_bits_n + (m_param - 1).bit_length() + 4
    index_width = (num_atoms - 1).bit_length()
    return ComponentCost(
        "PrepareUVFirstQuantizationWithProj",
        nu_ports + index_width,
        calls=(
            CostCall(nu, bystander_qubits=index_width),
            CostCall(zeta, bystander_qubits=nu_ports),
        ),
    )


def source_projectile_prepare_uv_adjoint_gate_cost(
    num_bits_p: int, num_bits_n: int, num_atoms: int, lambda_zeta: int
) -> GateCost:
    """Count adjoint gates separately from the undefined source peak width."""
    nu = source_projectile_prepare_nu_adjoint_gate_cost(num_bits_p, num_bits_n)
    zeta = source_prepare_zeta_cost(num_atoms, lambda_zeta, adjoint=True)
    return nu.plus(zeta.gates, 1)


def source_projectile_select_first_quantization_cost(
    num_bits_p: int,
    num_bits_n: int,
    eta: int,
    num_atoms: int,
    *,
    m_param: int = 2**8,
    num_bits_nuc_pos: int = 16,
) -> ComponentCost:
    """Estimate the source top-level projectile SELECT graph."""
    _require_positive(num_bits_p, num_bits_n, eta, num_atoms, m_param, num_bits_nuc_pos)
    p, n = num_bits_p, num_bits_n
    controlled_exchange = source_controlled_multiplexed_cswap_3d_cost(eta, p, n, 1)
    electron_exchange = source_multiplexed_cswap_3d_cost(eta, p)
    coordinate_swap = ComponentCost("CSwap", 2 * n + 1, GateCost(cswap=n))
    kinetic = source_projectile_select_t_cost(n)
    potential = source_projectile_select_uv_cost(p, n, num_atoms, num_bits_nuc_pos)
    ports = (
        6  # Four Hamiltonian flags, i != j and plus_t.
        + 2 * (eta - 1).bit_length()
        + 6  # Two three-bit coordinate selectors.
        + 3 * n  # r, s and mu.
        + 3 * (n + 1)  # Momentum-transfer coordinates.
        + (m_param - 1).bit_length()
        + (num_atoms - 1).bit_length()
        + 3 * (eta * p + n)  # Electron and projectile systems.
    )
    one_momentum = ports + 3 * n
    both_momenta = one_momentum + 3 * p
    # QubitCount postpones the q and rl allocations until first use and
    # releases each projectile coordinate immediately after its last swap.
    return ComponentCost(
        "SelectFirstQuantizationWithProj",
        ports,
        calls=(
            CostCall(
                controlled_exchange,
                bystander_qubits=one_momentum - controlled_exchange.qubits,
            ),
            *(
                CostCall(
                    coordinate_swap,
                    bystander_qubits=one_momentum - coordinate_swap.qubits,
                )
                for _ in range(3)
            ),
            CostCall(kinetic, bystander_qubits=one_momentum - kinetic.qubits),
            CostCall(
                electron_exchange,
                bystander_qubits=both_momenta - electron_exchange.qubits,
            ),
            CostCall(
                potential,
                bystander_qubits=both_momenta + num_bits_nuc_pos - potential.qubits,
            ),
            CostCall(
                controlled_exchange,
                bystander_qubits=both_momenta - controlled_exchange.qubits,
            ),
            *(
                CostCall(
                    coordinate_swap,
                    bystander_qubits=both_momenta - axis * n - coordinate_swap.qubits,
                )
                for axis in range(3)
            ),
            CostCall(
                electron_exchange,
                bystander_qubits=ports + 3 * p - electron_exchange.qubits,
            ),
        ),
    )


def source_projectile_prepare_tuv_cost(
    num_bits_t: int, *, adjoint: bool = False
) -> ComponentCost:
    """Count the source term preparation on nine through wires."""
    _require_positive(num_bits_t)
    return ComponentCost(
        "PrepareTUVSuperpositions",
        9,
        GateCost(toffoli=0 if adjoint else 6 * num_bits_t + 2),
    )


def source_projectile_prepare_first_quantization_cost(
    num_bits_p: int,
    num_bits_n: int,
    eta: int,
    num_atoms: int,
    lambda_zeta: int,
    *,
    m_param: int = 2**8,
    num_bits_t: int = 16,
    num_bits_rot_aa: int = 8,
) -> ComponentCost:
    """Estimate the forward top-level projectile preparation."""
    particles = source_prepare_ij_cost(eta, num_bits_rot_aa)
    if eta < 2:
        raise ValueError("Source projectile PREPARE requires at least two particles")
    term = source_projectile_prepare_tuv_cost(num_bits_t)
    kinetic = source_projectile_prepare_t_cost(num_bits_p, num_bits_n)
    potential = source_projectile_prepare_uv_cost(
        num_bits_p, num_bits_n, num_atoms, m_param, lambda_zeta
    )
    particle_ports = 2 * (eta - 1).bit_length()
    # The disjoint child groups leave plus_t live throughout.
    ports = term.qubits + particle_ports + kinetic.qubits + potential.qubits + 1
    return ComponentCost(
        "PrepareFirstQuantizationWithProj",
        ports,
        calls=(
            CostCall(term, bystander_qubits=ports - term.qubits),
            CostCall(particles, bystander_qubits=ports - particle_ports),
            CostCall(kinetic, bystander_qubits=ports - kinetic.qubits),
            CostCall(potential, bystander_qubits=ports - potential.qubits),
        ),
    )


def source_projectile_prepare_first_quantization_adjoint_gate_cost(
    num_bits_p: int,
    num_bits_n: int,
    eta: int,
    num_atoms: int,
    lambda_zeta: int,
    *,
    num_bits_rot_aa: int = 8,
) -> GateCost:
    """Count adjoint gates independently of the invalid nested source width."""
    particles = source_prepare_ij_cost(eta, num_bits_rot_aa)
    if eta < 2:
        raise ValueError("Source projectile PREPARE requires at least two particles")
    potential = source_projectile_prepare_uv_adjoint_gate_cost(
        num_bits_p, num_bits_n, num_atoms, lambda_zeta
    )
    kinetic = estimate_component_resources(
        source_projectile_prepare_t_cost(num_bits_p, num_bits_n, adjoint=True)
    )
    # The adjoint term preparation declares zero gates at every precision.
    return potential.plus(kinetic.gates, 1).plus(particles.gates, 1)

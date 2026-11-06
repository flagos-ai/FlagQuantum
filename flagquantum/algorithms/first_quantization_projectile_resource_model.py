# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 first-quantized projectile cost declarations.
# https://github.com/quantumlib/Qualtran/tree/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/chemistry/pbc/first_quantization/projectile

"""Resource models for first-quantized chemistry with a quantum projectile."""

from ..compiler._component_trace.resource_model import ComponentCost, CostCall, GateCost
from ..compiler.registers.arithmetic_resource_model import source_add_cost
from .first_quantization_nu_resource_model import (
    source_projectile_prepare_nu_adjoint_gate_cost,
    source_projectile_prepare_nu_cost,
)
from .first_quantization_resource_model import (
    _require_positive,
    _signed_conversion_cost,
    source_apply_nuclear_phase_cost,
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

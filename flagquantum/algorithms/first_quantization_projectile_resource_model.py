# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 first-quantized projectile cost declarations.
# https://github.com/quantumlib/Qualtran/tree/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/chemistry/pbc/first_quantization/projectile

"""Resource models for first-quantized chemistry with a quantum projectile."""

from ..compiler._component_trace.resource_model import ComponentCost, CostCall, GateCost
from .first_quantization_resource_model import _require_positive


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

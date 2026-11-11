# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 GeneralizedQSP.build_call_graph.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/qsp/generalized_qsp.py

"""Declared finite GQSP component calls, independent of physical emission."""

from ...compiler._component_trace.resource_model import ComponentCost, CostCall


def source_generalized_qsp_cost(
    coefficient_count: int,
    system_width: int,
    *,
    negative_power: int = 0,
    signal_rotation: ComponentCost,
    zero_control_unitary: ComponentCost,
    unitary_adjoint: ComponentCost,
    controlled_adjoint: ComponentCost,
) -> ComponentCost:
    """Compose source call multiplicities with caller-declared child costs.

    Controlled child widths include the signal. The bare adjoint excludes it;
    the rotation child excludes system wires. Child costs must account for
    their own workspace. No synthesis price or U implementation is inferred.
    """
    if type(coefficient_count) is not int or coefficient_count < 2:
        raise ValueError("GQSP model requires at least two coefficients")
    if type(system_width) is not int or system_width < 0:
        raise ValueError("GQSP model system width must be a nonnegative integer")
    if type(negative_power) is not int or negative_power < 0:
        raise ValueError("GQSP model negative_power must be a nonnegative integer")
    degree = coefficient_count - 1
    calls = (
        CostCall(signal_rotation, coefficient_count, system_width),
        CostCall(zero_control_unitary, max(0, degree - negative_power)),
        CostCall(unitary_adjoint, max(0, negative_power - degree), 1),
        CostCall(controlled_adjoint, min(degree, negative_power)),
    )
    return ComponentCost(
        "GeneralizedQSP",
        system_width + 1,
        calls=tuple(call for call in calls if call.repetitions),
    )

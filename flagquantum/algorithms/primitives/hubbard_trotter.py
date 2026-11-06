# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 Hubbard second-order Trotter schedules.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/chemistry/trotter/hubbard/trotter_step.py

"""Resource schedules for second-order Hubbard plaquette evolution."""

from ...compiler._component_trace.resource_model import ComponentCost, CostCall
from .hubbard_resource_model import (
    source_hubbard_hopping_tile_cost,
    source_hubbard_interaction_cost,
)


def source_hubbard_suzuki_cost(
    length: int,
    hubb_u: float,
    timestep: float,
    hubb_t: float = 1,
    *,
    hamming_weight: bool = False,
    strip_layer: bool = False,
) -> ComponentCost:
    """Model the ordered interaction, pink and gold calls without an emitted circuit.

    Strip mode removes the leading interaction and doubles the trailing one
    for use when consecutive steps share that layer. The model uses the
    source signature and recursive width fallback for opaque leaves.
    """
    schedule = (
        (("pink", 0.5), ("gold", 1), ("pink", 0.5), ("interaction", 1))
        if strip_layer
        else (
            ("interaction", 0.5),
            ("pink", 0.5),
            ("gold", 1),
            ("pink", 0.5),
            ("interaction", 0.5),
        )
    )
    calls = []
    for term, coefficient in schedule:
        angle = 2 * coefficient * timestep
        if term == "interaction":
            cost = source_hubbard_interaction_cost(
                length, angle, hubb_u, hamming_weight=hamming_weight
            )
        else:
            cost = source_hubbard_hopping_tile_cost(
                length,
                angle,
                hubb_t,
                pink=term == "pink",
                hamming_weight=hamming_weight,
            )
        calls.append(CostCall(cost))
    return ComponentCost("HubbardSuzukiStep", 2 * length, calls=tuple(calls))

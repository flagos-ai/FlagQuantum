# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 CtrlSpecAnd and QECGatesCost.
# https://github.com/quantumlib/Qualtran/tree/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/mcmt

"""Concrete predicate models using the existing typed QEC query authority."""

from dataclasses import fields

from .._component_trace.qec_gate_cost import query_qec_gate_cost
from .._component_trace.resource_generalization import ControlPredicateCall
from .._component_trace.resource_model import ComponentCost, GateCost


def control_predicate_cost(
    values: tuple[int, ...],
    *,
    uncompute: bool = False,
    legacy_shims: bool = False,
) -> ComponentCost:
    """Model an owned prefix predicate, including controls and computed work.

    Both directions expose the same maximum live storage. Logical gate
    categories follow the selected source QEC profile, not emitted IR counts.
    """
    if type(uncompute) is not bool or type(legacy_shims) is not bool:
        raise ValueError("Predicate direction and cost profile must be booleans")
    cost = query_qec_gate_cost(
        ControlPredicateCall(values, uncompute), legacy_shims=legacy_shims
    )
    return ComponentCost(
        "ControlPredicate†" if uncompute else "ControlPredicate",
        2 * len(values) - 1,
        GateCost(
            **{field.name: int(getattr(cost, field.name)) for field in fields(GateCost)}
        ),
    )

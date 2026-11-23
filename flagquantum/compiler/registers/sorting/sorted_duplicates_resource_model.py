# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 arithmetic/lists/has_duplicates.py.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/arithmetic/lists/has_duplicates.py

"""Original duplicate-predicate counts and phase-specific logical lifetimes."""

from dataclasses import fields
from typing import cast

from ..._component_trace.qec_gate_cost import QECGateCost
from ..._component_trace.resource_model import ComponentCost, CostCall, GateCost
from ..._component_trace.symbolic_math import SymbolicInt, smax
from ..comparison.half_ordering import (
    source_half_comparison_gate_cost,
    source_half_comparison_resource_formula,
)
from ..control.multi_control import (
    source_multi_control_x_cost,
    source_multi_control_x_resource_formula,
)
from ..resources.symbolic_resource_model import SymbolicResourceFormula


def source_sorted_duplicates_gate_cost(
    length: SymbolicInt,
    width: SymbolicInt,
    *,
    signed: bool = False,
    controlled: bool = False,
) -> QECGateCost:
    """Query original declarations independently of failed source peak queries."""
    if type(controlled) is not bool:
        raise ValueError("Duplicate model controlled mode must be boolean")
    forward = source_half_comparison_gate_cost(width, signed=signed)
    reverse = source_half_comparison_gate_cost(width, signed=signed, adjoint=True)
    controls = length - 1 + int(controlled)
    shell = source_multi_control_x_resource_formula(controls)
    return (
        QECGateCost(clifford=1)
        .plus(shell, 1)
        .plus(forward, length - 1)
        .plus(reverse, length - 1)
    )


def source_sorted_duplicates_resource_formula(
    length: SymbolicInt,
    width: SymbolicInt,
    *,
    signed: bool = False,
    controlled: bool = False,
) -> SymbolicResourceFormula:
    """Preserve source builder schedules or its symbolic-length child fallback.

    Failed uncontrolled-singleton and signed-symbolic source peaks remain
    explicit errors. Their original QEC categories are independently queryable.
    """
    if isinstance(length, (bool, float)) or isinstance(length, int) and length < 0:
        raise ValueError("Duplicate model length must be nonnegative or symbolic")
    gates = source_sorted_duplicates_gate_cost(
        length, width, signed=signed, controlled=controlled
    )
    ports = length * width + 1 + int(controlled)
    if type(length) is int and length <= 1:
        if not controlled:
            raise ValueError(
                "Original uncontrolled singleton peak fails on the empty controls port"
            )
        return SymbolicResourceFormula(
            gates.and_bloq, gates.clifford, gates.measurement, smax(0, ports)
        )
    half = source_half_comparison_resource_formula(width, signed=signed)
    controls = length - 1 + int(controlled)
    mcx = source_multi_control_x_resource_formula(controls)
    if type(length) is int:
        edges = length - 1
        snapshots = [ports + (i + 1) * (width + 2) for i in range(edges)]
        comparisons = [
            ports - 2 * width + i * (width + 2) + half.logical_qubits
            for i in range(edges)
        ]
        reduction = ports + edges * (width + 2) - controls - 1 + mcx.logical_qubits
        peak = smax(0, *snapshots, *comparisons, reduction)
    else:
        peak = smax(ports, half.logical_qubits, mcx.logical_qubits)
    return SymbolicResourceFormula(
        gates.and_bloq, gates.clifford, gates.measurement, peak
    )


def source_sorted_duplicates_cost(
    length: int, width: int, *, signed: bool = False, controlled: bool = False
) -> ComponentCost:
    """Compose the four original direct calls with their full scheduled peak."""
    if type(length) is not int or type(width) is not int:
        raise ValueError("Concrete duplicate model requires Python integer dimensions")
    root = source_sorted_duplicates_resource_formula(
        length, width, signed=signed, controlled=controlled
    )
    calls = []
    for adjoint in (False, True):
        half = source_half_comparison_resource_formula(
            width, signed=signed, adjoint=adjoint
        )
        gates = GateCost(
            **{f.name: cast(int, getattr(half, f.name)) for f in fields(GateCost)}
        )
        calls.append(
            CostCall(
                ComponentCost(
                    "HalfLessThanAdjoint" if adjoint else "HalfLessThan",
                    cast(int, half.logical_qubits),
                    gates,
                ),
                length - 1,
            )
        )
    calls.append(
        CostCall(source_multi_control_x_cost((1,) * (length - 1 + int(controlled))))
    )
    calls.append(
        CostCall(
            ComponentCost(
                "CNOT" if controlled else "X",
                2 if controlled else 1,
                GateCost(clifford=1),
            )
        )
    )
    return ComponentCost(
        "HasDuplicates", cast(int, root.logical_qubits), calls=tuple(calls)
    )

# Copyright 2023 Google LLC
# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 UnaryIterationGate.build_call_graph and QubitCount.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/multiplexers/unary_iteration_bloq.py
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/resource_counting/_qubit_counts.py

"""Compose indexed child models with per-call live parent predicate widths."""

from math import prod

from .._component_trace.resource_model import ComponentCost, CostCall, GateCost
from .indexed_component import indexed_component_workspace


def _prefix_depth(index: int, extent: int) -> int:
    # A populated node splits only when its right half starts below extent.
    return sum(
        ((index >> (bit + 1)) << (bit + 1)) + (1 << bit) < extent
        for bit in range((extent - 1).bit_length())
    )


def indexed_component_cost(
    axis_widths: tuple[int, ...],
    extents: tuple[int, ...],
    children: tuple[ComponentCost, ...],
    *,
    target_width: int,
    control_count: int = 0,
    child_target_width: int | None = None,
    adjoint: bool = False,
) -> ComponentCost:
    """Aggregate controlled THRU children without emitting or executing gates.

    Each child includes one positive predicate and child_target_width target
    ports, plus its own declared work. The child direction is caller-owned.
    Parent target ports not used by that child remain bystanders. Preserve
    separate child models even when their costs or gate identities coincide.
    """
    indexed_component_workspace(axis_widths, control_count)
    if len(axis_widths) != len(extents) or any(
        type(extent) is not int or not 1 <= extent <= 1 << width
        for width, extent in zip(axis_widths, extents, strict=True)
    ):
        raise ValueError("Indexed model extents must fit their address words")
    if control_count == 0 and extents[0] == 1:
        raise ValueError(
            "Source uncontrolled singleton first axis has no model traversal"
        )
    if type(target_width) is not int or target_width < 0 or type(adjoint) is not bool:
        raise ValueError(
            "Indexed model needs nonnegative target width and boolean direction"
        )
    if child_target_width is None:
        child_target_width = target_width
    if (
        type(child_target_width) is not int
        or not 0 <= child_target_width <= target_width
    ):
        raise ValueError("Child target width must fit the parent target ports")
    rows = prod(extents)
    if len(children) != rows:
        raise ValueError(
            "Indexed model requires one controlled child per populated index"
        )
    child_ports = child_target_width + 1
    if any(
        type(child.qubits) is not int or child.qubits < child_ports
        for child in children
    ):
        raise ValueError("Every child model must include its controlled THRU ports")
    signature = control_count + sum(axis_widths) + target_width
    prefix = max(0, control_count - 1)
    traversal_ands = rows - (1 if control_count else 2)
    ands = traversal_ands + prefix
    peak_work = sum((extent - 1).bit_length() for extent in extents) + control_count - 1
    calls = []
    order = reversed(range(rows)) if adjoint else range(rows)
    for flat_index in order:
        remaining = flat_index
        active = prefix - int(control_count == 0)
        for extent in reversed(extents):
            remaining, index = divmod(remaining, extent)
            active += _prefix_depth(index, extent)
        calls.append(
            CostCall(
                children[flat_index], bystander_qubits=signature + active - child_ports
            )
        )
    return ComponentCost(
        "IndexedComponent†" if adjoint else "IndexedComponent",
        signature + peak_work,
        GateCost(
            and_bloq=ands,
            clifford=2 * traversal_ands + prefix + 2 * int(control_count == 0),
            measurement=ands,
        ),
        tuple(calls),
    )

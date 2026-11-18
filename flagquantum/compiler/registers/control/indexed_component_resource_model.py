# Copyright 2023 Google LLC
# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 UnaryIterationGate.build_call_graph and QubitCount.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/multiplexers/unary_iteration_bloq.py
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/resource_counting/_qubit_counts.py

"""Compose indexed child models with per-call live parent predicate widths."""

from collections.abc import Callable
from math import prod

from ..._component_trace.resource_model import ComponentCost, CostCall, GateCost
from .control_predicate_resource_model import control_predicate_cost
from .indexed_component import indexed_component_workspace


def _prefix_depth(index: int, extent: int) -> int:
    # A populated node splits only when its right half starts below extent.
    return sum(
        ((index >> (bit + 1)) << (bit + 1)) + (1 << bit) < extent
        for bit in range((extent - 1).bit_length())
    )


def _folded_calls(
    axis_widths: tuple[int, ...],
    extents: tuple[int, ...],
    control_count: int,
    break_early: Callable[[tuple[int, ...], int, int], bool],
) -> tuple[list[tuple[int, int]], int]:
    """Plan retained source calls and live predicates without emitting gates."""
    if break_early((), 0, extents[0]):
        raise ValueError("Folded first-axis root requires a zero-selection recipe")
    calls: list[tuple[int, int]] = []
    ands = -int(control_count == 0)

    def axis(depth: int, indices: tuple[int, ...], flat: int, active: int) -> None:
        extent = extents[depth]

        def node(left: int, right: int, live: int) -> None:
            nonlocal ands
            if left >= extent:
                return
            if right <= extent and (
                right - left == 1 or break_early(indices, left, right)
            ):
                index = flat * extent + left
                if depth + 1 == len(extents):
                    calls.append((index, live))
                else:
                    axis(depth + 1, (*indices, left), index, live)
                return
            middle = (left + right) // 2
            if extent <= middle:
                node(left, middle, live)
                return
            ands += 1
            node(left, middle, live + 1)
            node(middle, right, live + 1)

        node(0, 1 << axis_widths[depth], active)

    prefix = max(0, control_count - 1) - int(control_count == 0)
    axis(0, (), 0, prefix)
    return calls, ands


def indexed_component_cost(
    axis_widths: tuple[int, ...],
    extents: tuple[int, ...],
    children: tuple[ComponentCost, ...],
    *,
    target_width: int,
    control_count: int = 0,
    child_target_width: int | None = None,
    adjoint: bool = False,
    legacy_shims: bool = False,
    break_early: Callable[[tuple[int, ...], int, int], bool] | None = None,
    zero_selection: ComponentCost | None = None,
) -> ComponentCost:
    """Aggregate controlled THRU children without emitting or executing gates.

    Child models must use the same QEC profile as this parent.
    Each child includes one positive predicate and child_target_width target
    ports, plus its own declared work. The child direction is caller-owned.
    Parent target ports not used by that child remain bystanders. Preserve
    separate child models even when their costs or gate identities coincide.
    Early folding retains only the representative child's model and its live
    predicate depth. The callback has the same contract as the native emitter.
    An explicit root-fold model covers the entire parent signature and the
    zero-selection recipe's storage, independently of the unused child table.
    """
    indexed_component_workspace(axis_widths, control_count)
    if type(legacy_shims) is not bool:
        raise ValueError("Indexed cost profile must be a boolean")
    if len(axis_widths) != len(extents) or any(
        type(extent) is not int or not 1 <= extent <= 1 << width
        for width, extent in zip(axis_widths, extents, strict=True)
    ):
        raise ValueError("Indexed model extents must fit their address words")
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
    signature = control_count + sum(axis_widths) + target_width
    if not any(axis_widths) or (
        break_early is not None and break_early((), 0, extents[0])
    ):
        if zero_selection is None:
            raise ValueError("Folded first-axis root requires a zero-selection recipe")
        if type(zero_selection.qubits) is not int or zero_selection.qubits < signature:
            raise ValueError(
                "Zero-selection model must cover the full parent signature"
            )
        return ComponentCost(
            "IndexedComponent†" if adjoint else "IndexedComponent",
            signature,
            calls=(CostCall(zero_selection),),
        )
    if control_count == 0 and extents[0] == 1:
        raise ValueError(
            "Source uncontrolled singleton first axis has no model traversal"
        )
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
    prefix = max(0, control_count - 1)
    traversal_ands = rows - (1 if control_count else 2)
    peak_work = sum((extent - 1).bit_length() for extent in extents) + control_count - 1
    planned_calls = []
    if break_early is None:
        for flat_index in range(rows):
            remaining = flat_index
            active = prefix - int(control_count == 0)
            for extent in reversed(extents):
                remaining, index = divmod(remaining, extent)
                active += _prefix_depth(index, extent)
            planned_calls.append((flat_index, active))
    else:
        planned_calls, traversal_ands = _folded_calls(
            axis_widths, extents, control_count, break_early
        )
        peak_work = max(active for _, active in planned_calls)
    order = reversed(planned_calls) if adjoint else iter(planned_calls)
    calls = tuple(
        CostCall(
            children[flat_index], bystander_qubits=signature + active - child_ports
        )
        for flat_index, active in order
    )
    # Forward traversal computes (1,0) and erases (1,1); adjoint swaps
    # their directions. The source has one explicit CNOT per split.
    gates = GateCost(clifford=traversal_ands + 2 * int(control_count == 0))
    for values, reverse, count in (
        ((1, 0), adjoint, traversal_ands),
        ((1, 1), not adjoint, traversal_ands),
        ((1, 1), False, prefix),
        ((1, 1), True, prefix),
    ):
        gates = gates.plus(
            control_predicate_cost(
                values, uncompute=reverse, legacy_shims=legacy_shims
            ).gates,
            count,
        )
    return ComponentCost(
        "IndexedComponent†" if adjoint else "IndexedComponent",
        signature + peak_work,
        gates,
        calls,
    )

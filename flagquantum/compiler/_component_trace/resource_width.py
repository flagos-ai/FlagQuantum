# Copyright 2023 Google LLC
# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 QubitCount, Power and OnEach.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/resource_counting/_qubit_counts.py
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/basic_gates/on_each.py
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/basic_gates/power.py

"""Bind native resource identities to their owned logical width models."""

from __future__ import annotations

from collections.abc import Callable, MutableMapping

from ..registers.control.multi_and_width import multi_and_width
from ..registers.primitives.controlled_swap_width import controlled_swap_width
from .logical_width import query_logical_width
from .resource_decomposition import decomposed_resource_callees
from .resource_generalization import (
    BookkeepingCall,
    CliffordCall,
    ConjunctionCall,
    ControlledSwapCall,
    DeclaredResourceCalls,
    FixedTCall,
    OnEachCall,
    PhaseCall,
    PowerCall,
    ResourceCall,
    ZeroCostCall,
)
from .resource_model import ComponentCost, _active_calls, _component_width
from .symbolic_math import SymbolicInt, smax
from .width_schedule import WidthConnection, WidthStep, estimate_connection_width


def query_resource_logical_width(
    root: ResourceCall,
    *,
    declared_calls: DeclaredResourceCalls | None = None,
    cache: MutableMapping[ResourceCall, SymbolicInt] | None = None,
) -> SymbolicInt:
    """Query owned atoms, components, wrappers, And ladders and exact swaps.

    ComponentCost owns local peaks and per-call bystanders. Gate declaration
    overrides affect the fallback call graph, while an available width
    decomposition retains its own connections and ownership. Unsupported
    resource identities raise NotImplementedError. Cache entries retain the
    decomposition's visitation, independently of gate-category graph nodes.
    """

    def decomposition(
        node: ResourceCall, child: Callable[[ResourceCall], SymbolicInt]
    ) -> SymbolicInt | None:
        if isinstance(node, ComponentCost):
            peak: SymbolicInt = _component_width(node)
            for call in _active_calls(node):
                peak = smax(peak, child(call.component) + call.bystander_qubits)
            return peak
        if isinstance(node, ConjunctionCall):
            return multi_and_width(node, child)
        if isinstance(node, ControlledSwapCall):
            return controlled_swap_width(node, child)
        if isinstance(node, PowerCall):
            if type(node.repetitions) is not int:
                raise ValueError("Symbolic power has no source width decomposition")
            return child(node.child)
        if isinstance(node, OnEachCall):
            if _signature_width(node.child) != 1:
                raise ValueError("OnEach requires a supported single-qubit gate")
            if type(node.width) is not int:
                return None
            split = child(BookkeepingCall("split", node.width))
            if node.width == 0:
                return smax(split, child(BookkeepingCall("join", 0)))
            local = child(node.child)
            join = child(BookkeepingCall("join", node.width))
            # Each single-wire call has all other wires as live bystanders.
            # Equal independent calls can share one representative occurrence.
            return estimate_connection_width(
                [
                    WidthStep("input"),
                    WidthStep("component", split),
                    WidthStep("component", local),
                    WidthStep("component", join),
                    WidthStep("output"),
                ],
                [
                    WidthConnection(0, 1, node.width),
                    WidthConnection(1, 2, 1),
                    WidthConnection(1, 3, node.width - 1),
                    WidthConnection(2, 3, 1),
                    WidthConnection(3, 4, node.width),
                ],
            )
        return None

    def callees(node: ResourceCall) -> tuple[tuple[ResourceCall, SymbolicInt], ...]:
        if isinstance(node, (PowerCall, OnEachCall)) or (
            declared_calls is not None and node in declared_calls
        ):
            return tuple(decomposed_resource_callees(node, declared_calls))
        if isinstance(node, ConjunctionCall):
            size = (
                len(node.controls)
                if isinstance(node.controls, tuple)
                else node.controls
            )
            if size == 2:
                return ()
            return ((ConjunctionCall((1, 1), node.adjoint), size - 1),)
        if isinstance(node, ControlledSwapCall):
            return tuple(decomposed_resource_callees(node, declared_calls))
        return ()

    return query_logical_width(
        root,
        signature_width=_signature_width,
        callees=callees,
        decomposition_width=decomposition,
        cache=cache,
    )


def _signature_width(node: ResourceCall) -> SymbolicInt:
    if isinstance(node, ConjunctionCall):
        size = len(node.controls) if isinstance(node.controls, tuple) else node.controls
        if isinstance(size, int) and (type(size) is bool or size < 2):
            raise ValueError("And width requires at least two controls")
        return 2 * size - 1
    if isinstance(node, ControlledSwapCall) and not node.approximate:
        return 2 * node.width + 1
    if isinstance(node, ComponentCost):
        return _component_width(node)
    if isinstance(node, PhaseCall):
        return 1
    if isinstance(node, CliffordCall):
        if node.opcode != "arbitrary":
            expected = 2 if node.opcode in {"cx", "cy", "cz", "swap_bit"} else 1
            if node.width != expected:
                raise ValueError("Clifford width must match its gate signature")
        return node.width
    if isinstance(node, FixedTCall):
        return 3
    if isinstance(node, ZeroCostCall) and node.opcode == "identity":
        if len(node.parameters) == 1:
            from sympy import Expr

            width = node.parameters[0]
            if isinstance(width, (int, Expr)):
                return width
    if isinstance(node, BookkeepingCall) and node.opcode in {"split", "join"}:
        if type(node.layout) is int:
            return node.layout
    if isinstance(node, PowerCall):
        return _signature_width(node.child)
    if isinstance(node, OnEachCall):
        return node.width
    raise NotImplementedError(f"No native logical width binding for {node!r}")

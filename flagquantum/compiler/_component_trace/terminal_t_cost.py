# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 t_counts_from_sigma and TComplexity.rotation_cost.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/resource_counting/t_counts_from_sigma.py
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/cirq_interop/t_complexity_protocol.py

"""Compute the locked precision-sensitive model from terminal multiplicities."""

from collections.abc import Callable, Iterable, Mapping, MutableMapping
from math import pi

from .call_graph import compose_cost_generalizer
from .cost_query import _query_cost
from .resource_classification import classify_resource_gate
from .resource_decomposition import decomposed_resource_callees
from .resource_generalization import (
    ArithmeticCall,
    BookkeepingCall,
    ConjunctionCall,
    ControlledRotationCall,
    DeclaredResourceCalls,
    FanoutCall,
    FixedTCall,
    MeasurementCall,
    OnEachCall,
    PhaseCall,
    PowerCall,
    ResourceCall,
    RotationCall,
    ZeroCostCall,
    _resource_callees,
)
from .resource_model import ComponentCost
from .symbolic_math import SymbolicInt, SymbolicReal, ceil, log2


def rotation_t_cost(precision: SymbolicReal) -> SymbolicInt:
    """Return the source's per-rotation estimate before applying multiplicity."""
    return ceil(1.149 * log2(1.0 / precision) + 9.2)


def estimate_terminal_t_cost(sigma: Mapping[ResourceCall, SymbolicInt]) -> SymbolicInt:
    """Apply the source terminal-count convention to explicit resource keys.

    Exact bitwise controlled swaps use seven T gates in this model. Only
    explicit T gates contribute at T-like angles. Rotation synthesis costs
    are rounded per call before multiplication.
    """
    return sum(count * _unit_t_cost(call) for call, count in sigma.items())


def query_terminal_t_cost(
    root: ResourceCall,
    *,
    declared_calls: DeclaredResourceCalls | None = None,
    cache: MutableMapping[ResourceCall, SymbolicInt] | None = None,
) -> SymbolicInt:
    """Cache full-subtree costs using the locked terminal-count convention.

    Explicit rows replace component declarations, including exclusive gates.
    Otherwise exclusive T-bearing categories must be represented as child
    calls. In particular, aggregate rotation counts lack synthesis precision.
    Known compound keys expand through the shared resource recipes.
    Reuse a cache only with the same declarations. Visibility cutoffs belong
    to presentation and do not affect this query. Cycles raise ValueError.
    """

    def compute(
        node: ResourceCall, child: Callable[[ResourceCall], SymbolicInt]
    ) -> SymbolicInt:
        explicit = declared_calls is not None and node in declared_calls
        if isinstance(node, ComponentCost) and not explicit:
            gates = node.gates
            if any(
                (gates.t, gates.toffoli, gates.cswap, gates.and_bloq, gates.rotation)
            ):
                raise ValueError(
                    "Declare exclusive T-bearing component gates as resource calls"
                )
        callees = tuple(_terminal_resource_callees(node, declared_calls))
        for _, count in callees:
            if isinstance(count, (int, float)) and (
                type(count) is not int or count < 0
            ):
                raise ValueError("Resource counts must be nonnegative integers")
        if callees:
            return sum(count * child(call) for call, count in callees)
        return _unit_t_cost(node)

    return _query_cost(root, compute, 0, compose_cost_generalizer(None), cache, None)


def _terminal_resource_callees(
    node: ResourceCall, declared_calls: DeclaredResourceCalls | None
) -> Iterable[tuple[ResourceCall, SymbolicInt]]:
    """Expand compound keys while retaining the source terminal-T boundary."""
    if (declared_calls is not None and node in declared_calls) or isinstance(
        node, (ComponentCost, PowerCall, OnEachCall)
    ):
        yield from _resource_callees(node, declared_calls)
    elif (
        classify_resource_gate(node) is not None
        or isinstance(
            node,
            (FixedTCall, BookkeepingCall, MeasurementCall, ZeroCostCall, FanoutCall),
        )
        or (
            isinstance(node, ConjunctionCall)
            and isinstance(node.controls, tuple)
            and len(node.controls) == 2
        )
    ):
        return
    else:
        yield from decomposed_resource_callees(node, declared_calls)


def _unit_t_cost(call: ResourceCall) -> SymbolicInt:
    if isinstance(call, ArithmeticCall):
        raise NotImplementedError(
            "Expand arithmetic resource declarations before terminal T estimation"
        )
    if isinstance(call, PhaseCall):
        return int(call.opcode == "t")
    if isinstance(call, FixedTCall):
        return 4 if call.opcode == "toffoli" else 7
    if isinstance(call, ConjunctionCall):
        return (
            4
            if isinstance(call.controls, tuple)
            and len(call.controls) == 2
            and not call.adjoint
            else 0
        )
    if classify_resource_gate(call) != "rotation":
        return 0
    if isinstance(call, RotationCall):
        return rotation_t_cost(call.precision)
    if isinstance(call, ControlledRotationCall):
        inner = call.call
        if isinstance(inner, PhaseCall):
            raise ValueError(
                "The source terminal model has no precision for controlled S or T"
            )
        value = inner.parameter
        if not isinstance(value, (int, float)):
            raise NotImplementedError(
                "Symbolic controlled rotations have no source stabilizer verdict"
            )
        exponent = value / pi if inner.opcode in {"rx", "ry", "rz"} else float(value)
        # A single controlled Pauli power is stabilizer at integral exponents.
        if exponent.is_integer():
            return 0
        return rotation_t_cost(inner.precision)
    raise AssertionError("Unexpected rotation resource key")

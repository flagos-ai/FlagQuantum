# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 typed callee rules used by QECGatesCost.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/resource_counting/_bloq_counts.py

"""Shared compound resource recipes for category costs and terminal T graphs."""

from collections.abc import Iterable
from math import pi

from .resource_generalization import (
    CliffordCall,
    ConjunctionCall,
    ControlledRegisterSwapCall,
    ControlledSwapCall,
    ControlledZPowerCall,
    ControlledZRotationCall,
    ControlPredicateCall,
    DeclaredResourceCalls,
    FanoutCall,
    FixedTCall,
    MergedFredkinCall,
    OnEachCall,
    PartitionedCall,
    PhaseCall,
    PowerCall,
    PredicateControlledCall,
    ResourceCall,
    RotationCall,
    _resource_callees,
    arbitrary_clifford_cost_key,
)
from .resource_model import ComponentCost
from .symbolic_math import SymbolicInt


def decomposed_resource_callees(
    node: ResourceCall, declared_calls: DeclaredResourceCalls | None
) -> Iterable[tuple[ResourceCall, SymbolicInt]]:
    """Yield explicit overrides or the existing compound-call recipe.

    Consumers choose terminal boundaries before invoking this function.
    Explicit rows replace all automatic children. Unsupported identities
    retain the historical QEC diagnostic for existing callers.
    """
    if isinstance(node, (ComponentCost, PowerCall, OnEachCall)) or (
        declared_calls is not None and node in declared_calls
    ):
        yield from _resource_callees(node, declared_calls)
    elif isinstance(node, ConjunctionCall):
        controls = node.controls
        size = len(controls) if isinstance(controls, tuple) else controls
        yield ConjunctionCall((1, 1), adjoint=node.adjoint), size - 1
        if isinstance(controls, tuple) and all(isinstance(cv, int) for cv in controls):
            inversions = sum(cv == 0 for cv in controls)
            if inversions:
                yield CliffordCall("x", 1), 2 * inversions
    elif isinstance(node, FanoutCall):
        if not isinstance(node.width, int) or node.width < 1:
            raise TypeError("Legacy fanout requires a positive concrete width")
        yield CliffordCall("cx", 2), 2 * node.width - 1
    elif isinstance(node, ControlledSwapCall):
        if node.approximate:
            yield PhaseCall("t", adjoint=node.adjoint), 4 * node.width
            yield arbitrary_clifford_cost_key(1, adjoint=node.adjoint), 16 * node.width
            yield arbitrary_clifford_cost_key(
                2, adjoint=node.adjoint
            ), 6 * node.width - 1
        else:
            yield FixedTCall("fredkin"), node.width
    elif isinstance(node, ControlledRegisterSwapCall):
        if not isinstance(node.width, int) or node.width < 1:
            raise TypeError("Controlled register swap requires positive concrete width")
        yield MergedFredkinCall((1, 1)), node.width
    elif isinstance(node, MergedFredkinCall):
        if len(node.controls) < 2:
            raise ValueError("Merged Fredkin requires at least two controls")
        yield ConjunctionCall(node.controls), 1
        yield FixedTCall("fredkin"), 1
        yield ConjunctionCall(node.controls, adjoint=True), 1
    elif isinstance(node, ControlledZPowerCall):
        yield ConjunctionCall((1, 1)), 1
        yield RotationCall("zpow", node.exponent, node.precision), 1
        yield ConjunctionCall((1, 1), adjoint=True), 1
    elif isinstance(node, ControlledZRotationCall):
        half = (node.angle / pi) / 2
        yield RotationCall("zpow", half, node.precision / 2), 1
        yield CliffordCall("cx", 2), 2
        yield RotationCall("zpow", -half, node.precision / 2), 1
    elif isinstance(node, ControlPredicateCall):
        controls = node.controls
        size = len(controls) if isinstance(controls, tuple) else controls
        if isinstance(size, int) and size < 2:
            raise ValueError("Control predicate requires at least two control bits")
        if isinstance(controls, tuple):
            if any(type(bit) is not int or bit not in (0, 1) for bit in controls):
                raise ValueError("Control pattern requires binary activation values")
            conjunction = ConjunctionCall(controls, node.adjoint)
        else:
            conjunction = ConjunctionCall(
                (None, None) if type(controls) is int and controls == 2 else controls,
                node.adjoint,
            )
        yield conjunction, 1
    elif isinstance(node, PredicateControlledCall):
        yield node.positive, 1
        if isinstance(node.controls, tuple) and len(node.controls) == 1:
            if node.controls == (0,):
                yield CliffordCall("x", 1), 2
            elif node.controls != (1,):
                raise ValueError("Control pattern requires binary activation values")
        else:
            yield ControlPredicateCall(node.controls), 1
            yield ControlPredicateCall(node.controls, adjoint=True), 1
    elif isinstance(node, PartitionedCall):
        yield node.call, 1
    else:
        raise NotImplementedError(f"No QEC gate model or declared callees for {node!r}")

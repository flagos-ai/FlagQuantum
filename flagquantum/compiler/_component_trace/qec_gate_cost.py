# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 QECGatesCost and typed callee rules.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/resource_counting/_bloq_counts.py

"""Count QEC gate categories over typed declarations before circuit lowering."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, MutableMapping, Sequence
from dataclasses import dataclass

from .call_graph import Generalizer, compose_cost_generalizer
from .cost_query import _query_cost
from .gate_budget import GateCategories
from .resource_classification import classify_resource_gate
from .resource_generalization import (
    BookkeepingCall,
    CliffordCall,
    ConjunctionCall,
    ControlledSwapCall,
    ControlledZPowerCall,
    DeclaredResourceCalls,
    FanoutCall,
    FixedTCall,
    MeasurementCall,
    PartitionedCall,
    PhaseCall,
    ResourceCall,
    RotationCall,
    ZeroCostCall,
    _resource_callees,
)
from .resource_model import ComponentCost
from .symbolic_math import SymbolicInt


@dataclass(frozen=True)
class QECGateCost:
    """The source gate categories with concrete or symbolic multiplicities."""

    t: SymbolicInt = 0
    toffoli: SymbolicInt = 0
    cswap: SymbolicInt = 0
    and_bloq: SymbolicInt = 0
    clifford: SymbolicInt = 0
    rotation: SymbolicInt = 0
    measurement: SymbolicInt = 0

    def plus(self, other: GateCategories, repetitions: SymbolicInt) -> QECGateCost:
        return QECGateCost(
            t=self.t + repetitions * other.t,
            toffoli=self.toffoli + repetitions * other.toffoli,
            cswap=self.cswap + repetitions * other.cswap,
            and_bloq=self.and_bloq + repetitions * other.and_bloq,
            clifford=self.clifford + repetitions * other.clifford,
            rotation=self.rotation + repetitions * other.rotation,
            measurement=self.measurement + repetitions * other.measurement,
        )


def query_qec_gate_cost(
    root: ResourceCall,
    *,
    declared_calls: DeclaredResourceCalls | None = None,
    legacy_shims: bool = False,
    legacy_annotations: Mapping[ResourceCall, GateCategories] | None = None,
    cache: MutableMapping[ResourceCall, QECGateCost] | None = None,
    static_costs: Mapping[ResourceCall, QECGateCost] | None = None,
    generalizer: (
        Generalizer[ResourceCall] | Sequence[Generalizer[ResourceCall]] | None
    ) = None,
) -> QECGateCost:
    """Prefer seeded and static costs, then classify or recurse into callees.

    In legacy mode an explicit legacy annotation supplies its T, Clifford,
    and rotation fields before normal classification. Reuse a cache only
    with the same declaration rows, overrides, mode and generalizer.
    """

    def compute(
        node: ResourceCall, child: Callable[[ResourceCall], QECGateCost]
    ) -> QECGateCost:
        if (
            legacy_shims
            and legacy_annotations is not None
            and node in legacy_annotations
        ):
            old = legacy_annotations[node]
            return QECGateCost(t=old.t, clifford=old.clifford, rotation=old.rotation)
        leaf = _leaf_cost(node, legacy_shims)
        if leaf is not None:
            return leaf
        gates = (
            QECGateCost().plus(node.gates, 1)
            if isinstance(node, ComponentCost)
            else QECGateCost()
        )
        for call, count in _qec_callees(node, declared_calls):
            gates = gates.plus(child(call), count)
        return gates

    generalize: Generalizer[ResourceCall] = compose_cost_generalizer(generalizer)
    return _query_cost(root, compute, QECGateCost(), generalize, cache, static_costs)


def _leaf_cost(node: ResourceCall, legacy: bool) -> QECGateCost | None:
    family = classify_resource_gate(node)
    if family == "t":
        return QECGateCost(t=1)
    if isinstance(node, FixedTCall):
        return (
            QECGateCost(toffoli=1) if node.opcode == "toffoli" else QECGateCost(cswap=1)
        )
    if isinstance(node, MeasurementCall):
        return QECGateCost(measurement=1)
    if (
        isinstance(node, ConjunctionCall)
        and isinstance(node.controls, tuple)
        and len(node.controls) == 2
    ):
        flips = 2 * sum(cv == 0 for cv in node.controls) if legacy else 0
        if node.adjoint:
            return QECGateCost(clifford=3 + flips if legacy else 1, measurement=1)
        return QECGateCost(and_bloq=1, clifford=flips)
    if family == "clifford" or (isinstance(node, FanoutCall) and not legacy):
        return QECGateCost(clifford=1)
    if isinstance(node, (BookkeepingCall, ZeroCostCall)):
        return QECGateCost()
    if family == "rotation":
        return QECGateCost(rotation=1)
    return None


def _qec_callees(
    node: ResourceCall, declared_calls: DeclaredResourceCalls | None
) -> Iterable[tuple[ResourceCall, SymbolicInt]]:
    if isinstance(node, ComponentCost) or (
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
            yield CliffordCall("arbitrary", 1, node.adjoint), 16 * node.width
            yield CliffordCall("arbitrary", 2, node.adjoint), 6 * node.width - 1
        else:
            yield FixedTCall("fredkin"), node.width
    elif isinstance(node, ControlledZPowerCall):
        yield ConjunctionCall((1, 1)), 1
        yield RotationCall("zpow", node.exponent, node.precision), 1
        yield ConjunctionCall((1, 1), adjoint=True), 1
    elif isinstance(node, PartitionedCall):
        yield node.call, 1
    else:
        raise NotImplementedError(f"No QEC gate model or declared callees for {node!r}")

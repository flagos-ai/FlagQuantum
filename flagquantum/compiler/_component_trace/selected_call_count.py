# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 BloqCount.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/resource_counting/_bloq_counts.py

"""Count exact declared call identities at a chosen gate boundary."""

from collections.abc import Collection, Sequence

from .call_graph import Generalizer, compose_cost_generalizer
from .resource_generalization import (
    DeclaredResourceCalls,
    FixedTCall,
    PhaseCall,
    ResourceCall,
    _resource_callees,
)
from .symbolic_math import SymbolicInt

_T_GATES: tuple[ResourceCall, ...] = (PhaseCall("t"), PhaseCall("t", adjoint=True))
_T_TOFFOLI = (*_T_GATES, FixedTCall("toffoli"))
_GATESETS = {
    "t": _T_GATES,
    "t+tof": _T_TOFFOLI,
    "t+tof+cswap": (*_T_TOFFOLI, FixedTCall("fredkin")),
}


def resource_gateset(name: str) -> tuple[ResourceCall, ...]:
    """Return the locked named gate keys, keeping T and its adjoint separate."""
    if name not in _GATESETS:
        raise ValueError(f"Unknown gateset name {name}")
    return _GATESETS[name]


def count_selected_calls(
    root: ResourceCall,
    selected: Collection[ResourceCall],
    *,
    declared_calls: DeclaredResourceCalls | None = None,
    generalizer: (
        Generalizer[ResourceCall] | Sequence[Generalizer[ResourceCall]] | None
    ) = None,
) -> dict[ResourceCall, SymbolicInt]:
    """Match generalized keys by equality and stop expansion at each match.

    Explicit call rows override component declarations for traversal. Counts
    describe invocations. Exclusive gates inside a component contribute only
    when separately declared as calls.
    """
    targets = frozenset(selected)
    generalize: Generalizer[ResourceCall] = compose_cost_generalizer(generalizer)
    cache: dict[ResourceCall, dict[ResourceCall, SymbolicInt]] = {}
    active: set[ResourceCall] = set()

    def visit(original: ResourceCall) -> dict[ResourceCall, SymbolicInt]:
        node = generalize(original)
        if node is None:
            return {}
        if node in cache:
            return cache[node]
        if node in active:
            raise ValueError("Selected call count contains a cycle")
        if node in targets:
            totals: dict[ResourceCall, SymbolicInt] = {node: 1}
        else:
            totals = {}
            active.add(node)
            for child, repetitions in _resource_callees(node, declared_calls):
                for gate, count in visit(child).items():
                    totals[gate] = totals.get(gate, 0) + repetitions * count
            active.remove(node)
        cache[node] = totals
        return totals

    return visit(root)

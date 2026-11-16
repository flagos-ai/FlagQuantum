# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 for_testing/costing.py.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/for_testing/costing.py

"""Model-only fixture declarations for native resource queries."""

from collections.abc import Sequence
from dataclasses import dataclass
from graphlib import TopologicalSorter
from typing import Literal

from flagquantum.compiler._component_trace.call_graph import SymbolicCallGraph
from flagquantum.compiler._component_trace.qec_gate_cost import QECGateCost
from flagquantum.compiler._component_trace.resource_generalization import (
    CliffordCall,
    FixedTCall,
    PhaseCall,
    ResourceCall,
)
from flagquantum.compiler._component_trace.resource_model import ComponentCost
from flagquantum.compiler._component_trace.symbolic_math import SymbolicInt
from flagquantum.core.ir.program.signatures import Signature

StaticKey = Literal["qec", "width"]


@dataclass(frozen=True, kw_only=True)
class DeclaredCostFixture(ComponentCost):
    """Retain source structural identity before converting sequences to mappings."""

    callees: Sequence[tuple[ResourceCall, SymbolicInt]] = ()
    static_costs: Sequence[tuple[StaticKey, QECGateCost | int]] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "callees", tuple(self.callees))
        object.__setattr__(self, "static_costs", tuple(self.static_costs))

    @property
    def signature(self) -> Signature:
        return Signature.build(register=self.qubits)

    def declared_calls(self) -> dict[ResourceCall, SymbolicInt]:
        return dict(self.callees)

    def __str__(self) -> str:
        return self.name


def costing_example() -> tuple[DeclaredCostFixture, ...]:
    first = DeclaredCostFixture(
        "Func1",
        10,
        callees=[
            (PhaseCall("t"), 10),
            (PhaseCall("t", True), 10),
            (CliffordCall("h", 1), 10),
        ],
    )
    second = DeclaredCostFixture("Func2", 3, callees=[(FixedTCall("toffoli"), 100)])
    return (
        DeclaredCostFixture("Algo", 100, callees=[(first, 2), (second, 1)]),
        first,
        second,
    )


def fixture_label(node: ResourceCall) -> str:
    if isinstance(node, PhaseCall):
        return node.opcode.upper() + ("†" if node.adjoint else "")
    if isinstance(node, CliffordCall):
        return node.opcode.upper()
    if isinstance(node, FixedTCall):
        return node.opcode.capitalize()
    return str(node)


def debug_edges(graph: SymbolicCallGraph[ResourceCall]) -> str:
    sorter: TopologicalSorter[ResourceCall] = TopologicalSorter()
    sorter.add(graph.root)
    edges: dict[ResourceCall, dict[ResourceCall, SymbolicInt]] = {}
    for (parent, child), count in graph.direct_calls.items():
        sorter.add(child, parent)
        edges.setdefault(parent, {})[child] = count
    sorter.prepare()
    lines = []
    while sorter.is_active():
        ready = sorter.get_ready()
        for parent in sorted(ready, key=fixture_label):
            children = edges.get(parent, {})
            for child in sorted(children, key=fixture_label):
                lines.append(
                    f"{fixture_label(parent)} -- {children[child]} -> {fixture_label(child)}"
                )
        sorter.done(*ready)
    return "\n".join(lines)

# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 resource gate predicates.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/resource_counting/classify_bloqs.py

"""Classify resource call keys and attribute weighted model T costs."""

from collections.abc import Callable, Iterable, Mapping, Sequence
from math import pi
from typing import Literal

from .call_graph import (
    Generalizer,
    _build_call_graph,
    _component_calls,
    compose_cost_generalizer,
)
from .cost_query import GATE_COST, query_component_cost
from .resource_generalization import (
    CliffordCall,
    ControlledRotationCall,
    PhaseCall,
    ResourceCall,
    RotationCall,
    _inside_partition,
    ignore_allocation,
    ignore_register_partition,
)
from .resource_model import ComponentCost
from .symbolic_math import SymbolicInt

_CLIFFORD_EXPONENTS = (1.0, 0.5, 1.5, -1.0, -0.5, -1.5, 0.0, -2.0, 2.0)
_CLIFFORD_ANGLES = tuple(exponent * pi for exponent in _CLIFFORD_EXPONENTS)
_T_EXPONENTS = (0.25, -0.25)
_T_ANGLES = (pi / 4, -pi / 4)
_ANGLE_ATOL = 1e-12


def classify_resource_gate(
    call: ResourceCall,
) -> Literal["clifford", "t", "rotation"] | None:
    """Recognize a modeled gate family, including the source controlled shim."""
    if isinstance(call, CliffordCall):
        return "clifford"
    if isinstance(call, PhaseCall):
        return "clifford" if call.opcode == "s" else "t"
    if isinstance(call, ControlledRotationCall):
        return "rotation" if len(call.controls) == 1 else None
    if not isinstance(call, RotationCall):
        return None

    value = call.parameter
    if not isinstance(value, (int, float)):
        return "rotation"
    angle = call.opcode in {"rx", "ry", "rz"}
    cliffords = _CLIFFORD_ANGLES if angle else _CLIFFORD_EXPONENTS
    t_values = _T_ANGLES if angle else _T_EXPONENTS
    if any(abs(value - target) < _ANGLE_ATOL for target in cliffords):
        return "clifford"
    if any(abs(value - target) < _ANGLE_ATOL for target in t_values):
        return "t"
    return "rotation"


def ignore_known_cliffords(call: ResourceCall) -> ResourceCall | None:
    """Omit source-recognized Clifford keys, recursing through partitions."""
    if classify_resource_gate(call) == "clifford":
        return None
    return _inside_partition(ignore_known_cliffords, call)


def attribute_t_cost(
    root: ResourceCall,
    family: Callable[[ResourceCall], str | None],
    *,
    declared_calls: (
        Mapping[ResourceCall, Mapping[ResourceCall, SymbolicInt]] | None
    ) = None,
    unit_t_cost: Callable[[ResourceCall], SymbolicInt] | None = None,
    generalizer: (
        Generalizer[ResourceCall] | Sequence[Generalizer[ResourceCall]] | None
    ) = None,
) -> dict[str, SymbolicInt]:
    """Attribute weighted T equivalents to the first classified calls on each path.

    Explicit call rows override component declarations for graph traversal.
    Classified calls are terminals and contribute their full unit cost. The
    remaining leaves use the `other` family. Represent exclusive costs of
    expanded interior nodes as child calls to attribute them separately.
    A custom unit cost is required for terminal keys outside ComponentCost.
    """

    def calls(node: ResourceCall) -> Iterable[tuple[ResourceCall, SymbolicInt]]:
        if declared_calls is not None and node in declared_calls:
            yield from declared_calls[node].items()
        elif isinstance(node, ComponentCost):
            yield from _component_calls(node)

    def keep(node: ResourceCall) -> bool:
        return family(node) not in {None, "other"}

    graph = _build_call_graph(
        root,
        calls,
        (
            ignore_register_partition,
            ignore_allocation,
            ignore_known_cliffords,
            compose_cost_generalizer(generalizer),
        ),
        keep,
        None,
    )
    by_family: dict[str, SymbolicInt] = {}
    for node, repetitions in graph.leaf_counts.items():
        if unit_t_cost is not None:
            cost = unit_t_cost(node)
        elif isinstance(node, ComponentCost):
            cost = query_component_cost(node, GATE_COST).total_t_count()
        else:
            raise ValueError("A terminal resource key requires a unit T cost")
        if cost > 0:
            label = family(node) or "other"
            by_family[label] = by_family.get(label, 0) + repetitions * cost
    return by_family

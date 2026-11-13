# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Connection liveness follows locked Qualtran v0.7.0 QubitCount.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/resource_counting/_qubit_counts.py

"""Project finite through-wire calls into the existing width graph model."""

from __future__ import annotations

from collections.abc import Callable, Iterable

from ..._component_trace.resource_generalization import ResourceCall
from ..._component_trace.symbolic_math import SymbolicInt
from ..._component_trace.width_schedule import (
    WidthConnection,
    WidthStep,
    estimate_connection_width,
    schedule_width_steps,
)


def through_call_width(
    wire_count: int,
    calls: Iterable[tuple[ResourceCall, tuple[int, ...]]],
    child: Callable[[ResourceCall], SymbolicInt],
) -> SymbolicInt:
    """Query local peaks in connection order, retaining all idle bystanders.

    Each call consumes and returns exactly its listed wires. Allocation,
    freeing and non-through logical outputs belong to their separate owners.
    Register views may list multiple wires but retain one resource-call key.
    """
    producers = dict.fromkeys(range(wire_count), 0)
    steps = [WidthStep("input")]
    keys = []
    edges = []
    for key, wires in calls:
        index = len(steps)
        if not wires or len(set(wires)) != len(wires):
            raise ValueError("Through calls require distinct nonempty wire bindings")
        for wire in wires:
            if type(wire) is not int or wire not in producers:
                raise ValueError("Through call wire is outside its input boundary")
            edges.append(WidthConnection(producers[wire], index, 1))
            producers[wire] = index
        steps.append(WidthStep("component"))
        keys.append(key)
    output = len(steps)
    edges.extend(WidthConnection(source, output, 1) for source in producers.values())
    steps.append(WidthStep("output"))
    for group in schedule_width_steps(steps, edges):
        for index in group:
            if 0 < index < output:
                steps[index] = WidthStep("component", child(keys[index - 1]))
    return estimate_connection_width(steps, edges)

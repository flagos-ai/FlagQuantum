# Copyright 2023 Google LLC
# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 _cbloq_max_width and greedy_topological_sort.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/resource_counting/_qubit_counts.py
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/_infra/binst_graph_iterators.py

"""Estimate sequential logical width from declared register connections."""

from collections.abc import Sequence
from dataclasses import dataclass
from heapq import heappop, heappush
from typing import Literal, cast

from .symbolic_math import SymbolicInt, smax

_ALLOCATION_PRIORITY = 10**16


@dataclass(frozen=True)
class WidthStep:
    """One component occurrence with its modeled local peak.

    Input and output steps represent the graph's boundary ports. Allocation
    and free steps describe lifetime changes, with widths modeled separately.
    """

    kind: Literal["component", "input", "output", "allocate", "free"]
    qubits: SymbolicInt = 0


@dataclass(frozen=True)
class WidthConnection:
    """One register flowing between two steps indexed in the supplied sequence."""

    source: int
    target: int
    qubits: SymbolicInt


def estimate_connection_width(
    steps: Sequence[WidthStep], connections: Sequence[WidthConnection]
) -> SymbolicInt:
    """Use stable greedy order and take the maximum over weak components.

    Register connections account for live qubits between steps. During a
    step its local peak replaces its consumed inputs. The remaining live
    registers count as bystanders. Parallel edges represent distinct registers.
    """
    incoming: list[list[WidthConnection]] = [[] for _ in steps]
    outgoing: list[list[WidthConnection]] = [[] for _ in steps]
    for connection in connections:
        if not (
            0 <= connection.source < len(steps) and 0 <= connection.target < len(steps)
        ):
            raise ValueError("Width connection endpoint is outside the step sequence")
        _validate_width(connection.qubits)
        incoming[connection.target].append(connection)
        outgoing[connection.source].append(connection)

    priorities = []
    for index, step in enumerate(steps):
        _validate_width(step.qubits)
        if step.kind in {"input", "allocate"} and incoming[index]:
            raise ValueError("Input and allocation steps cannot consume registers")
        if step.kind in {"output", "free"} and outgoing[index]:
            raise ValueError("Output and free steps cannot produce registers")
        priorities.append(_priority(step, incoming[index], outgoing[index]))

    pending = [len(edges) for edges in incoming]
    seen: set[int] = set()
    peak: SymbolicInt = 0
    for first in range(len(steps)):
        if first in seen:
            continue
        component = [first]
        seen.add(first)
        for node in component:
            neighbors = (
                *(edge.source for edge in incoming[node]),
                *(edge.target for edge in outgoing[node]),
            )
            for other in neighbors:
                if other not in seen:
                    seen.add(other)
                    component.append(other)

        ready: list[tuple[int, int]] = []
        for node in component:
            if pending[node] == 0:
                heappush(ready, (priorities[node], node))
        live: SymbolicInt = 0
        visited = 0
        while ready:
            _, node = heappop(ready)
            visited += 1
            live -= sum(edge.qubits for edge in incoming[node])
            step = steps[node]
            if step.kind not in {"input", "output"}:
                peak = cast(SymbolicInt, smax(peak, live + step.qubits))
            live += sum(edge.qubits for edge in outgoing[node])
            peak = cast(SymbolicInt, smax(peak, live))
            for edge in outgoing[node]:
                pending[edge.target] -= 1
                if pending[edge.target] == 0:
                    heappush(ready, (priorities[edge.target], edge.target))
        if visited != len(component):
            raise ValueError("Width connection graph contains a cycle")

    return peak


def _validate_width(width: SymbolicInt) -> None:
    if isinstance(width, int) and (type(width) is bool or width < 0):
        raise ValueError("Logical widths must be nonnegative integers")


def _priority(
    step: WidthStep,
    incoming: Sequence[WidthConnection],
    outgoing: Sequence[WidthConnection],
) -> int:
    if step.kind == "allocate":
        return _ALLOCATION_PRIORITY
    if step.kind == "free":
        return -_ALLOCATION_PRIORITY
    if step.kind != "component":
        return 0
    if any(not isinstance(edge.qubits, int) for edge in (*incoming, *outgoing)):
        return 0
    return cast(
        int,
        sum(edge.qubits for edge in outgoing) - sum(edge.qubits for edge in incoming),
    )

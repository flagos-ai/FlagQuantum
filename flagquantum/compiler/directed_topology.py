"""Private directed-CX topology and explicit placement routing."""

from __future__ import annotations

import hashlib
import json
from collections import deque
from dataclasses import dataclass, replace
from typing import Any, Iterable

from ..core.ir import CircuitIR, Instruction, ensure_circuit_ir


@dataclass(frozen=True)
class DirectedCouplingMap:
    """Canonical directed physical connectivity for CX instructions."""

    n_wires: int
    edges: tuple[tuple[int, int], ...]

    def __init__(self, n_wires: int, edges: Iterable[tuple[int, int]]) -> None:
        normalized_count = int(n_wires)
        if normalized_count <= 0:
            raise ValueError("Directed coupling map requires a positive wire count.")
        normalized = tuple(sorted((int(left), int(right)) for left, right in edges))
        if len(normalized) != len(set(normalized)):
            raise ValueError("Directed coupling edges must be unique.")
        for left, right in normalized:
            if left == right:
                raise ValueError("Directed coupling map cannot contain self edges.")
            if min(left, right) < 0 or max(left, right) >= normalized_count:
                raise ValueError("Directed coupling edge is outside the device.")
        object.__setattr__(self, "n_wires", normalized_count)
        object.__setattr__(self, "edges", normalized)

    @property
    def topology_identity(self) -> str:
        payload = {
            "n_wires": self.n_wires,
            "directed_edges": self.edges,
            "direction_semantics": "directed_cx",
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def has_edge(self, control: int, target: int) -> bool:
        self._validate_wire(control)
        self._validate_wire(target)
        return (control, target) in self.edges

    def has_weak_edge(self, left: int, right: int) -> bool:
        return self.has_edge(left, right) or self.has_edge(right, left)

    def shortest_path(self, start: int, goal: int) -> tuple[int, ...]:
        self._validate_wire(start)
        self._validate_wire(goal)
        if start == goal:
            return (start,)
        adjacency: list[set[int]] = [set() for _ in range(self.n_wires)]
        for left, right in self.edges:
            adjacency[left].add(right)
            adjacency[right].add(left)
        parents = {start: -1}
        queue: deque[int] = deque((start,))
        while queue:
            wire = queue.popleft()
            for neighbor in sorted(adjacency[wire]):
                if neighbor in parents:
                    continue
                parents[neighbor] = wire
                if neighbor == goal:
                    path = [goal]
                    while path[-1] != start:
                        path.append(parents[path[-1]])
                    return tuple(reversed(path))
                queue.append(neighbor)
        raise ValueError(f"No coupling path between wires {start} and {goal}.")

    def _validate_wire(self, wire: int) -> None:
        if type(wire) is not int or not 0 <= wire < self.n_wires:
            raise ValueError(
                f"Directed coupling wire {wire!r} is outside "
                f"[0, {self.n_wires - 1}]."
            )


def _validate_layout(
    ir: CircuitIR, coupling: DirectedCouplingMap, layout: object
) -> tuple[int, ...]:
    if coupling.n_wires != ir.n_wires:
        raise ValueError(
            "Directed topology v1 requires physical-wire count to equal "
            "CircuitIR logical-wire count."
        )
    if layout is None:
        return tuple(range(ir.n_wires))
    if not isinstance(layout, tuple) or any(type(item) is not int for item in layout):
        raise TypeError("initial_layout must be a tuple of integer physical wires")
    if len(layout) != ir.n_wires or set(layout) != set(range(ir.n_wires)):
        raise ValueError("initial_layout must be a complete physical-wire permutation")
    return layout


def _mapped_instruction(
    instruction: Instruction,
    wires: tuple[int, ...],
    *,
    source_index: int,
) -> Instruction:
    return Instruction(
        instruction.name,
        wires,
        params=instruction.params,
        matrix=instruction.matrix,
        metadata=dict(instruction.metadata)
        | {
            "layout_mapped": wires != instruction.wires,
            "logical_wires": instruction.wires,
            "routing_strategy": "persistent_layout",
            "source_instruction_index": source_index,
        },
    )


def route_to_directed_topology(
    circuit_or_ir: Any,
    coupling_map: DirectedCouplingMap,
    *,
    initial_layout: tuple[int, ...] | None = None,
) -> CircuitIR:
    """Route on weak connectivity and restore the logical output layout."""

    ir = ensure_circuit_ir(circuit_or_ir)
    if not isinstance(coupling_map, DirectedCouplingMap):
        raise TypeError("coupling_map must be a DirectedCouplingMap")
    layout = _validate_layout(ir, coupling_map, initial_layout)
    logical_to_physical = list(layout)
    physical_to_logical = [0] * ir.n_wires
    for logical, physical in enumerate(layout):
        physical_to_logical[physical] = logical
    routed: list[Instruction] = []
    swap_count = 0
    routed_gate_count = 0

    def swap(
        left: int,
        right: int,
        source: Instruction,
        source_index: int,
        phase: str,
    ) -> None:
        nonlocal swap_count
        routed.append(
            Instruction(
                "swap",
                (left, right),
                metadata={
                    "routed_from": source.name,
                    "routing_strategy": "persistent_layout",
                    "routing_phase": phase,
                    "logical_wires": source.wires,
                    "source_instruction_index": source_index,
                },
            )
        )
        left_logical = physical_to_logical[left]
        right_logical = physical_to_logical[right]
        physical_to_logical[left], physical_to_logical[right] = (
            right_logical,
            left_logical,
        )
        logical_to_physical[left_logical] = right
        logical_to_physical[right_logical] = left
        swap_count += 1

    last_source: tuple[Instruction, int] | None = None
    for source_index, instruction in enumerate(ir.instructions):
        last_source = (instruction, source_index)
        mapped_wires = tuple(logical_to_physical[wire] for wire in instruction.wires)
        if len(instruction.wires) == 2 and not instruction.metadata.get("is_channel"):
            if not coupling_map.has_weak_edge(*mapped_wires):
                path = coupling_map.shortest_path(*mapped_wires)
                routed_gate_count += 1
                for path_index in range(len(path) - 2):
                    swap(
                        path[path_index],
                        path[path_index + 1],
                        instruction,
                        source_index,
                        "forward",
                    )
                mapped_wires = tuple(
                    logical_to_physical[wire] for wire in instruction.wires
                )
        routed.append(
            _mapped_instruction(
                instruction,
                mapped_wires,
                source_index=source_index,
            )
        )

    pre_restore_layout = tuple(logical_to_physical)
    if logical_to_physical != list(range(ir.n_wires)):
        if last_source is None:
            raise ValueError("Non-identity initial layout requires a nonempty circuit.")
        source, source_index = last_source
        while True:
            misplaced_logical = next(
                (
                    item
                    for item, physical in enumerate(logical_to_physical)
                    if item != physical
                ),
                None,
            )
            if misplaced_logical is None:
                break
            path = coupling_map.shortest_path(
                logical_to_physical[misplaced_logical], misplaced_logical
            )
            edges = tuple(zip(path, path[1:]))
            for left, right in (*edges, *reversed(edges[:-1])):
                swap(left, right, source, source_index, "final_restore")

    identity_layout = tuple(range(ir.n_wires))
    if tuple(logical_to_physical) != identity_layout:
        raise RuntimeError("Directed routing failed to restore the final layout.")
    metadata = dict(ir.metadata)
    metadata["routing"] = {
        "schema": "flagquantum_directed_routing_plan_v1",
        "strategy": "persistent_layout",
        "coupling_n_wires": coupling_map.n_wires,
        "coupling_edges": coupling_map.edges,
        "initial_logical_to_physical": layout,
        "pre_restore_logical_to_physical": pre_restore_layout,
        "final_logical_to_physical": identity_layout,
        "mapping_restored": True,
        "direction_semantics": "directed_cx",
        "topology_gate_count": sum(
            len(item.wires) == 2 and not item.metadata.get("is_channel")
            for item in ir.instructions
        ),
        "routed_gate_count": routed_gate_count,
        "inserted_swap_count": swap_count,
        "skipped_channel_count": sum(
            len(item.wires) == 2 and bool(item.metadata.get("is_channel"))
            for item in ir.instructions
        ),
    }
    return replace(ir, instructions=tuple(routed), metadata=metadata)


__all__ = ("DirectedCouplingMap", "route_to_directed_topology")

"""Compiler-owned coupling maps and topology-aware SWAP routing."""

from __future__ import annotations

from collections import OrderedDict, deque
from dataclasses import dataclass, field, replace
from threading import Lock
from typing import Any, Iterable

from ..core.ir import CircuitIR, Instruction, ensure_circuit_ir


@dataclass(frozen=True)
class CouplingMap:
    """Undirected hardware connectivity used by native routing passes."""

    n_wires: int
    edges: tuple[tuple[int, int], ...]
    _adjacency: tuple[tuple[int, ...], ...] = field(
        init=False,
        repr=False,
        compare=False,
        hash=False,
    )
    _path_cache: OrderedDict[tuple[int, int], tuple[int, ...]] = field(
        init=False,
        repr=False,
        compare=False,
        hash=False,
    )
    path_cache_capacity: int = field(default=4096, compare=False, hash=False)
    _path_cache_lock: Lock = field(
        init=False,
        repr=False,
        compare=False,
        hash=False,
    )
    _path_cache_hits: int = field(
        init=False,
        repr=False,
        compare=False,
        hash=False,
    )
    _path_cache_misses: int = field(
        init=False,
        repr=False,
        compare=False,
        hash=False,
    )
    _path_cache_evictions: int = field(
        init=False,
        repr=False,
        compare=False,
        hash=False,
    )

    def __init__(
        self,
        n_wires: int,
        edges: Iterable[tuple[int, int]],
        *,
        path_cache_capacity: int = 4096,
    ) -> None:
        if type(n_wires) is not int:
            raise ValueError("Coupling wire count must be an integer.")
        if n_wires <= 0:
            raise ValueError("Coupling map requires a positive wire count.")
        if type(path_cache_capacity) is not int:
            raise ValueError("Path cache capacity must be an integer.")
        if path_cache_capacity == 1 or path_cache_capacity < 0:
            raise ValueError("Path cache capacity must be zero or at least two.")
        normalized = []
        seen = set()
        for left, right in edges:
            if type(left) is not int or type(right) is not int:
                raise ValueError("Coupling edge endpoints must be integers.")
            if left < 0 or right < 0 or left >= n_wires or right >= n_wires:
                raise ValueError("Coupling edge contains a wire outside the device.")
            if left == right:
                continue
            edge = (min(left, right), max(left, right))
            if edge not in seen:
                normalized.append(edge)
                seen.add(edge)
        object.__setattr__(self, "n_wires", int(n_wires))
        object.__setattr__(self, "edges", tuple(normalized))
        adjacency: list[set[int]] = [set() for _ in range(n_wires)]
        for left, right in normalized:
            adjacency[left].add(right)
            adjacency[right].add(left)
        object.__setattr__(
            self,
            "_adjacency",
            tuple(tuple(sorted(neighbors)) for neighbors in adjacency),
        )
        object.__setattr__(self, "path_cache_capacity", path_cache_capacity)
        object.__setattr__(self, "_path_cache", OrderedDict())
        object.__setattr__(self, "_path_cache_lock", Lock())
        object.__setattr__(self, "_path_cache_hits", 0)
        object.__setattr__(self, "_path_cache_misses", 0)
        object.__setattr__(self, "_path_cache_evictions", 0)

    @classmethod
    def line(
        cls,
        n_wires: int,
        *,
        path_cache_capacity: int = 4096,
    ) -> "CouplingMap":
        if type(n_wires) is not int:
            raise ValueError("Coupling wire count must be an integer.")
        return cls(
            n_wires,
            ((wire, wire + 1) for wire in range(n_wires - 1)),
            path_cache_capacity=path_cache_capacity,
        )

    @classmethod
    def ring(
        cls,
        n_wires: int,
        *,
        path_cache_capacity: int = 4096,
    ) -> "CouplingMap":
        if type(n_wires) is not int:
            raise ValueError("Coupling wire count must be an integer.")
        edges = [(wire, wire + 1) for wire in range(n_wires - 1)]
        if n_wires > 2:
            edges.append((n_wires - 1, 0))
        return cls(
            n_wires,
            edges,
            path_cache_capacity=path_cache_capacity,
        )

    @classmethod
    def grid(
        cls,
        rows: int,
        cols: int,
        *,
        path_cache_capacity: int = 4096,
    ) -> "CouplingMap":
        if type(rows) is not int or type(cols) is not int:
            raise ValueError("Coupling grid dimensions must be integers.")
        if rows <= 0 or cols <= 0:
            raise ValueError("Coupling grid dimensions must be positive.")
        edges = []
        for row in range(rows):
            for col in range(cols):
                wire = row * cols + col
                if col + 1 < cols:
                    edges.append((wire, wire + 1))
                if row + 1 < rows:
                    edges.append((wire, wire + cols))
        return cls(
            rows * cols,
            edges,
            path_cache_capacity=path_cache_capacity,
        )

    def _validate_wire(self, wire: int) -> int:
        if type(wire) is not int:
            raise ValueError("Coupling wire index must be an integer.")
        if wire < 0 or wire >= self.n_wires:
            raise ValueError(
                f"Coupling wire {wire} is outside [0, {self.n_wires - 1}]."
            )
        return wire

    def neighbors(self, wire: int) -> tuple[int, ...]:
        wire = self._validate_wire(wire)
        return self._adjacency[wire]

    def has_edge(self, left: int, right: int) -> bool:
        left = self._validate_wire(left)
        right = self._validate_wire(right)
        return right in self._adjacency[left]

    def shortest_path(self, start: int, goal: int) -> tuple[int, ...]:
        start = self._validate_wire(start)
        goal = self._validate_wire(goal)
        cache_key = (start, goal)
        with self._path_cache_lock:
            cached = self._path_cache.get(cache_key)
            if cached is not None:
                self._path_cache.move_to_end(cache_key)
                object.__setattr__(self, "_path_cache_hits", self._path_cache_hits + 1)
                return cached
            object.__setattr__(
                self,
                "_path_cache_misses",
                self._path_cache_misses + 1,
            )
        if start == goal:
            identity_path = (start,)
            self._cache_paths((cache_key, identity_path))
            return identity_path
        parents = {start: -1}
        queue: deque[int] = deque([start])
        while queue:
            wire = queue.popleft()
            for neighbor in self.neighbors(wire):
                if neighbor in parents:
                    continue
                parents[neighbor] = wire
                if neighbor == goal:
                    path_nodes = [goal]
                    while path_nodes[-1] != start:
                        path_nodes.append(parents[path_nodes[-1]])
                    resolved = tuple(reversed(path_nodes))
                    self._cache_paths(
                        (cache_key, resolved),
                        ((goal, start), tuple(reversed(resolved))),
                    )
                    return resolved
                queue.append(neighbor)
        raise ValueError(f"No coupling path between wires {start} and {goal}.")

    def _cache_paths(
        self,
        *entries: tuple[tuple[int, int], tuple[int, ...]],
    ) -> None:
        if self.path_cache_capacity == 0:
            return
        with self._path_cache_lock:
            for key, path in entries:
                self._path_cache[key] = path
                self._path_cache.move_to_end(key)
                while len(self._path_cache) > self.path_cache_capacity:
                    self._path_cache.popitem(last=False)
                    object.__setattr__(
                        self,
                        "_path_cache_evictions",
                        self._path_cache_evictions + 1,
                    )

    def path_cache_info(self) -> dict[str, int]:
        """Return bounded-cache diagnostics for compiler observability."""

        with self._path_cache_lock:
            return {
                "capacity": self.path_cache_capacity,
                "size": len(self._path_cache),
                "hits": self._path_cache_hits,
                "misses": self._path_cache_misses,
                "evictions": self._path_cache_evictions,
            }


@dataclass(frozen=True)
class RoutingCostEstimate:
    """Lightweight routing size estimate without materializing instructions."""

    strategy: str
    source_instruction_count: int
    estimated_instruction_count: int
    topology_gate_count: int
    routed_gate_count: int
    planned_inserted_swap_count: int
    skipped_channel_count: int
    pre_restore_logical_to_physical: tuple[int, ...]
    final_logical_to_physical: tuple[int, ...]
    path_cache_delta: dict[str, int]

    def summary(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "source_instruction_count": self.source_instruction_count,
            "estimated_instruction_count": self.estimated_instruction_count,
            "topology_gate_count": self.topology_gate_count,
            "routed_gate_count": self.routed_gate_count,
            "planned_inserted_swap_count": self.planned_inserted_swap_count,
            "skipped_channel_count": self.skipped_channel_count,
            "pre_restore_logical_to_physical": (self.pre_restore_logical_to_physical),
            "final_logical_to_physical": self.final_logical_to_physical,
            "path_cache_delta": dict(self.path_cache_delta),
        }


@dataclass(frozen=True)
class RoutingStrategySelection:
    """Auditable strategy choice made before routed IR materialization."""

    selected_strategy: str
    restore_after_each_gate: RoutingCostEstimate
    persistent_layout: RoutingCostEstimate

    def summary(self) -> dict[str, Any]:
        return {
            "schema": "flagquantum_routing_strategy_selection_v1",
            "objective": "minimize_planned_inserted_swaps_then_instructions",
            "selected_strategy": self.selected_strategy,
            "tie_breaker": "restore_after_each_gate",
            "candidates": {
                "restore_after_each_gate": (self.restore_after_each_gate.summary()),
                "persistent_layout": self.persistent_layout.summary(),
            },
        }


def estimate_routing_cost(
    circuit_or_ir: Any,
    coupling_map: CouplingMap | Iterable[tuple[int, int]],
    *,
    strategy: str = "restore_after_each_gate",
) -> RoutingCostEstimate:
    """Estimate routing growth without constructing routed instructions."""

    if strategy not in {"restore_after_each_gate", "persistent_layout"}:
        raise ValueError(
            "routing strategy must be 'restore_after_each_gate' or 'persistent_layout'"
        )
    ir = ensure_circuit_ir(circuit_or_ir)
    coupling = (
        coupling_map
        if isinstance(coupling_map, CouplingMap)
        else CouplingMap(ir.n_wires, coupling_map)
    )
    if coupling.n_wires < ir.n_wires:
        raise ValueError("Coupling map has fewer wires than the circuit.")

    cache_before = coupling.path_cache_info()
    logical_to_physical = list(range(ir.n_wires))
    physical_to_logical = list(range(ir.n_wires))
    topology_gate_count = 0
    routed_gate_count = 0
    forward_swap_count = 0
    skipped_channel_count = 0
    for instruction in ir:
        if len(instruction.wires) != 2:
            continue
        if instruction.metadata.get("is_channel"):
            skipped_channel_count += 1
            continue
        topology_gate_count += 1
        mapped_wires = tuple(logical_to_physical[wire] for wire in instruction.wires)
        if coupling.has_edge(*mapped_wires):
            continue
        path = coupling.shortest_path(*mapped_wires)
        if any(wire >= ir.n_wires for wire in path):
            raise ValueError(
                "Routing through physical ancilla wires outside the circuit IR "
                "is not supported; provide an explicit layout/lowering step."
            )
        routed_gate_count += 1
        gate_swap_count = max(0, len(path) - 2)
        forward_swap_count += gate_swap_count
        if strategy == "persistent_layout":
            for path_index in range(gate_swap_count):
                left = path[path_index]
                right = path[path_index + 1]
                left_logical = physical_to_logical[left]
                right_logical = physical_to_logical[right]
                physical_to_logical[left], physical_to_logical[right] = (
                    right_logical,
                    left_logical,
                )
                logical_to_physical[left_logical] = right
                logical_to_physical[right_logical] = left

    identity_layout = tuple(range(ir.n_wires))
    pre_restore_layout = (
        tuple(logical_to_physical)
        if strategy == "persistent_layout"
        else identity_layout
    )
    planned_swaps = 2 * forward_swap_count
    cache_after = coupling.path_cache_info()
    return RoutingCostEstimate(
        strategy=strategy,
        source_instruction_count=len(ir),
        estimated_instruction_count=len(ir) + planned_swaps,
        topology_gate_count=topology_gate_count,
        routed_gate_count=routed_gate_count,
        planned_inserted_swap_count=planned_swaps,
        skipped_channel_count=skipped_channel_count,
        pre_restore_logical_to_physical=pre_restore_layout,
        final_logical_to_physical=identity_layout,
        path_cache_delta={
            key: cache_after[key] - cache_before[key]
            for key in ("hits", "misses", "evictions")
        },
    )


def select_routing_strategy(
    circuit_or_ir: Any,
    coupling_map: CouplingMap | Iterable[tuple[int, int]],
) -> RoutingStrategySelection:
    """Select a routing strategy from lightweight cost estimates."""

    ir = ensure_circuit_ir(circuit_or_ir)
    coupling = (
        coupling_map
        if isinstance(coupling_map, CouplingMap)
        else CouplingMap(ir.n_wires, coupling_map)
    )
    restore = estimate_routing_cost(
        ir,
        coupling,
        strategy="restore_after_each_gate",
    )
    persistent = estimate_routing_cost(
        ir,
        coupling,
        strategy="persistent_layout",
    )
    persistent_cost = (
        persistent.planned_inserted_swap_count,
        persistent.estimated_instruction_count,
    )
    restore_cost = (
        restore.planned_inserted_swap_count,
        restore.estimated_instruction_count,
    )
    selected = (
        "persistent_layout"
        if persistent_cost < restore_cost
        else "restore_after_each_gate"
    )
    return RoutingStrategySelection(
        selected_strategy=selected,
        restore_after_each_gate=restore,
        persistent_layout=persistent,
    )


def _swap_instruction(
    left: int,
    right: int,
    source: Instruction,
    *,
    strategy: str,
    phase: str,
    source_instruction_index: int,
) -> Instruction:
    return Instruction(
        name="swap",
        wires=(left, right),
        metadata={
            "routed_from": source.name,
            "routing_strategy": strategy,
            "routing_phase": phase,
            "logical_wires": source.wires,
            "source_instruction_index": source_instruction_index,
        },
    )


def _routing_metadata(
    *,
    ir: CircuitIR,
    coupling: CouplingMap,
    strategy: str,
    topology_gate_count: int,
    routed_gate_count: int,
    inserted_swap_count: int,
    skipped_channel_count: int,
    pre_restore_layout: tuple[int, ...],
    path_cache_before: dict[str, int],
    path_cache_after: dict[str, int],
) -> dict[str, Any]:
    identity_layout = tuple(range(ir.n_wires))
    return {
        "schema": "flagquantum_routing_plan_v1",
        "strategy": strategy,
        "coupling_n_wires": coupling.n_wires,
        "coupling_edges": coupling.edges,
        "initial_logical_to_physical": identity_layout,
        "pre_restore_logical_to_physical": pre_restore_layout,
        "final_logical_to_physical": identity_layout,
        "mapping_restored": True,
        "direction_semantics": "logical_wire_order_preserved",
        "topology_gate_count": topology_gate_count,
        "routed_gate_count": routed_gate_count,
        "inserted_swap_count": inserted_swap_count,
        "planned_inserted_swap_count": inserted_swap_count,
        "post_optimization_inserted_swap_count": None,
        "skipped_channel_count": skipped_channel_count,
        "persistent_layout_supported": True,
        "path_cache": {
            "before": path_cache_before,
            "after": path_cache_after,
            "delta": {
                key: path_cache_after[key] - path_cache_before[key]
                for key in ("hits", "misses", "evictions")
            },
        },
    }


def _remap_instruction(
    instruction: Instruction,
    wires: tuple[int, ...],
    *,
    strategy: str,
    source_instruction_index: int,
) -> Instruction:
    return Instruction(
        name=instruction.name,
        wires=wires,
        params=instruction.params,
        matrix=instruction.matrix,
        metadata=dict(instruction.metadata)
        | {
            "layout_mapped": wires != instruction.wires,
            "logical_wires": instruction.wires,
            "routing_strategy": strategy,
            "source_instruction_index": source_instruction_index,
        },
    )


def _route_persistent_layout(
    ir: CircuitIR,
    coupling: CouplingMap,
    *,
    path_cache_before: dict[str, int],
) -> CircuitIR:
    strategy = "persistent_layout"
    logical_to_physical = list(range(ir.n_wires))
    physical_to_logical = list(range(ir.n_wires))
    routed: list[Instruction] = []
    routing_swaps: list[tuple[int, int, Instruction, int]] = []
    topology_gate_count = 0
    routed_gate_count = 0
    skipped_channel_count = 0

    def apply_mapping_swap(
        left: int,
        right: int,
        source: Instruction,
        *,
        phase: str,
        source_instruction_index: int,
    ) -> None:
        routed.append(
            _swap_instruction(
                left,
                right,
                source,
                strategy=strategy,
                phase=phase,
                source_instruction_index=source_instruction_index,
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

    for source_index, instruction in enumerate(ir):
        mapped_wires = tuple(logical_to_physical[wire] for wire in instruction.wires)
        if len(instruction.wires) != 2:
            routed.append(
                _remap_instruction(
                    instruction,
                    mapped_wires,
                    strategy=strategy,
                    source_instruction_index=source_index,
                )
            )
            continue
        if instruction.metadata.get("is_channel"):
            skipped_channel_count += 1
            routed.append(
                _remap_instruction(
                    instruction,
                    mapped_wires,
                    strategy=strategy,
                    source_instruction_index=source_index,
                )
            )
            continue

        topology_gate_count += 1
        left, right = mapped_wires
        if not coupling.has_edge(left, right):
            path = coupling.shortest_path(left, right)
            if any(wire >= ir.n_wires for wire in path):
                raise ValueError(
                    "Routing through physical ancilla wires outside the circuit IR "
                    "is not supported; provide an explicit layout/lowering step."
                )
            routed_gate_count += 1
            for path_index in range(len(path) - 2):
                swap = (
                    path[path_index],
                    path[path_index + 1],
                    instruction,
                    source_index,
                )
                apply_mapping_swap(
                    swap[0],
                    swap[1],
                    swap[2],
                    phase="forward",
                    source_instruction_index=source_index,
                )
                routing_swaps.append(swap)
            mapped_wires = tuple(
                logical_to_physical[wire] for wire in instruction.wires
            )
        routed.append(
            _remap_instruction(
                instruction,
                mapped_wires,
                strategy=strategy,
                source_instruction_index=source_index,
            )
        )

    pre_restore_layout = tuple(logical_to_physical)
    for left, right, source, source_index in reversed(routing_swaps):
        apply_mapping_swap(
            left,
            right,
            source,
            phase="final_restore",
            source_instruction_index=source_index,
        )
    if logical_to_physical != list(range(ir.n_wires)):
        raise RuntimeError("persistent routing failed to restore the final layout")

    metadata = dict(ir.metadata)
    metadata["routing"] = _routing_metadata(
        ir=ir,
        coupling=coupling,
        strategy=strategy,
        topology_gate_count=topology_gate_count,
        routed_gate_count=routed_gate_count,
        inserted_swap_count=2 * len(routing_swaps),
        skipped_channel_count=skipped_channel_count,
        pre_restore_layout=pre_restore_layout,
        path_cache_before=path_cache_before,
        path_cache_after=coupling.path_cache_info(),
    )
    return replace(ir, instructions=tuple(routed), metadata=metadata)


def route_to_topology(
    circuit_or_ir: Any,
    coupling_map: CouplingMap | Iterable[tuple[int, int]],
    *,
    strategy: str = "restore_after_each_gate",
) -> CircuitIR:
    """Insert SWAP gates so two-qubit operations respect hardware topology."""

    if strategy not in {"restore_after_each_gate", "persistent_layout"}:
        raise ValueError(
            "routing strategy must be 'restore_after_each_gate' or 'persistent_layout'"
        )
    ir = ensure_circuit_ir(circuit_or_ir)
    coupling = (
        coupling_map
        if isinstance(coupling_map, CouplingMap)
        else CouplingMap(ir.n_wires, coupling_map)
    )
    if coupling.n_wires < ir.n_wires:
        raise ValueError("Coupling map has fewer wires than the circuit.")
    path_cache_before = coupling.path_cache_info()
    if strategy == "persistent_layout":
        return _route_persistent_layout(
            ir,
            coupling,
            path_cache_before=path_cache_before,
        )

    routed: list[Instruction] = []
    routed_gate_count = 0
    inserted_swap_count = 0
    topology_gate_count = 0
    skipped_channel_count = 0
    for source_index, instruction in enumerate(ir):
        if len(instruction.wires) != 2:
            routed.append(
                _remap_instruction(
                    instruction,
                    instruction.wires,
                    strategy=strategy,
                    source_instruction_index=source_index,
                )
            )
            continue
        if instruction.metadata.get("is_channel"):
            skipped_channel_count += 1
            routed.append(
                _remap_instruction(
                    instruction,
                    instruction.wires,
                    strategy=strategy,
                    source_instruction_index=source_index,
                )
            )
            continue

        topology_gate_count += 1
        left, right = instruction.wires
        if coupling.has_edge(left, right):
            routed.append(
                _remap_instruction(
                    instruction,
                    instruction.wires,
                    strategy=strategy,
                    source_instruction_index=source_index,
                )
            )
            continue

        path = coupling.shortest_path(left, right)
        if any(wire >= ir.n_wires for wire in path):
            raise ValueError(
                "Routing through physical ancilla wires outside the circuit IR "
                "is not supported; provide an explicit layout/lowering step."
            )
        forward_swaps = [
            (path[index], path[index + 1]) for index in range(len(path) - 2)
        ]
        routed_gate_count += 1
        inserted_swap_count += 2 * len(forward_swaps)
        for swap_left, swap_right in forward_swaps:
            routed.append(
                _swap_instruction(
                    swap_left,
                    swap_right,
                    instruction,
                    strategy=strategy,
                    phase="forward",
                    source_instruction_index=source_index,
                )
            )
        routed.append(
            Instruction(
                name=instruction.name,
                wires=(path[-2], path[-1]),
                params=instruction.params,
                matrix=instruction.matrix,
                metadata=dict(instruction.metadata)
                | {
                    "routed": True,
                    "logical_wires": instruction.wires,
                    "routing_strategy": strategy,
                    "source_instruction_index": source_index,
                },
            )
        )
        for swap_left, swap_right in reversed(forward_swaps):
            routed.append(
                _swap_instruction(
                    swap_left,
                    swap_right,
                    instruction,
                    strategy=strategy,
                    phase="gate_restore",
                    source_instruction_index=source_index,
                )
            )
    metadata = dict(ir.metadata)
    metadata["routing"] = _routing_metadata(
        ir=ir,
        coupling=coupling,
        strategy=strategy,
        topology_gate_count=topology_gate_count,
        routed_gate_count=routed_gate_count,
        inserted_swap_count=inserted_swap_count,
        skipped_channel_count=skipped_channel_count,
        pre_restore_layout=tuple(range(ir.n_wires)),
        path_cache_before=path_cache_before,
        path_cache_after=coupling.path_cache_info(),
    )
    return replace(ir, instructions=tuple(routed), metadata=metadata)


def record_post_routing_optimization(ir: CircuitIR) -> CircuitIR:
    """Attach retained routing-SWAP counts after compiler optimization."""

    routing = ir.metadata.get("routing")
    if not isinstance(routing, dict):
        return ir
    retained_swap_count = sum(
        instruction.name == "swap"
        and instruction.metadata.get("routing_strategy")
        in {"restore_after_each_gate", "persistent_layout"}
        for instruction in ir
    )
    metadata = dict(ir.metadata)
    metadata["routing"] = dict(routing) | {
        "post_optimization_inserted_swap_count": retained_swap_count,
        "post_optimization_instruction_count": len(ir),
    }
    return replace(ir, metadata=metadata)


__all__ = [
    "CouplingMap",
    "RoutingCostEstimate",
    "RoutingStrategySelection",
    "estimate_routing_cost",
    "record_post_routing_optimization",
    "route_to_topology",
    "select_routing_strategy",
]

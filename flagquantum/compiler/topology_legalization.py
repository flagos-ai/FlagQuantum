"""Deterministic topology legalization and routing evidence for CircuitIR."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

from ..core.ir import CircuitIR, ensure_circuit_ir
from ..core.target_capabilities import TargetCapabilitySnapshot
from ..errors import CompilationError
from .directed_topology import DirectedCouplingMap, route_to_directed_topology
from .routing import CouplingMap, route_to_topology, select_routing_strategy


class TopologyLegalizationError(CompilationError):
    """A circuit cannot be legalized for the supplied coupling topology."""


@dataclass(frozen=True)
class TopologyLegalizationResult:
    """Routed CircuitIR and immutable evidence bound to one target snapshot."""

    source_program: CircuitIR = field(repr=False)
    program: CircuitIR
    source_content_hash: str
    target_snapshot_id: str
    topology_identity: str
    strategy: str
    source_instruction_count: int
    routed_instruction_count: int
    inserted_swap_count: int
    final_logical_to_physical: tuple[int, ...]
    legalization_identity: str
    initial_logical_to_physical: tuple[int, ...] = ()
    direction_semantics: str = "undirected"

    def __post_init__(self) -> None:
        if self.source_program.content_hash != self.source_content_hash:
            raise ValueError(
                "topology source program does not match source_content_hash"
            )


def _topology_identity(coupling: CouplingMap | DirectedCouplingMap) -> str:
    if isinstance(coupling, DirectedCouplingMap):
        return coupling.topology_identity
    payload = {
        "n_wires": coupling.n_wires,
        "edges": coupling.edges,
        "direction_semantics": "undirected",
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _legalization_identity(
    source: CircuitIR,
    routed: CircuitIR,
    snapshot: TargetCapabilitySnapshot,
    topology_identity: str,
    strategy: str,
) -> str:
    payload = {
        "source_content_hash": source.content_hash,
        "routed_content_hash": routed.content_hash,
        "target_snapshot_id": snapshot.snapshot_id,
        "topology_identity": topology_identity,
        "strategy": strategy,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def legalize_circuit_topology(
    program: object,
    *,
    coupling_map: CouplingMap | DirectedCouplingMap,
    snapshot: TargetCapabilitySnapshot,
    strategy: str = "auto",
    max_added_operations: int = 256,
    initial_layout: tuple[int, ...] | None = None,
) -> TopologyLegalizationResult:
    """Route two-wire instructions and prove the resulting edge legality."""

    source = ensure_circuit_ir(program)
    if not isinstance(coupling_map, (CouplingMap, DirectedCouplingMap)):
        raise TypeError("coupling_map must be a Compiler coupling map")
    if not isinstance(snapshot, TargetCapabilitySnapshot):
        raise TypeError("snapshot must be a Core TargetCapabilitySnapshot")
    if not isinstance(max_added_operations, int) or isinstance(
        max_added_operations, bool
    ):
        raise TypeError("max_added_operations must be an integer")
    if max_added_operations < 0:
        raise ValueError("max_added_operations must be non-negative")
    if coupling_map.n_wires < source.n_wires:
        raise TopologyLegalizationError(
            "coupling map has fewer wires than the CircuitIR"
        )

    if isinstance(coupling_map, DirectedCouplingMap):
        if strategy not in {"auto", "persistent_layout"}:
            raise TopologyLegalizationError(
                "directed topology supports only 'auto' or 'persistent_layout'"
            )
        try:
            routed = route_to_directed_topology(
                source,
                coupling_map,
                initial_layout=initial_layout,
            )
        except (RuntimeError, TypeError, ValueError) as error:
            raise TopologyLegalizationError(str(error)) from error
        selected_strategy = "persistent_layout"
        deterministic_coupling: CouplingMap | DirectedCouplingMap = coupling_map
    else:
        if initial_layout is not None:
            raise TopologyLegalizationError(
                "initial_layout requires a DirectedCouplingMap"
            )
        deterministic_coupling = CouplingMap(coupling_map.n_wires, coupling_map.edges)
        try:
            if strategy == "auto":
                selected_strategy = select_routing_strategy(
                    source, deterministic_coupling
                ).selected_strategy
                deterministic_coupling = CouplingMap(
                    coupling_map.n_wires, coupling_map.edges
                )
            elif strategy in {"restore_after_each_gate", "persistent_layout"}:
                selected_strategy = strategy
            else:
                raise ValueError(
                    "strategy must be 'auto', 'restore_after_each_gate', or 'persistent_layout'"
                )
            routed = route_to_topology(
                source,
                deterministic_coupling,
                strategy=selected_strategy,
            )
        except (RuntimeError, ValueError) as error:
            raise TopologyLegalizationError(str(error)) from error

    added_operations = len(routed.instructions) - len(source.instructions)
    if added_operations > max_added_operations:
        raise TopologyLegalizationError("topology routing exceeds max_added_operations")
    illegal = tuple(
        (index, item.name, item.wires)
        for index, item in enumerate(routed.instructions)
        if len(item.wires) == 2
        and not (
            deterministic_coupling.has_weak_edge(*item.wires)
            if isinstance(deterministic_coupling, DirectedCouplingMap)
            else deterministic_coupling.has_edge(*item.wires)
        )
    )
    if illegal:
        raise TopologyLegalizationError(
            f"routed CircuitIR retains nonlocal two-wire instructions: {illegal}"
        )

    routing = routed.metadata.get("routing")
    if not isinstance(routing, dict):
        raise TopologyLegalizationError("router did not emit routing evidence")
    final_layout = tuple(routing.get("final_logical_to_physical", ()))
    identity_layout = tuple(range(source.n_wires))
    if final_layout != identity_layout or routing.get("mapping_restored") is not True:
        raise TopologyLegalizationError(
            "topology routing did not restore the logical output layout"
        )
    inserted_swap_count = routing.get("inserted_swap_count")
    if not isinstance(inserted_swap_count, int) or inserted_swap_count < 0:
        raise TopologyLegalizationError("router emitted an invalid SWAP count")
    initial = tuple(routing.get("initial_logical_to_physical", ()))
    if len(initial) != source.n_wires or set(initial) != set(identity_layout):
        raise TopologyLegalizationError("router emitted an invalid initial layout")
    direction_semantics = str(routing.get("direction_semantics", ""))
    expected_direction = (
        "directed_cx"
        if isinstance(deterministic_coupling, DirectedCouplingMap)
        else "logical_wire_order_preserved"
    )
    if direction_semantics != expected_direction:
        raise TopologyLegalizationError("router emitted invalid direction semantics")

    topology_identity = _topology_identity(deterministic_coupling)
    return TopologyLegalizationResult(
        source_program=source,
        program=routed,
        source_content_hash=source.content_hash,
        target_snapshot_id=snapshot.snapshot_id,
        topology_identity=topology_identity,
        strategy=selected_strategy,
        source_instruction_count=len(source.instructions),
        routed_instruction_count=len(routed.instructions),
        inserted_swap_count=inserted_swap_count,
        final_logical_to_physical=final_layout,
        legalization_identity=_legalization_identity(
            source,
            routed,
            snapshot,
            topology_identity,
            selected_strategy,
        ),
        initial_logical_to_physical=initial,
        direction_semantics=(
            "directed_cx"
            if isinstance(deterministic_coupling, DirectedCouplingMap)
            else "undirected"
        ),
    )


__all__ = [
    "TopologyLegalizationError",
    "TopologyLegalizationResult",
    "legalize_circuit_topology",
]

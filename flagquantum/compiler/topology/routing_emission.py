"""Emit routed CircuitIR instructions, restore layouts and record routing evidence."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from ...core.ir import CircuitIR, Instruction
from ..lowering.operand_semantics import _require_multi_qubit_device_local
from ..optimization.ordering_barrier import is_ordering_barrier
from .routing_cost import (
    _source_routing_counts,
    _TopologyCostModel,
    emitted_link_faults,
)
from .routing_strategy import _plan_sabre_for_topology
from .sabre import SabrePlan, plan_restore_swaps, plan_sabre_layout


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
    topology: _TopologyCostModel,
    strategy: str,
    topology_gate_count: int,
    routed_gate_count: int,
    inserted_swap_count: int,
    direction_swap_count: int,
    skipped_channel_count: int,
    expected_link_faults: float | None,
    initial_layout: tuple[int, ...],
    pre_restore_layout: tuple[int, ...],
    path_cache_before: dict[str, int],
    path_cache_after: dict[str, int],
) -> dict[str, Any]:
    identity_layout = tuple(range(ir.n_wires))
    return {
        "schema": "flagquantum_routing_plan_v1",
        "strategy": strategy,
        "coupling_n_wires": topology.n_qubits,
        "coupling_edges": topology.edges,
        "initial_logical_to_physical": initial_layout,
        "pre_restore_logical_to_physical": pre_restore_layout,
        "final_logical_to_physical": identity_layout,
        "mapping_restored": True,
        "direction_semantics": topology.direction_semantics,
        "direction_swap_count": direction_swap_count,
        "expected_link_faults": expected_link_faults,
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
    qubits: tuple[int, ...],
    *,
    strategy: str,
    source_instruction_index: int,
) -> Instruction:
    return Instruction(
        name=instruction.name,
        wires=qubits,
        params=instruction.params,
        matrix=instruction.matrix,
        metadata=dict(instruction.metadata)
        | {
            "layout_mapped": qubits != instruction.wires,
            "logical_wires": instruction.wires,
            "routing_strategy": strategy,
            "source_instruction_index": source_instruction_index,
        },
    )


def _route_persistent_layout(
    ir: CircuitIR,
    topology: _TopologyCostModel,
    *,
    path_cache_before: dict[str, int],
) -> CircuitIR:
    strategy = "persistent_layout"
    logical_to_physical = list(range(ir.n_wires))
    physical_to_logical = list(range(ir.n_wires))
    routed: list[Instruction] = []
    routing_swaps: list[tuple[int, int, Instruction, int]] = []
    direction_swap_count = 0
    # Counted against the source qubits, like every other strategy and like the
    # cost estimate: a gate this layout happened to make local is still a gate
    # the device did not carry, and reporting it as unrouted would make the same
    # field mean something different per strategy.
    topology_gate_count, routed_gate_count, skipped_channel_count = (
        _source_routing_counts(ir, topology)
    )

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
        if is_ordering_barrier(instruction):
            routed.append(
                _remap_instruction(
                    instruction,
                    mapped_wires,
                    strategy=strategy,
                    source_instruction_index=source_index,
                )
            )
            continue
        if len(instruction.wires) != 2:
            _require_multi_qubit_device_local(
                instruction, mapped_wires, topology.weak_edge
            )
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
            routed.append(
                _remap_instruction(
                    instruction,
                    mapped_wires,
                    strategy=strategy,
                    source_instruction_index=source_index,
                )
            )
            continue

        left, right = mapped_wires
        if not topology.executes(instruction, mapped_wires):
            path = topology.path(mapped_wires)
            if any(wire >= ir.n_wires for wire in path):
                raise ValueError(
                    "Routing through physical ancilla qubits outside the circuit IR "
                    "is not supported; provide an explicit layout/lowering step."
                )
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
        # A link whose declared direction is the opposite of these operands needs
        # one more hop. It is kept in the mapping like every other hop this
        # strategy takes, so the same final restore returns it and the estimate's
        # doubled hop count describes the program that is emitted.
        if topology.needs_orientation_swap(instruction, mapped_wires):
            apply_mapping_swap(
                mapped_wires[0],
                mapped_wires[1],
                instruction,
                phase="orientation",
                source_instruction_index=source_index,
            )
            routing_swaps.append(
                (mapped_wires[0], mapped_wires[1], instruction, source_index)
            )
            mapped_wires = tuple(
                logical_to_physical[wire] for wire in instruction.wires
            )
            direction_swap_count += 1
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
        topology=topology,
        strategy=strategy,
        topology_gate_count=topology_gate_count,
        routed_gate_count=routed_gate_count,
        inserted_swap_count=2 * len(routing_swaps),
        direction_swap_count=direction_swap_count,
        skipped_channel_count=skipped_channel_count,
        expected_link_faults=emitted_link_faults(routed, topology),
        initial_layout=tuple(range(ir.n_wires)),
        pre_restore_layout=pre_restore_layout,
        path_cache_before=path_cache_before,
        path_cache_after=topology.path_cache_info(),
    )
    return replace(ir, instructions=tuple(routed), metadata=metadata)


def _route_sabre(
    ir: CircuitIR,
    topology: _TopologyCostModel,
    *,
    strategy: str,
    path_cache_before: dict[str, int],
    plan: SabrePlan | None = None,
) -> CircuitIR:
    """Materialize one SABRE persistent layout, then restore the output layout.

    Instructions are emitted in the order the planner placed them, not in source
    order. The planner introduces a SWAP when a two-qubit operation becomes
    reachable, and every operation keeps its planned physical qubits, so a source
    order that disagrees with the plan would replay a different layout than the
    one the plan was costed against.

    The ``sabre`` strategy starts from the identity layout, so replaying its
    forward SWAPs in reverse restores the output layout. ``sabre_layout`` first
    searches for a shorter initial layout; replaying its forward SWAPs in reverse
    would only return that initial layout, so it routes every logical qubit home
    with an explicit restore instead.

    On an ordered device a placed gate whose operands run against its link is
    wrapped in a SWAP sandwich. The sandwich is layout-neutral, which is what the
    planner's own placements require: they were costed against one layout, so a
    repair that moved a qubit would invalidate every placement after it. The
    planner itself is handed the weak projection, because a SWAP may cross a link
    in either direction.
    """

    coupling = topology.weak_coupling
    if plan is None:
        initial_layout = (
            plan_sabre_layout(ir, coupling)
            if strategy == "sabre_layout"
            else tuple(range(ir.n_wires))
        )
        plan = _plan_sabre_for_topology(ir, topology, initial_layout)
    initial_layout = plan.initial_logical_to_physical
    routed: list[Instruction] = []
    # Counted against the source qubits rather than the placed ones, exactly as the
    # shortest-path strategies and the cost estimate count them, so the number
    # describes the work the topology forces rather than the work this layout
    # happened to arrange for free.
    topology_gate_count, routed_gate_count, skipped_channel_count = (
        _source_routing_counts(ir, topology)
    )
    direction_swap_count = 0
    swap_index = 0
    placed_so_far = 0
    for source_index in plan.sequence:
        while (
            swap_index < len(plan.swaps)
            and plan.swap_placements[swap_index] == placed_so_far
        ):
            anchor_index = plan.swap_anchors[swap_index]
            swap_left, swap_right = plan.swaps[swap_index]
            routed.append(
                _swap_instruction(
                    swap_left,
                    swap_right,
                    ir.instructions[anchor_index],
                    strategy=strategy,
                    phase="forward",
                    source_instruction_index=anchor_index,
                )
            )
            swap_index += 1
        placed_so_far += 1
        instruction = ir.instructions[source_index]
        placed_wires = plan.placements[source_index]
        if is_ordering_barrier(instruction):
            # The planner never blocks the front layer on the directive, so no
            # planned SWAP exists to place it and none was inserted for it.
            routed.append(
                _remap_instruction(
                    instruction,
                    placed_wires,
                    strategy=strategy,
                    source_instruction_index=source_index,
                )
            )
            continue
        if len(instruction.wires) != 2 or instruction.metadata.get("is_channel"):
            _require_multi_qubit_device_local(
                instruction, placed_wires, topology.weak_edge
            )
            routed.append(
                _remap_instruction(
                    instruction,
                    placed_wires,
                    strategy=strategy,
                    source_instruction_index=source_index,
                )
            )
            continue
        gate_wires = placed_wires
        orientation = topology.needs_orientation_swap(instruction, placed_wires)
        if orientation:
            routed.append(
                _swap_instruction(
                    placed_wires[0],
                    placed_wires[1],
                    instruction,
                    strategy=strategy,
                    phase="orientation",
                    source_instruction_index=source_index,
                )
            )
            # The SWAP exchanges the operands' states, so the gate executes
            # between the two halves of the sandwich with its operands swapped;
            # the closing SWAP puts both qubits back before the next planned
            # instruction, which is what keeps the planner's remaining placements
            # valid. Emitting both halves together would apply the gate on the
            # restored qubits and silently undo the repair.
            gate_wires = (placed_wires[1], placed_wires[0])
            direction_swap_count += 1
        routed.append(
            _remap_instruction(
                instruction,
                gate_wires,
                strategy=strategy,
                source_instruction_index=source_index,
            )
        )
        if orientation:
            routed.append(
                _swap_instruction(
                    placed_wires[0],
                    placed_wires[1],
                    instruction,
                    strategy=strategy,
                    phase="orientation_restore",
                    source_instruction_index=source_index,
                )
            )
    if swap_index != len(plan.swaps):
        raise RuntimeError("SABRE routing plan did not place every inserted SWAP")
    restore = plan_restore_swaps(plan, coupling)
    for swap_left, swap_right, anchor_index in restore:
        routed.append(
            _swap_instruction(
                swap_left,
                swap_right,
                ir.instructions[anchor_index],
                strategy=strategy,
                phase="final_restore",
                source_instruction_index=anchor_index,
            )
        )
    metadata = dict(ir.metadata)
    metadata["routing"] = _routing_metadata(
        ir=ir,
        topology=topology,
        strategy=strategy,
        topology_gate_count=topology_gate_count,
        routed_gate_count=routed_gate_count,
        inserted_swap_count=len(plan.swaps) + len(restore) + 2 * direction_swap_count,
        direction_swap_count=direction_swap_count,
        skipped_channel_count=skipped_channel_count,
        expected_link_faults=emitted_link_faults(routed, topology),
        initial_layout=initial_layout,
        pre_restore_layout=plan.final_logical_to_physical,
        path_cache_before=path_cache_before,
        path_cache_after=topology.path_cache_info(),
    )
    return replace(ir, instructions=tuple(routed), metadata=metadata)


def _route_restore_after_each_gate(
    ir: CircuitIR,
    topology: _TopologyCostModel,
    *,
    strategy: str,
    path_cache_before: dict[str, int],
) -> CircuitIR:
    routed: list[Instruction] = []
    topology_gate_count, routed_gate_count, skipped_channel_count = (
        _source_routing_counts(ir, topology)
    )
    inserted_swap_count = 0
    direction_swap_count = 0
    for source_index, instruction in enumerate(ir):
        if is_ordering_barrier(instruction):
            # Nothing executes the directive, so no coupling has to carry it and
            # no SWAP can make it more executable. It still has to come out the
            # other side, on the physical qubits this strategy leaves it on.
            routed.append(
                _remap_instruction(
                    instruction,
                    instruction.wires,
                    strategy=strategy,
                    source_instruction_index=source_index,
                )
            )
            continue
        if len(instruction.wires) != 2:
            _require_multi_qubit_device_local(
                instruction, instruction.wires, topology.weak_edge
            )
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
            routed.append(
                _remap_instruction(
                    instruction,
                    instruction.wires,
                    strategy=strategy,
                    source_instruction_index=source_index,
                )
            )
            continue

        if topology.executes(instruction, instruction.wires):
            routed.append(
                _remap_instruction(
                    instruction,
                    instruction.wires,
                    strategy=strategy,
                    source_instruction_index=source_index,
                )
            )
            continue

        path = topology.path(instruction.wires)
        if any(wire >= ir.n_wires for wire in path):
            raise ValueError(
                "Routing through physical ancilla qubits outside the circuit IR "
                "is not supported; provide an explicit layout/lowering step."
            )
        forward_swaps = [
            (path[index], path[index + 1]) for index in range(len(path) - 2)
        ]
        placement = (path[-2], path[-1])
        orientation = topology.needs_orientation_swap(instruction, placement)
        inserted_swap_count += 2 * (len(forward_swaps) + int(orientation))
        direction_swap_count += int(orientation)
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
        if orientation:
            routed.append(
                _swap_instruction(
                    placement[0],
                    placement[1],
                    instruction,
                    strategy=strategy,
                    phase="orientation",
                    source_instruction_index=source_index,
                )
            )
            # The SWAP exchanges the two operands' states, so the gate now runs
            # with its control on ``placement[1]`` and its target on
            # ``placement[0]`` -- which is the direction the device declares,
            # since that is exactly what asked for the SWAP.
            placement = (placement[1], placement[0])
        routed.append(
            Instruction(
                name=instruction.name,
                wires=placement,
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
        if orientation:
            routed.append(
                _swap_instruction(
                    placement[0],
                    placement[1],
                    instruction,
                    strategy=strategy,
                    phase="orientation_restore",
                    source_instruction_index=source_index,
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
        topology=topology,
        strategy=strategy,
        topology_gate_count=topology_gate_count,
        routed_gate_count=routed_gate_count,
        inserted_swap_count=inserted_swap_count,
        direction_swap_count=direction_swap_count,
        skipped_channel_count=skipped_channel_count,
        expected_link_faults=emitted_link_faults(routed, topology),
        initial_layout=tuple(range(ir.n_wires)),
        pre_restore_layout=tuple(range(ir.n_wires)),
        path_cache_before=path_cache_before,
        path_cache_after=topology.path_cache_info(),
    )
    return replace(ir, instructions=tuple(routed), metadata=metadata)


def record_post_routing_optimization(ir: CircuitIR) -> CircuitIR:
    """Attach retained routing-SWAP counts after compiler optimization."""

    routing = ir.metadata.get("routing")
    if not isinstance(routing, dict):
        return ir
    # ``routing_phase`` is the marker every strategy stamps on an inserted SWAP,
    # so the count does not depend on a list of strategy names.
    retained_swap_count = sum(
        instruction.name == "swap"
        and instruction.metadata.get("routing_phase") is not None
        for instruction in ir
    )
    metadata = dict(ir.metadata)
    metadata["routing"] = dict(routing) | {
        "post_optimization_inserted_swap_count": retained_swap_count,
        "post_optimization_instruction_count": len(ir),
    }
    return replace(ir, metadata=metadata)

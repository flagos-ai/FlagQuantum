"""Distributed statevector topology and memory planning."""

from __future__ import annotations

from math import log2
from typing import Any, Sequence

from ....compiler import schedule_layers
from ....core.ir import CircuitIR, Instruction, ensure_circuit_ir
from .models import (
    DistributedStatevectorPlan,
    StatevectorBufferPlan,
    StatevectorCommunicationEdge,
    StatevectorExecutionSegment,
    StatevectorFusionBlock,
    StatevectorGatePlan,
    StatevectorPerformanceEstimate,
    StatevectorRankTopology,
    StatevectorShard,
    StatevectorTraceEvent,
    StatevectorTraceReport,
)

_DIAGONAL_GATES = {
    "z",
    "s",
    "sdg",
    "t",
    "tdg",
    "rz",
    "phase",
    "p",
    "u1",
    "cz",
    "cphase",
}
_TARGET_LAST_GATES = {"cx", "cnot", "cy", "crx", "cry", "crz"}


def _is_power_of_two(value: int) -> bool:
    return value > 0 and value & (value - 1) == 0


def _rank_bits(world_size: int) -> int:
    return int(log2(world_size)) if _is_power_of_two(world_size) else 0


def _balanced_shards(
    *,
    total_amplitudes: int,
    world_size: int,
    bsz: int,
    complex_bytes: int,
) -> tuple[StatevectorShard, ...]:
    base = total_amplitudes // world_size
    remainder = total_amplitudes % world_size
    start = 0
    shards: list[StatevectorShard] = []
    for rank in range(world_size):
        count = base + (1 if rank < remainder else 0)
        end = start + count
        shards.append(
            StatevectorShard(
                rank=rank,
                world_size=world_size,
                amplitude_start=start,
                amplitude_end=end,
                local_amplitudes=count,
                local_state_bytes=count * int(bsz) * int(complex_bytes),
            )
        )
        start = end
    return tuple(shards)


def _gate_communication(
    instruction: Instruction,
    *,
    world_size: int,
    power_two_world: bool,
    sharded_wires: set[int],
    per_rank_state_bytes: int,
) -> tuple[str, str, tuple[int, ...], int]:
    name = instruction.name.lower()
    wires = tuple(int(wire) for wire in instruction.wires)
    touched = tuple(wire for wire in wires if wire in sharded_wires)
    if world_size <= 1:
        return "local", "local", (), 0
    if not power_two_world:
        return "distributed", "indexed_all_to_all", touched, per_rank_state_bytes * 2
    if not touched or name in _DIAGONAL_GATES:
        return "local", "local", touched, 0
    if len(wires) == 1:
        return "distributed", "pair_exchange", touched, per_rank_state_bytes * 2
    if name in _TARGET_LAST_GATES and wires[-1] not in sharded_wires:
        return "local", "local", touched, 0
    return "distributed", "all_to_all", touched, per_rank_state_bytes * 2


def _instruction_layers(ir: CircuitIR) -> dict[int, int]:
    layers: dict[int, int] = {}
    instruction_ids = {
        id(instruction): index for index, instruction in enumerate(ir.instructions)
    }
    for layer_index, layer in enumerate(schedule_layers(ir)):
        for instruction in layer:
            layers[instruction_ids[id(instruction)]] = layer_index
    return layers


def _layer_gate_indices(
    gate_plans: Sequence[StatevectorGatePlan],
) -> tuple[tuple[int, ...], ...]:
    if not gate_plans:
        return ()
    layer_count = max(plan.layer for plan in gate_plans) + 1
    layers: list[list[int]] = [[] for _ in range(layer_count)]
    for plan in gate_plans:
        layers[plan.layer].append(plan.index)
    return tuple(tuple(layer) for layer in layers)


def _fusion_blocks(
    gate_plans: Sequence[StatevectorGatePlan],
    *,
    max_fusion_gate_width: int,
) -> tuple[StatevectorFusionBlock, ...]:
    blocks: list[StatevectorFusionBlock] = []
    pending: list[StatevectorGatePlan] = []
    pending_wires: set[int] = set()

    def flush() -> None:
        nonlocal pending, pending_wires
        if not pending:
            return
        blocks.append(
            StatevectorFusionBlock(
                index=len(blocks),
                gate_indices=tuple(plan.index for plan in pending),
                wires=tuple(sorted(pending_wires)),
                communication_barrier=False,
                estimated_gate_width=len(pending_wires),
            )
        )
        pending = []
        pending_wires = set()

    for plan in gate_plans:
        gate_wires = set(plan.wires)
        if plan.requires_communication:
            flush()
            blocks.append(
                StatevectorFusionBlock(
                    index=len(blocks),
                    gate_indices=(plan.index,),
                    wires=plan.wires,
                    communication_barrier=True,
                    estimated_gate_width=len(plan.wires),
                )
            )
            continue
        if pending and len(pending_wires | gate_wires) > max_fusion_gate_width:
            flush()
        pending.append(plan)
        pending_wires.update(gate_wires)
    flush()
    return tuple(blocks)


def _execution_segments(
    fusion_blocks: Sequence[StatevectorFusionBlock],
    gate_plans: Sequence[StatevectorGatePlan],
) -> tuple[StatevectorExecutionSegment, ...]:
    gate_by_index = {plan.index: plan for plan in gate_plans}
    segments: list[StatevectorExecutionSegment] = []
    pending_comm: list[StatevectorFusionBlock] = []

    def flush_communication() -> None:
        nonlocal pending_comm
        if not pending_comm:
            return
        plans = [gate_by_index[block.gate_indices[0]] for block in pending_comm]
        wires = sorted({wire for plan in plans for wire in plan.wires})
        communication = plans[0].communication
        segments.append(
            StatevectorExecutionSegment(
                index=len(segments),
                kind="communication_batch",
                gate_indices=tuple(plan.index for plan in plans),
                communication=communication,
                wires=tuple(wires),
                estimated_transfer_bytes=sum(
                    plan.estimated_transfer_bytes for plan in plans
                ),
                can_overlap_with_compute=communication
                in {"pair_exchange", "all_to_all"},
            )
        )
        pending_comm = []

    for block in fusion_blocks:
        if block.communication_barrier:
            plan = gate_by_index[block.gate_indices[0]]
            batch_key = (
                plan.communication,
                (
                    plan.sharded_wires_touched
                    if plan.communication == "pair_exchange"
                    else ()
                ),
            )
            if pending_comm:
                previous = gate_by_index[pending_comm[-1].gate_indices[0]]
                previous_key = (
                    previous.communication,
                    (
                        previous.sharded_wires_touched
                        if previous.communication == "pair_exchange"
                        else ()
                    ),
                )
                if previous_key != batch_key:
                    flush_communication()
            if (
                pending_comm
                and gate_by_index[pending_comm[-1].gate_indices[0]].communication
                != plan.communication
            ):
                flush_communication()
            pending_comm.append(block)
            continue
        flush_communication()
        segments.append(
            StatevectorExecutionSegment(
                index=len(segments),
                kind="local_fusion",
                gate_indices=block.gate_indices,
                communication="local",
                wires=block.wires,
                estimated_transfer_bytes=0,
            )
        )
    flush_communication()
    return tuple(segments)


def _rank_coordinates(
    world_size: int, rank_address_bits: int
) -> tuple[tuple[int, ...], ...]:
    if rank_address_bits <= 0:
        return tuple((rank,) for rank in range(world_size))
    return tuple(
        tuple((rank >> shift) & 1 for shift in range(rank_address_bits - 1, -1, -1))
        for rank in range(world_size)
    )


def _node_count(world_size: int, local_world_size: int) -> int:
    return max(
        1,
        (max(1, int(world_size)) + max(1, int(local_world_size)) - 1)
        // max(1, int(local_world_size)),
    )


def _communication_tier(src_rank: int, dst_rank: int, *, local_world_size: int) -> str:
    local_world_size = max(1, int(local_world_size))
    return (
        "intra_node"
        if int(src_rank) // local_world_size == int(dst_rank) // local_world_size
        else "inter_node"
    )


def _segment_edge_pairs(
    segment: StatevectorExecutionSegment,
    *,
    world_size: int,
    sharded_wires: Sequence[int] = (),
) -> tuple[tuple[int, int], ...]:
    if world_size <= 1 or segment.communication == "local":
        return ()
    if segment.communication == "pair_exchange":
        touched = [wire for wire in segment.wires if wire in set(sharded_wires)]
        if touched and sharded_wires:
            bit_index = tuple(sharded_wires).index(touched[0])
            mask = 1 << (len(tuple(sharded_wires)) - bit_index - 1)
        else:
            mask = 1
        return tuple(
            (rank, rank ^ mask)
            for rank in range(world_size)
            if (rank ^ mask) < world_size and rank < (rank ^ mask)
        )
    return tuple(
        (src, dst) for src in range(world_size) for dst in range(src + 1, world_size)
    )


def _rank_topology(
    *,
    world_size: int,
    local_world_size: int,
    rank_address_bits: int,
    sharded_wires: Sequence[int],
    execution_segments: Sequence[StatevectorExecutionSegment],
    distribution: str,
) -> StatevectorRankTopology:
    edge_data: dict[tuple[int, int, str], list[int | tuple[int, int]]] = {}
    for segment in execution_segments:
        for src, dst in _segment_edge_pairs(
            segment,
            world_size=world_size,
            sharded_wires=sharded_wires,
        ):
            key = (src, dst, segment.communication)
            if key not in edge_data:
                edge_data[key] = [0, ()]
            edge_data[key][0] = (
                int(edge_data[key][0]) + segment.estimated_transfer_bytes
            )
            edge_data[key][1] = tuple(edge_data[key][1]) + (segment.index,)
    edges = tuple(
        StatevectorCommunicationEdge(
            src_rank=src,
            dst_rank=dst,
            communication=communication,
            segment_indices=tuple(segment_indices),
            estimated_transfer_bytes=int(estimated_bytes),
            tier=_communication_tier(src, dst, local_world_size=local_world_size),
        )
        for (src, dst, communication), (estimated_bytes, segment_indices) in sorted(
            edge_data.items()
        )
    )
    layout = (
        "hypercube" if distribution == "qubit_address_sharded" else "range_partition"
    )
    if world_size == 1:
        layout = "single_rank"
    return StatevectorRankTopology(
        world_size=world_size,
        local_world_size=max(1, int(local_world_size)),
        node_count=_node_count(world_size, local_world_size),
        layout=layout,
        rank_coordinates=_rank_coordinates(world_size, rank_address_bits),
        edges=edges,
    )


def _buffer_plans(
    execution_segments: Sequence[StatevectorExecutionSegment],
    *,
    world_size: int,
) -> tuple[StatevectorBufferPlan, ...]:
    plans: list[StatevectorBufferPlan] = []
    peer_divisor = max(1, world_size - 1)
    for segment in execution_segments:
        if segment.communication == "local":
            continue
        if segment.communication == "pair_exchange":
            send_bytes = segment.estimated_transfer_bytes // 4
            recv_bytes = segment.estimated_transfer_bytes // 4
        elif segment.communication == "all_to_all":
            send_bytes = segment.estimated_transfer_bytes // 2
            recv_bytes = segment.estimated_transfer_bytes // 2
        else:
            send_bytes = segment.estimated_transfer_bytes // peer_divisor
            recv_bytes = segment.estimated_transfer_bytes // peer_divisor
        plans.append(
            StatevectorBufferPlan(
                segment_index=segment.index,
                communication=segment.communication,
                send_buffer_bytes=max(0, int(send_bytes)),
                recv_buffer_bytes=max(0, int(recv_bytes)),
                scratch_buffer_bytes=max(0, int(send_bytes // 2)),
            )
        )
    return tuple(plans)


def plan_distributed_statevector(
    circuit_or_ir: Any,
    *,
    bsz: int = 1,
    world_size: int = 1,
    local_world_size: int | None = None,
    complex_bytes: int = 8,
    max_fusion_gate_width: int = 4,
) -> DistributedStatevectorPlan:
    """Build a communication-aware dense-state distributed plan."""

    ir = ensure_circuit_ir(circuit_or_ir)
    world_size = max(1, int(world_size))
    local_world_size = max(
        1,
        min(
            world_size,
            int(local_world_size if local_world_size is not None else world_size),
        ),
    )
    node_count = _node_count(world_size, local_world_size)
    bsz = max(1, int(bsz))
    complex_bytes = int(complex_bytes)
    total_amplitudes = 2 ** int(ir.n_wires)
    total_state_bytes = total_amplitudes * bsz * complex_bytes
    power_two_world = _is_power_of_two(world_size)
    rank_address_bits = _rank_bits(world_size)
    sharded_wires = (
        tuple(range(max(0, ir.n_wires - rank_address_bits), ir.n_wires))
        if power_two_world
        else ()
    )
    shards = _balanced_shards(
        total_amplitudes=total_amplitudes,
        world_size=world_size,
        bsz=bsz,
        complex_bytes=complex_bytes,
    )
    per_rank_state_bytes = max((shard.local_state_bytes for shard in shards), default=0)
    layer_by_index = _instruction_layers(ir)
    gate_plans: list[StatevectorGatePlan] = []
    for index, instruction in enumerate(ir.instructions):
        execution, communication, touched, transfer_bytes = _gate_communication(
            instruction,
            world_size=world_size,
            power_two_world=power_two_world,
            sharded_wires=set(sharded_wires),
            per_rank_state_bytes=per_rank_state_bytes,
        )
        gate_plans.append(
            StatevectorGatePlan(
                index=index,
                name=instruction.name,
                wires=tuple(int(wire) for wire in instruction.wires),
                layer=layer_by_index.get(index, index),
                execution=execution,
                communication=communication,
                sharded_wires_touched=touched,
                estimated_transfer_bytes=transfer_bytes,
            )
        )
    fusion_blocks = _fusion_blocks(
        gate_plans,
        max_fusion_gate_width=max(1, int(max_fusion_gate_width)),
    )
    execution_segments = _execution_segments(fusion_blocks, gate_plans)
    communication_gate_count = sum(
        1 for plan in gate_plans if plan.requires_communication
    )
    distribution = (
        "qubit_address_sharded" if power_two_world else "contiguous_amplitude_range"
    )
    if world_size == 1:
        distribution = "replicated_single_rank"
    topology = _rank_topology(
        world_size=world_size,
        local_world_size=local_world_size,
        rank_address_bits=rank_address_bits,
        sharded_wires=sharded_wires,
        execution_segments=execution_segments,
        distribution=distribution,
    )
    buffer_plans = _buffer_plans(execution_segments, world_size=world_size)
    return DistributedStatevectorPlan(
        n_wires=int(ir.n_wires),
        bsz=bsz,
        world_size=world_size,
        local_world_size=local_world_size,
        node_count=node_count,
        complex_bytes=complex_bytes,
        total_amplitudes=total_amplitudes,
        total_state_bytes=total_state_bytes,
        per_rank_state_bytes=per_rank_state_bytes,
        rank_address_bits=rank_address_bits,
        sharded_wires=sharded_wires,
        shards=shards,
        gate_plans=tuple(gate_plans),
        fusion_blocks=fusion_blocks,
        execution_segments=execution_segments,
        topology=topology,
        buffer_plans=buffer_plans,
        layers=_layer_gate_indices(gate_plans),
        communication_gate_count=communication_gate_count,
        estimated_transfer_bytes=sum(
            plan.estimated_transfer_bytes for plan in gate_plans
        ),
        distribution=distribution,
    )


def estimate_distributed_statevector_performance(
    plan: DistributedStatevectorPlan,
    *,
    bandwidth_bytes_per_second: float = 50e9,
    local_gate_amplitudes_per_second: float = 5e11,
) -> StatevectorPerformanceEstimate:
    """Estimate compute/communication time from a distributed statevector plan."""

    bandwidth = max(float(bandwidth_bytes_per_second), 1.0)
    gate_rate = max(float(local_gate_amplitudes_per_second), 1.0)
    local_compute_gates = sum(
        len(segment.gate_indices)
        for segment in plan.execution_segments
        if segment.communication == "local"
    )
    communication_gates = sum(
        len(segment.gate_indices)
        for segment in plan.execution_segments
        if segment.communication != "local"
    )
    compute_seconds = (
        (local_compute_gates + communication_gates) * plan.per_rank_state_bytes
    ) / (plan.complex_bytes * gate_rate)
    communication_seconds = plan.estimated_transfer_bytes / bandwidth
    overlap_bytes = sum(
        segment.estimated_transfer_bytes
        for segment in plan.execution_segments
        if segment.can_overlap_with_compute
    )
    overlapped_seconds = min(communication_seconds, overlap_bytes / bandwidth * 0.5)
    total = max(0.0, compute_seconds + communication_seconds - overlapped_seconds)
    if communication_seconds > compute_seconds * 1.2:
        bottleneck = "communication"
    elif compute_seconds > communication_seconds * 1.2:
        bottleneck = "compute"
    else:
        bottleneck = "balanced"
    return StatevectorPerformanceEstimate(
        total_estimated_seconds=total,
        compute_seconds=compute_seconds,
        communication_seconds=communication_seconds,
        overlapped_seconds=overlapped_seconds,
        bottleneck=bottleneck,
        bandwidth_bytes_per_second=bandwidth,
        local_gate_amplitudes_per_second=gate_rate,
    )


def _segment_peers(
    topology: StatevectorRankTopology,
    *,
    segment_index: int,
    rank: int,
) -> tuple[int, ...]:
    peers: set[int] = set()
    for edge in topology.edges:
        if segment_index not in edge.segment_indices:
            continue
        if edge.src_rank == rank:
            peers.add(edge.dst_rank)
        elif edge.dst_rank == rank:
            peers.add(edge.src_rank)
    return tuple(sorted(peers))


def _buffer_by_segment(
    buffer_plans: Sequence[StatevectorBufferPlan],
) -> dict[int, StatevectorBufferPlan]:
    return {buffer_plan.segment_index: buffer_plan for buffer_plan in buffer_plans}


def _validate_distributed_statevector_plan(
    plan: DistributedStatevectorPlan,
) -> tuple[str, ...]:
    errors: list[str] = []
    if plan.world_size != len(plan.shards):
        errors.append("world_size does not match shard count")
    if tuple(shard.rank for shard in plan.shards) != tuple(range(plan.world_size)):
        errors.append("shard ranks are not contiguous")
    expected_start = 0
    for shard in plan.shards:
        if shard.amplitude_start != expected_start:
            errors.append(
                f"shard {shard.rank} starts at {shard.amplitude_start}, expected {expected_start}"
            )
        if shard.amplitude_end < shard.amplitude_start:
            errors.append(f"shard {shard.rank} has a negative amplitude range")
        expected_start = shard.amplitude_end
    if expected_start != plan.total_amplitudes:
        errors.append("shards do not cover the full statevector")
    if tuple(segment.index for segment in plan.execution_segments) != tuple(
        range(len(plan.execution_segments))
    ):
        errors.append("execution segment indices are not contiguous")
    gate_indices = tuple(
        index for segment in plan.execution_segments for index in segment.gate_indices
    )
    if sorted(gate_indices) != list(range(len(plan.gate_plans))):
        errors.append("execution segments do not cover every gate exactly once")
    communication_segments = {
        segment.index
        for segment in plan.execution_segments
        if segment.communication != "local"
    }
    buffer_segments = {buffer_plan.segment_index for buffer_plan in plan.buffer_plans}
    if communication_segments != buffer_segments:
        errors.append("communication segments do not match buffer plans")
    segment_indices = {segment.index for segment in plan.execution_segments}
    for edge in plan.topology.edges:
        if edge.src_rank == edge.dst_rank:
            errors.append("topology contains a self edge")
        if (
            edge.src_rank < 0
            or edge.dst_rank < 0
            or edge.src_rank >= plan.world_size
            or edge.dst_rank >= plan.world_size
        ):
            errors.append("topology edge rank is out of bounds")
        if not set(edge.segment_indices).issubset(segment_indices):
            errors.append("topology edge references an unknown segment")
    if len(plan.topology.rank_coordinates) != plan.world_size:
        errors.append("rank coordinate count does not match world_size")
    return tuple(errors)


def trace_distributed_statevector_plan(
    plan: DistributedStatevectorPlan,
) -> StatevectorTraceReport:
    """Build a deterministic per-rank dry-run trace for a statevector plan."""

    errors = _validate_distributed_statevector_plan(plan)
    buffer_by_segment = _buffer_by_segment(plan.buffer_plans)
    events: list[StatevectorTraceEvent] = []
    peak_buffers = [0] * plan.world_size
    for segment in plan.execution_segments:
        buffer_plan = buffer_by_segment.get(segment.index)
        buffer_bytes = buffer_plan.peak_bytes if buffer_plan is not None else 0
        for rank in range(plan.world_size):
            peers = _segment_peers(
                plan.topology, segment_index=segment.index, rank=rank
            )
            if segment.communication == "local":
                action = "apply_local_fusion"
            elif peers:
                action = "exchange_statevector_slices"
            else:
                action = "wait_for_segment"
            peak_buffers[rank] = max(peak_buffers[rank], buffer_bytes if peers else 0)
            events.append(
                StatevectorTraceEvent(
                    rank=rank,
                    step=len(events),
                    segment_index=segment.index,
                    action=action,
                    communication=segment.communication,
                    gate_indices=segment.gate_indices,
                    peer_ranks=peers,
                    buffer_bytes=buffer_bytes if peers else 0,
                )
            )
    per_rank_counts = tuple(
        sum(1 for event in events if event.rank == rank)
        for rank in range(plan.world_size)
    )
    return StatevectorTraceReport(
        valid=not errors,
        errors=errors,
        events=tuple(events),
        per_rank_event_counts=per_rank_counts,
        peak_rank_buffer_bytes=tuple(peak_buffers),
    )


def validate_distributed_statevector_plan(
    plan: DistributedStatevectorPlan,
) -> StatevectorTraceReport:
    """Validate a distributed statevector plan and return a trace-shaped report."""

    return trace_distributed_statevector_plan(plan)

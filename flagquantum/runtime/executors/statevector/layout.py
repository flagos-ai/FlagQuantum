"""Persistent logical-to-physical qubit layouts for sharded statevectors."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any

import torch
import torch.distributed as dist

from ....core.ir import CircuitIR, ensure_circuit_ir


@dataclass(frozen=True)
class StatevectorLayoutSwap:
    """Exchange one local and one rank-address bit without restoring the layout."""

    before_instruction: int
    local_logical_qubit: int
    sharded_logical_qubit: int
    local_physical_qubit: int
    sharded_physical_qubit: int


@dataclass(frozen=True)
class StatevectorLayoutSegment:
    """A maximal instruction interval sharing one persistent qubit layout."""

    start_instruction: int
    stop_instruction: int
    logical_to_physical: tuple[int, ...]


@dataclass(frozen=True)
class PersistentStatevectorLayoutPlan:
    """Belady-style qubit residency plan for distributed statevector execution."""

    n_qubits: int
    rank_bits: int
    initial_logical_to_physical: tuple[int, ...]
    final_logical_to_physical: tuple[int, ...]
    swaps: tuple[StatevectorLayoutSwap, ...]
    segments: tuple[StatevectorLayoutSegment, ...]
    baseline_communication_gate_count: int

    @property
    def layout_swap_count(self) -> int:
        return len(self.swaps)

    @property
    def estimated_communication_event_reduction(self) -> int:
        return self.baseline_communication_gate_count - self.layout_swap_count


def _local_bit_view(
    amplitudes: torch.Tensor,
    *,
    bit_position: int,
    bit_value: int,
) -> torch.Tensor:
    inner = 1 << bit_position
    outer = amplitudes.shape[-1] // (inner << 1)
    return amplitudes.reshape(amplitudes.shape[0], outer, 2, inner)[:, :, bit_value, :]


def distributed_swap_rank_local_bits(
    amplitudes: torch.Tensor,
    *,
    rank: int,
    n_qubits: int,
    rank_bits: int,
    local_physical_qubit: int,
    sharded_physical_qubit: int,
    process_group: Any | None = None,
    output: torch.Tensor | None = None,
    send_buffer: torch.Tensor | None = None,
    receive_buffer: torch.Tensor | None = None,
    fused_matrix: torch.Tensor | None = None,
    tag: int = 0,
) -> tuple[torch.Tensor, int, int]:
    """Persistently transpose one rank-address bit with one local index bit.

    Exactly half of the local shard is sent to the rank whose selected address
    bit differs.  The returned storage uses the transposed physical layout.
    """

    if amplitudes.ndim != 2:
        raise ValueError("statevector amplitudes must have shape [batch, local]")
    if rank_bits < 1 or rank_bits >= n_qubits:
        raise ValueError("rank_bits must leave at least one local qubit")
    local_limit = n_qubits - rank_bits
    if not 0 <= local_physical_qubit < local_limit:
        raise ValueError("local_physical_qubit is not a local index bit")
    if not local_limit <= sharded_physical_qubit < n_qubits:
        raise ValueError("sharded_physical_qubit is not a rank-address bit")

    local_bit_position = n_qubits - local_physical_qubit - 1 - rank_bits
    rank_bit_position = n_qubits - sharded_physical_qubit - 1
    rank_mask = 1 << rank_bit_position
    rank_value = (rank >> rank_bit_position) & 1
    peer = rank ^ rank_mask
    global_peer = (
        dist.get_global_rank(process_group, peer) if process_group is not None else peer
    )
    send_value = 1 - rank_value
    send_view = _local_bit_view(
        amplitudes, bit_position=local_bit_position, bit_value=send_value
    )
    if send_buffer is None:
        send = send_view.contiguous()
    else:
        if (
            send_buffer.numel() != send_view.numel()
            or send_buffer.dtype != send_view.dtype
        ):
            raise ValueError("send_buffer must match the packed half-shard")
        send = send_buffer.reshape_as(send_view)
        send.copy_(send_view)
    received = torch.empty_like(send) if receive_buffer is None else receive_buffer
    if (
        received.numel() != send.numel()
        or received.dtype != send.dtype
        or received.data_ptr() == send.data_ptr()
    ):
        raise ValueError(
            "receive_buffer must match and not alias the packed half-shard"
        )
    received = received.reshape_as(send)
    operations = [
        dist.P2POp(dist.isend, send, global_peer, process_group, tag),
        dist.P2POp(dist.irecv, received, global_peer, process_group, tag),
    ]
    for request in dist.batch_isend_irecv(operations):
        request.wait()

    result = amplitudes.clone() if output is None else output
    if fused_matrix is not None:
        if result.data_ptr() != amplitudes.data_ptr():
            raise ValueError("fused transpose gate currently requires in-place output")
        from ....kernels.triton.statevector_gates import (
            apply_complex64_transpose_1q_inplace,
        )

        apply_complex64_transpose_1q_inplace(
            result,
            received,
            fused_matrix,
            bit_position=local_bit_position,
            exchanged_bit_value=send_value,
        )
    else:
        if result.data_ptr() != amplitudes.data_ptr():
            result.copy_(amplitudes)
        _local_bit_view(
            result, bit_position=local_bit_position, bit_value=send_value
        ).copy_(received)
    transferred_bytes = send.numel() * send.element_size()
    return result, 1, transferred_bytes


def _next_use(
    instructions: Sequence[Any],
    *,
    after: int,
    logical_qubit: int,
) -> int:
    for index in range(after + 1, len(instructions)):
        if logical_qubit in instructions[index].wires:
            return index
    return len(instructions) + logical_qubit


def schedule_statevector_dependency_dag(
    circuit_or_ir: Any,
    *,
    world_size: int,
) -> CircuitIR:
    """List-schedule independent gates to keep the active qubit set resident.

    Instructions that share a qubit retain their original relative order.
    Instructions with no dependency path between them act on disjoint qubits and
    therefore commute, so the scheduler may choose among them without changing
    circuit semantics.  The ready instruction touching the fewest rank-address
    qubits is emitted first while a lightweight residency model follows the
    selected order.

    This is deliberately topology-agnostic: it improves alternating matchings,
    sparse ansatzes, and other dependency DAGs without recognizing workload
    names or gate patterns.
    """

    ir: CircuitIR = ensure_circuit_ir(circuit_or_ir)
    if ir.metadata.get("statevector_dependency_scheduled", False):
        return ir
    if world_size <= 1 or world_size & (world_size - 1):
        return ir
    rank_bits = world_size.bit_length() - 1
    if rank_bits >= ir.n_wires or len(ir.instructions) < 2:
        return ir

    instructions = ir.instructions
    instruction_count = len(instructions)
    per_qubit: list[list[int]] = [[] for _ in range(ir.n_wires)]
    for index, instruction in enumerate(instructions):
        for qubit in instruction.wires:
            per_qubit[int(qubit)].append(index)
    cursors = [0] * ir.n_wires
    completed: set[int] = set()
    scheduled: list[Any] = []
    sharded_positions = frozenset(range(ir.n_wires - rank_bits, ir.n_wires))
    mapping = list(range(ir.n_wires))

    def is_ready(index: int) -> bool:
        return all(
            per_qubit[int(qubit)][cursors[int(qubit)]] == index
            for qubit in instructions[index].wires
        )

    def pending_use(logical_qubit: int) -> int:
        cursor = cursors[logical_qubit]
        uses = per_qubit[logical_qubit]
        return uses[cursor] if cursor < len(uses) else instruction_count + logical_qubit

    while len(scheduled) < instruction_count:
        ready = [
            index
            for index in range(instruction_count)
            if index not in completed and is_ready(index)
        ]
        if not ready:
            raise RuntimeError("statevector dependency scheduler reached a cycle")
        index = min(
            ready,
            key=lambda candidate: (
                sum(
                    mapping[int(qubit)] in sharded_positions
                    for qubit in instructions[candidate].wires
                ),
                candidate,
            ),
        )
        requested = {int(qubit) for qubit in instructions[index].wires}
        while True:
            missing = [
                qubit for qubit in requested if mapping[qubit] in sharded_positions
            ]
            if not missing:
                break
            sharded_logical = min(missing, key=pending_use)
            residents = [
                qubit
                for qubit in range(ir.n_wires)
                if mapping[qubit] not in sharded_positions and qubit not in requested
            ]
            if not residents:
                break
            local_logical = max(residents, key=pending_use)
            mapping[local_logical], mapping[sharded_logical] = (
                mapping[sharded_logical],
                mapping[local_logical],
            )
        scheduled.append(instructions[index])
        completed.add(index)
        for qubit in requested:
            cursors[qubit] += 1

    reordered = tuple(scheduled)
    return replace(
        ir,
        instructions=reordered,
        metadata={
            **ir.metadata,
            "statevector_dependency_scheduled": True,
            "statevector_dependency_schedule_changed": reordered != instructions,
        },
    )


def plan_persistent_statevector_layout(
    circuit_or_ir: Any,
    *,
    world_size: int,
    initial_logical_to_physical: Sequence[int] | None = None,
) -> PersistentStatevectorLayoutPlan:
    """Keep requested qubits local using farthest-next-use residency replacement.

    Rank-address positions form the non-resident set.  When an instruction needs
    a non-resident logical qubit, one resident qubit not used by that instruction
    is evicted.  The new layout remains active for following instructions.
    """

    ir: CircuitIR = ensure_circuit_ir(circuit_or_ir)
    if world_size < 1 or world_size & (world_size - 1):
        raise ValueError(
            "persistent statevector layouts require power-of-two world_size"
        )
    rank_bits = world_size.bit_length() - 1
    if rank_bits >= ir.n_wires:
        raise ValueError("world_size must leave at least one local statevector qubit")
    initial = tuple(
        range(ir.n_wires)
        if initial_logical_to_physical is None
        else map(int, initial_logical_to_physical)
    )
    if tuple(sorted(initial)) != tuple(range(ir.n_wires)):
        raise ValueError("initial_logical_to_physical must be a qubit permutation")

    sharded_positions = frozenset(range(ir.n_wires - rank_bits, ir.n_wires))
    mapping = list(initial)
    baseline = sum(
        any(mapping[int(qubit)] in sharded_positions for qubit in instruction.wires)
        for instruction in ir.instructions
    )
    swaps: list[StatevectorLayoutSwap] = []
    segments: list[StatevectorLayoutSegment] = []
    segment_start = 0
    segment_layout = tuple(mapping)

    for index, instruction in enumerate(ir.instructions):
        requested = {int(qubit) for qubit in instruction.wires}
        while True:
            missing = [
                qubit for qubit in requested if mapping[qubit] in sharded_positions
            ]
            if not missing:
                break
            sharded_logical = min(
                missing,
                key=lambda qubit: _next_use(
                    ir.instructions, after=index, logical_qubit=qubit
                ),
            )
            residents = [
                qubit
                for qubit in range(ir.n_wires)
                if mapping[qubit] not in sharded_positions and qubit not in requested
            ]
            if not residents:
                # A gate wider than local capacity cannot be made entirely local.
                break
            local_logical = max(
                residents,
                key=lambda qubit: _next_use(
                    ir.instructions, after=index, logical_qubit=qubit
                ),
            )
            if segment_start < index:
                segments.append(
                    StatevectorLayoutSegment(segment_start, index, segment_layout)
                )
            local_physical = mapping[local_logical]
            sharded_physical = mapping[sharded_logical]
            swaps.append(
                StatevectorLayoutSwap(
                    before_instruction=index,
                    local_logical_qubit=local_logical,
                    sharded_logical_qubit=sharded_logical,
                    local_physical_qubit=local_physical,
                    sharded_physical_qubit=sharded_physical,
                )
            )
            mapping[local_logical], mapping[sharded_logical] = (
                sharded_physical,
                local_physical,
            )
            segment_start = index
            segment_layout = tuple(mapping)

    if segment_start < len(ir.instructions):
        segments.append(
            StatevectorLayoutSegment(
                segment_start, len(ir.instructions), segment_layout
            )
        )
    return PersistentStatevectorLayoutPlan(
        n_qubits=ir.n_wires,
        rank_bits=rank_bits,
        initial_logical_to_physical=initial,
        final_logical_to_physical=tuple(mapping),
        swaps=tuple(swaps),
        segments=tuple(segments),
        baseline_communication_gate_count=baseline,
    )


__all__ = (
    "PersistentStatevectorLayoutPlan",
    "StatevectorLayoutSegment",
    "StatevectorLayoutSwap",
    "distributed_swap_rank_local_bits",
    "plan_persistent_statevector_layout",
    "schedule_statevector_dependency_dag",
)

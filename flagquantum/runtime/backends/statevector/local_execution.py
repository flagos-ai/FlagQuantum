"""Local and torch-distributed statevector execution helpers."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch
import torch.distributed as dist

from ....core.ir import Instruction, ensure_circuit_ir
from ....core.parameters import value_to_tensor
from ....ops.matrices import GATE_MAT_DICT
from ...distributed.backend_policy import (
    DistributedBackendPolicy,
    resolve_distributed_backend_policy,
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
_PARAM_ALIASES = {
    "rx": ("theta",),
    "ry": ("theta",),
    "rz": ("theta",),
    "phase": ("theta",),
    "p": ("theta",),
    "u1": ("theta",),
    "u2": ("phi", "lbd"),
    "u3": ("theta", "phi", "lbd"),
    "crx": ("theta",),
    "cry": ("theta",),
    "crz": ("theta",),
    "cphase": ("theta",),
    "rxx": ("theta",),
    "ryy": ("theta",),
    "rzz": ("theta",),
}


from .models import (  # noqa: E402
    DistributedStatevectorPlan,
    LocalDistributedStatevectorResult,
    StatevectorCorrectnessRunSpec,
    StatevectorExecutionSegment,
    StatevectorExecutorReport,
    StatevectorRankResult,
    StatevectorSegmentResult,
    StatevectorShard,
    StatevectorShardState,
    StatevectorTransportEvent,
    StatevectorTransportReport,
)
from .planning import (  # noqa: E402
    plan_distributed_statevector,
    trace_distributed_statevector_plan,
)


def _segment_by_index(
    execution_segments: Sequence[StatevectorExecutionSegment],
) -> dict[int, StatevectorExecutionSegment]:
    return {segment.index: segment for segment in execution_segments}


def _shard_by_rank(shards: Sequence[StatevectorShard]) -> dict[int, StatevectorShard]:
    return {shard.rank: shard for shard in shards}


def _rank_global_indices(
    plan: DistributedStatevectorPlan, rank: int, *, device: torch.device
) -> torch.Tensor:
    if plan.distribution == "qubit_address_sharded" and plan.sharded_wires:
        shard = _shard_by_rank(plan.shards)[int(rank)]
        local_indices = torch.arange(
            shard.local_amplitudes, dtype=torch.long, device=device
        )
        global_indices = torch.zeros_like(local_indices)
        coordinates = plan.topology.rank_coordinates[int(rank)]
        sharded_coordinates = {
            int(wire): int(coord)
            for coord, wire in zip(coordinates, plan.sharded_wires)
        }
        local_bit = 0
        for wire in range(plan.n_wires - 1, -1, -1):
            destination_mask = _wire_mask(plan.n_wires, wire)
            if wire in sharded_coordinates:
                if sharded_coordinates[wire]:
                    global_indices |= destination_mask
                continue
            global_indices |= ((local_indices >> local_bit) & 1) * destination_mask
            local_bit += 1
        return global_indices
    shard = _shard_by_rank(plan.shards)[int(rank)]
    return torch.arange(
        shard.amplitude_start, shard.amplitude_end, dtype=torch.long, device=device
    )


def use_compact_global_indices(
    plan: DistributedStatevectorPlan,
    rank: int,
    *,
    local_amplitude_threshold: int = 1 << 24,
) -> bool:
    """Return whether shard ownership can be represented without an index tensor."""

    return bool(
        plan.distribution == "qubit_address_sharded"
        or plan.shards[int(rank)].local_amplitudes >= int(local_amplitude_threshold)
    )


def initialize_statevector_shard(
    plan: DistributedStatevectorPlan,
    *,
    rank: int,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.complex64,
    compact_global_indices: bool = False,
) -> StatevectorShardState:
    """Initialize the ``|0...0>`` state for one rank-local statevector shard."""

    shard = _shard_by_rank(plan.shards)[int(rank)]
    resolved_device = torch.device(device or "cpu")
    global_indices = (
        torch.empty(0, dtype=torch.long, device=resolved_device)
        if compact_global_indices
        else _rank_global_indices(plan, int(rank), device=resolved_device)
    )
    amplitudes = torch.zeros(
        (plan.bsz, shard.local_amplitudes),
        dtype=dtype,
        device=resolved_device,
    )
    if shard.amplitude_start == 0:
        amplitudes[:, 0] = 1
    return StatevectorShardState(
        rank=int(rank),
        shard=shard,
        amplitudes=amplitudes,
        global_indices=global_indices,
    )


def _wire_mask(n_wires: int, wire: int) -> int:
    return 1 << (int(n_wires) - int(wire) - 1)


def _basis_offset(n_wires: int, wires: Sequence[int], basis_index: int) -> int:
    offset = 0
    width = len(tuple(wires))
    for pos, wire in enumerate(tuple(wires)):
        if (int(basis_index) >> (width - pos - 1)) & 1:
            offset |= _wire_mask(n_wires, int(wire))
    return offset


def _basis_indices_for_wires(
    global_indices: torch.Tensor, *, n_wires: int, wires: Sequence[int]
) -> torch.Tensor:
    basis = torch.zeros_like(global_indices, dtype=torch.long)
    width = len(tuple(wires))
    for pos, wire in enumerate(tuple(wires)):
        bit = (global_indices >> (int(n_wires) - int(wire) - 1)) & 1
        basis = basis | (bit.to(dtype=torch.long) << (width - pos - 1))
    return basis


def apply_gate_to_statevector_shard(
    shard_state: StatevectorShardState,
    matrix: torch.Tensor,
    wires: Sequence[int],
    *,
    plan: DistributedStatevectorPlan,
) -> StatevectorShardState:
    """Apply a local gate to one shard without materializing the full state.

    The gate must be local to the rank shard: every amplitude touched by the
    gate must remain inside ``shard_state.shard``. Cross-shard gates are handled
    by communication kernels and intentionally raise here.
    """

    wires = tuple(int(wire) for wire in wires)
    width = len(wires)
    gate_dim = 2**width
    if matrix.shape[-2:] != (gate_dim, gate_dim):
        raise ValueError(
            f"Gate on {width} wires requires matrix shape {(gate_dim, gate_dim)}."
        )
    matrix = matrix.to(
        device=shard_state.amplitudes.device, dtype=shard_state.amplitudes.dtype
    )
    shard = shard_state.shard
    if torch.count_nonzero(matrix - torch.diag(torch.diagonal(matrix))) == 0:
        basis_indices = _basis_indices_for_wires(
            shard_state.global_indices,
            n_wires=plan.n_wires,
            wires=wires,
        )
        factors = torch.diagonal(matrix)[basis_indices].reshape(1, -1)
        return StatevectorShardState(
            rank=shard_state.rank,
            shard=shard,
            amplitudes=shard_state.amplitudes * factors,
            global_indices=shard_state.global_indices,
        )
    out = shard_state.amplitudes.clone()
    masks = tuple(_wire_mask(plan.n_wires, wire) for wire in wires)
    local_by_basis = tuple(
        _basis_offset(plan.n_wires, wires, basis) for basis in range(gate_dim)
    )
    position_by_global = {
        int(global_index): local_index
        for local_index, global_index in enumerate(
            shard_state.global_indices.detach().cpu().tolist()
        )
    }
    for global_index in position_by_global:
        if any(global_index & mask for mask in masks):
            continue
        global_indices = tuple(global_index | offset for offset in local_by_basis)
        if any(index not in position_by_global for index in global_indices):
            raise ValueError(
                "Gate touches amplitudes outside the local shard; communication is required."
            )
        local_indices = tuple(position_by_global[index] for index in global_indices)
        vector = shard_state.amplitudes[:, local_indices]
        out[:, local_indices] = vector @ matrix.transpose(-2, -1)
    return StatevectorShardState(
        rank=shard_state.rank,
        shard=shard,
        amplitudes=out,
        global_indices=shard_state.global_indices,
    )


def _parameter_tensor(
    name: str, params: Any, *, device: torch.device
) -> torch.Tensor | None:
    aliases = _PARAM_ALIASES.get(name)
    if aliases is None:
        return None
    values = []
    for alias in aliases:
        if alias not in params:
            return None
        tensor = value_to_tensor(params[alias]).to(device=device)
        values.append(tensor.reshape(()) if tensor.ndim == 0 else tensor)
    return torch.stack(values, dim=-1)


def _instruction_matrix(
    instruction: Instruction, *, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    name = instruction.name
    if name == "cnot":
        name = "cx"
    if instruction.matrix is not None:
        matrix = getattr(instruction.matrix, "tensor", instruction.matrix)
        return torch.as_tensor(matrix, dtype=dtype, device=device).reshape(
            2 ** len(instruction.wires), -1
        )
    gate = GATE_MAT_DICT[name]
    if callable(gate):
        params = _parameter_tensor(name, instruction.params, device=device)
        if params is None:
            raise ValueError(f"Gate {name!r} requires parameters.")
        matrix = gate(params)
    else:
        matrix = gate
    if matrix.ndim == 3:
        if matrix.shape[0] != 1:
            raise ValueError(
                "Local distributed simulator currently expects scalar gate parameters."
            )
        matrix = matrix[0]
    return matrix.to(device=device, dtype=dtype)


def _reconstruct_from_shards(
    shards: Sequence[StatevectorShardState], plan: DistributedStatevectorPlan
) -> torch.Tensor:
    first = shards[0]
    state = torch.zeros(
        (plan.bsz, plan.total_amplitudes),
        dtype=first.amplitudes.dtype,
        device=first.amplitudes.device,
    )
    for shard in shards:
        state[:, shard.global_indices] = shard.amplitudes
    return state


def _scatter_to_shards(
    state: torch.Tensor,
    shards: Sequence[StatevectorShardState],
) -> tuple[StatevectorShardState, ...]:
    return tuple(
        StatevectorShardState(
            rank=shard.rank,
            shard=shard.shard,
            amplitudes=state[:, shard.global_indices].clone(),
            global_indices=shard.global_indices,
        )
        for shard in shards
    )


def apply_gate_to_statevector_shards(
    shards: Sequence[StatevectorShardState],
    matrix: torch.Tensor,
    wires: Sequence[int],
    *,
    plan: DistributedStatevectorPlan,
) -> tuple[StatevectorShardState, ...]:
    """Apply a gate across rank-local shards without materializing the full state.

    The local development executor owns all rank shards in one process, but this
    kernel keeps the same data movement shape as production sharding: amplitudes
    are gathered only for the basis group touched by the gate and written back to
    their owning shards. It is therefore a semantic test bed for future
    torch.distributed P2P/all-to-all numerical kernels, not a dense-state
    shortcut.
    """

    shards = tuple(shards)
    if not shards:
        return ()
    wires = tuple(int(wire) for wire in wires)
    width = len(wires)
    gate_dim = 2**width
    if matrix.shape[-2:] != (gate_dim, gate_dim):
        raise ValueError(
            f"Gate on {width} wires requires matrix shape {(gate_dim, gate_dim)}."
        )

    reference = shards[0].amplitudes
    matrix = matrix.to(device=reference.device, dtype=reference.dtype)
    if torch.count_nonzero(matrix - torch.diag(torch.diagonal(matrix))) == 0:
        return tuple(
            apply_gate_to_statevector_shard(shard, matrix, wires, plan=plan)
            for shard in shards
        )

    masks = tuple(_wire_mask(plan.n_wires, wire) for wire in wires)
    offsets = tuple(
        _basis_offset(plan.n_wires, wires, basis) for basis in range(gate_dim)
    )
    positions_by_global: dict[int, tuple[int, int]] = {}
    global_indices_by_shard = []
    for shard_index, shard in enumerate(shards):
        indices = tuple(
            int(item) for item in shard.global_indices.detach().cpu().tolist()
        )
        global_indices_by_shard.append(indices)
        for local_index, global_index in enumerate(indices):
            positions_by_global[global_index] = (shard_index, local_index)

    out_amplitudes = [shard.amplitudes.clone() for shard in shards]
    processed_bases: set[int] = set()
    for indices in global_indices_by_shard:
        for global_index in indices:
            base = global_index
            for mask in masks:
                base &= ~mask
            if base in processed_bases:
                continue
            processed_bases.add(base)
            group_indices = tuple(base | offset for offset in offsets)
            missing = tuple(
                index for index in group_indices if index not in positions_by_global
            )
            if missing:
                raise ValueError(
                    f"Shard plan does not cover gate basis group; missing amplitudes {missing}."
                )
            locations = tuple(positions_by_global[index] for index in group_indices)
            vector = torch.stack(
                [
                    shards[shard_index].amplitudes[:, local_index]
                    for shard_index, local_index in locations
                ],
                dim=-1,
            )
            updated = vector @ matrix.transpose(-2, -1)
            for basis_index, (shard_index, local_index) in enumerate(locations):
                out_amplitudes[shard_index][:, local_index] = updated[:, basis_index]

    return tuple(
        StatevectorShardState(
            rank=shard.rank,
            shard=shard.shard,
            amplitudes=out_amplitudes[shard_index],
            global_indices=shard.global_indices,
        )
        for shard_index, shard in enumerate(shards)
    )


def _apply_matrix_to_full_state(
    state: torch.Tensor,
    matrix: torch.Tensor,
    wires: Sequence[int],
    *,
    n_wires: int,
) -> torch.Tensor:
    wires = tuple(int(wire) for wire in wires)
    gate_dim = 2 ** len(wires)
    out = state.clone()
    masks = tuple(_wire_mask(n_wires, wire) for wire in wires)
    offsets = tuple(_basis_offset(n_wires, wires, basis) for basis in range(gate_dim))
    for base in range(2**n_wires):
        if any(base & mask for mask in masks):
            continue
        indices = tuple(base | offset for offset in offsets)
        vector = state[:, indices]
        out[:, indices] = vector @ matrix.transpose(-2, -1)
    return out


def simulate_distributed_statevector_local(
    circuit_or_ir: Any,
    *,
    world_size: int = 2,
    bsz: int = 1,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.complex64,
    backend_policy: DistributedBackendPolicy | None = None,
    jax_distributed_plan: Mapping[str, Any] | None = None,
) -> LocalDistributedStatevectorResult:
    """Simulate distributed statevector execution in one CPU/GPU process.

    This is a development and CI tool: it uses the same distributed plan and
    rank-local shard ownership as torchrun deployment, while simulating
    cross-rank communication locally so users can debug without a cluster.
    """

    ir = ensure_circuit_ir(circuit_or_ir)
    backend_policy = backend_policy or resolve_distributed_backend_policy(
        profile="development"
    )
    plan = plan_distributed_statevector(
        ir,
        world_size=world_size,
        bsz=bsz,
        complex_bytes=16 if dtype == torch.complex128 else 8,
    )
    resolved_device = torch.device(device)
    shards = tuple(
        initialize_statevector_shard(
            plan, rank=rank, device=resolved_device, dtype=dtype
        )
        for rank in range(plan.world_size)
    )
    local_gate_count = 0
    distributed_gate_count = 0
    simulated_comm_count = 0
    simulated_comm_bytes = 0
    for instruction, gate_plan in zip(ir.instructions, plan.gate_plans):
        matrix = _instruction_matrix(instruction, device=resolved_device, dtype=dtype)
        if gate_plan.communication == "local":
            shards = tuple(
                apply_gate_to_statevector_shard(
                    shard, matrix, instruction.wires, plan=plan
                )
                for shard in shards
            )
            local_gate_count += 1
            continue
        shards = apply_gate_to_statevector_shards(
            shards, matrix, instruction.wires, plan=plan
        )
        distributed_gate_count += 1
        simulated_comm_count += 1
        simulated_comm_bytes += int(gate_plan.estimated_transfer_bytes)
    return LocalDistributedStatevectorResult(
        state=_reconstruct_from_shards(shards, plan),
        shards=shards,
        plan=plan,
        local_gate_count=local_gate_count,
        distributed_gate_count=distributed_gate_count,
        simulated_communication_count=simulated_comm_count,
        simulated_communication_bytes=simulated_comm_bytes,
        full_state_reconstruction_count=1,
        backend_policy=backend_policy,
        jax_distributed_plan=jax_distributed_plan,
    )


def execute_distributed_statevector_dry_run(
    plan: DistributedStatevectorPlan,
) -> StatevectorExecutorReport:
    """Execute a deterministic metadata-only distributed statevector dry run."""

    trace = trace_distributed_statevector_plan(plan)
    segment_by_index = _segment_by_index(plan.execution_segments)
    shard_by_rank = _shard_by_rank(plan.shards)
    rank_results: list[StatevectorRankResult] = []
    for rank in range(plan.world_size):
        segment_results: list[StatevectorSegmentResult] = []
        for event in (event for event in trace.events if event.rank == rank):
            segment = segment_by_index[event.segment_index]
            local_bytes = 0
            communication_bytes = 0
            if event.action == "apply_local_fusion":
                local_bytes = shard_by_rank[rank].local_state_bytes * len(
                    segment.gate_indices
                )
            elif event.action == "exchange_statevector_slices":
                communication_bytes = event.buffer_bytes
                local_bytes = shard_by_rank[rank].local_state_bytes * len(
                    segment.gate_indices
                )
            segment_results.append(
                StatevectorSegmentResult(
                    rank=rank,
                    segment_index=event.segment_index,
                    action=event.action,
                    gate_indices=event.gate_indices,
                    local_bytes_processed=local_bytes,
                    communication_bytes=communication_bytes,
                    buffer_bytes=event.buffer_bytes,
                    peer_ranks=event.peer_ranks,
                )
            )
        rank_results.append(
            StatevectorRankResult(
                rank=rank,
                segment_results=tuple(segment_results),
                local_bytes_processed=sum(
                    result.local_bytes_processed for result in segment_results
                ),
                communication_bytes=sum(
                    result.communication_bytes for result in segment_results
                ),
                peak_buffer_bytes=max(
                    (result.buffer_bytes for result in segment_results), default=0
                ),
            )
        )
    return StatevectorExecutorReport(
        valid=trace.valid,
        errors=trace.errors,
        rank_results=tuple(rank_results),
        total_local_bytes_processed=sum(
            result.local_bytes_processed for result in rank_results
        ),
        total_communication_bytes=sum(
            result.communication_bytes for result in rank_results
        ),
        peak_buffer_bytes=max(
            (result.peak_buffer_bytes for result in rank_results), default=0
        ),
        trace=trace,
    )


def build_statevector_correctness_run_spec(
    plan: DistributedStatevectorPlan,
    *,
    entrypoint: str = "tests/distributed/statevector_correctness.py",
    backend: str = "gloo",
) -> StatevectorCorrectnessRunSpec:
    """Build a torchrun launch spec for multi-rank statevector correctness."""

    return StatevectorCorrectnessRunSpec(
        world_size=plan.world_size,
        backend=backend,
        entrypoint=entrypoint,
        args=(
            "--world-size",
            str(plan.world_size),
            "--n-wires",
            str(plan.n_wires),
            "--distribution",
            plan.distribution,
            "--topology",
            plan.topology.layout,
        ),
        env=(
            ("FQ_DIST_BACKEND", backend),
            ("FQ_STATEVECTOR_DISTRIBUTION", plan.distribution),
            ("FQ_STATEVECTOR_TOPOLOGY", plan.topology.layout),
        ),
        expected_checks=(
            "rank_initialization",
            "plan_validation",
            "local_vs_distributed_expectation",
            "trace_event_coverage",
            "buffer_budget_respected",
        ),
    )


def _transport_payload(rank: int, segment_index: int) -> int:
    return int(rank) * 100000 + int(segment_index)


def _pair_exchange_transport(
    *,
    rank: int,
    segment_index: int,
    peer: int,
    device: torch.device,
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[str, ...]]:
    send = torch.tensor(
        [_transport_payload(rank, segment_index)], dtype=torch.int64, device=device
    )
    recv = torch.empty_like(send)
    errors: list[str] = []
    requests = dist.batch_isend_irecv(
        [
            dist.P2POp(dist.isend, send, peer),
            dist.P2POp(dist.irecv, recv, peer),
        ]
    )
    for request in requests:
        request.wait()
    received = int(recv.item())
    expected = _transport_payload(peer, segment_index)
    if received != expected:
        errors.append(
            f"rank {rank} received {received} from {peer}, expected {expected}"
        )
    return (int(send.item()),), (received,), tuple(errors)


def _all_to_all_transport(
    *,
    rank: int,
    world_size: int,
    segment_index: int,
    device: torch.device,
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[str, ...]]:
    send = torch.tensor(
        [_transport_payload(rank, segment_index)], dtype=torch.int64, device=device
    )
    gathered = [torch.empty_like(send) for _ in range(world_size)]
    dist.all_gather(gathered, send)
    received = tuple(int(item.item()) for item in gathered)
    expected = tuple(
        _transport_payload(peer, segment_index) for peer in range(world_size)
    )
    errors = ()
    if received != expected:
        errors = (f"rank {rank} gathered {received}, expected {expected}",)
    return (int(send.item()),), received, errors


def execute_distributed_statevector_transport(
    plan: DistributedStatevectorPlan,
    *,
    device: torch.device | str | None = None,
) -> StatevectorTransportReport:
    """Run real torch.distributed transport for pair-exchange and all-to-all segments."""

    if not dist.is_available() or not dist.is_initialized():
        rank = 0
        world_size = 1
        return StatevectorTransportReport(
            rank=rank,
            world_size=world_size,
            valid=plan.world_size == 1,
            errors=(
                ()
                if plan.world_size == 1
                else ("torch.distributed is not initialized",)
            ),
            events=(),
            pair_exchange_count=0,
            all_to_all_count=0,
        )

    rank = int(dist.get_rank())
    world_size = int(dist.get_world_size())
    resolved_device = torch.device(device or "cpu")
    trace = plan.trace()
    errors: list[str] = []
    events: list[StatevectorTransportEvent] = []
    for event in trace.events:
        if event.rank != rank or event.action != "exchange_statevector_slices":
            continue
        if event.communication == "pair_exchange":
            if len(event.peer_ranks) != 1:
                errors.append(f"rank {rank} pair_exchange has peers {event.peer_ranks}")
                continue
            sent, received, step_errors = _pair_exchange_transport(
                rank=rank,
                segment_index=event.segment_index,
                peer=event.peer_ranks[0],
                device=resolved_device,
            )
        elif event.communication in {"all_to_all", "indexed_all_to_all"}:
            sent, received, step_errors = _all_to_all_transport(
                rank=rank,
                world_size=world_size,
                segment_index=event.segment_index,
                device=resolved_device,
            )
        else:
            continue
        errors.extend(step_errors)
        events.append(
            StatevectorTransportEvent(
                rank=rank,
                segment_index=event.segment_index,
                communication=event.communication,
                peer_ranks=event.peer_ranks,
                sent_values=sent,
                received_values=received,
            )
        )
    pair_count = sum(1 for event in events if event.communication == "pair_exchange")
    all_count = sum(
        1
        for event in events
        if event.communication in {"all_to_all", "indexed_all_to_all"}
    )
    return StatevectorTransportReport(
        rank=rank,
        world_size=world_size,
        valid=not errors,
        errors=tuple(errors),
        events=tuple(events),
        pair_exchange_count=pair_count,
        all_to_all_count=all_count,
    )

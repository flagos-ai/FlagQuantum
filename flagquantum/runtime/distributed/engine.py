"""Distributed MPS and tensor-network execution plans.

This module provides FlagQuantum-native distributed result objects without
depending on an external graph or tensor-network package. Development backends
must preserve the same rank ownership and communication semantics as production
backends, while production backends are expected to execute one logical workload
across rank-local shards instead of replicating the full circuit per rank.
"""

from __future__ import annotations

import json
from importlib import import_module
from typing import Any, Mapping, Sequence

import torch
import torch.distributed as dist

from ...core.ir import CircuitIR, Instruction, ensure_circuit_ir
from ...simulation.mps.entrypoints import run_mps
from ...simulation.mps.models import (
    MPSAdaptiveBondPlan,
    MPSConfig,
    MPSTruncationRecord,
)
from ...simulation.mps.rank_local import (
    apply_one_mps_tensor as _apply_one_mps_tensor,
)
from ...simulation.mps.rank_local import (
    apply_two_mps_tensors_with_info as _apply_two_mps_tensors_with_info,
)
from ...simulation.mps.rank_local import (
    instruction_matrix_for_mps as _instruction_matrix_for_mps,
)
from ...simulation.mps.rank_local import (
    tensor_nbytes as _tensor_nbytes,
)
from ...simulation.mps.state import MPSState
from ..backends.jax import plan_jax_distributed_quantum_backend
from .identity import (
    DistributedIdentity,
    DistributedIdentityError,
    backend_uses_accelerator_tensors,
    require_verified_flagcx,
)
from .models import (
    DistributedBoundaryProtocol,
    DistributedBoundarySync,
    DistributedMPSState,
    DistributedShardPlan,
    ShardedMPSState,
    TorchDistributedContext,
    _broadcast_mps_site_tensor,
    _mps_shards,
    _resolve_backend_policy,
    _should_use_torch_distributed,
    destroy_torch_distributed,
    init_torch_distributed,
    torch_distributed_is_available,
)
from .mps_transport import (
    _recv_tensor_async_p2p,
    _recv_tensor_p2p,
    _send_tensor_async_p2p,
    _send_tensor_p2p,
)

_PARAM_ALIASES = {
    "rx": ("theta",),
    "ry": ("theta",),
    "rz": ("theta",),
    "phase": ("theta",),
    "p": ("theta",),
    "u1": ("theta",),
    "u2": ("phi", "lbd"),
    "u3": ("theta", "phi", "lbd"),
}


def run_distributed_mps(
    circuit_or_ir: Any,
    *,
    world_size: int = 1,
    distributed_executor: str = "auto",
    backend: str | None = None,
    init_method: str = "env://",
    rank: int | None = None,
    local_rank: int | None = None,
    **options: Any,
) -> DistributedMPSState:
    """Run MPS with torch.distributed rank-shard orchestration."""

    adaptive = bool(options.pop("adaptive", False))
    global_error_budget = options.pop("global_error_budget", None)
    initial_max_bond = options.pop("initial_max_bond", None)
    max_bond_cap = options.pop("max_bond_cap", None)
    growth_factor = float(options.pop("growth_factor", 2.0))
    if adaptive and initial_max_bond is not None:
        options["max_bond"] = int(initial_max_bond)
    gradient_fastpath = bool(options.pop("gradient_fastpath", False))
    strict_sharded = bool(options.pop("strict_sharded", False))
    backend_policy = _resolve_backend_policy(options)

    context = None
    use_torch = _should_use_torch_distributed(
        distributed_executor,
        backend_policy,
        world_size=world_size,
    )
    if use_torch:
        context = init_torch_distributed(
            backend=backend,
            init_method=init_method,
            rank=rank,
            world_size=world_size,
            local_rank=local_rank,
            device=options.get("device"),
            force_initialize=distributed_executor == "torch",
        )
        world_size = context.world_size
        options["device"] = context.device

    def _run_once(run_options: Mapping[str, Any]) -> tuple[MPSState, dict[str, Any]]:
        if context is not None and context.initialized:
            return _run_mps_site_sharded_sync(
                circuit_or_ir,
                context=context,
                world_size=world_size,
                strict_sharded=strict_sharded,
                **dict(run_options),
            )
        if world_size > 1 and backend_policy.torch_backend == "local_tensor":
            return _run_mps_site_sharded_local(
                circuit_or_ir,
                world_size=world_size,
                strict_sharded=strict_sharded,
                **dict(run_options),
            )
        local_state = run_mps(circuit_or_ir, **_mps_run_options(run_options))
        return local_state, {
            "owned_instruction_count": len(tuple(_as_ir(circuit_or_ir))),
            "sharded_kernel_count": 0,
            "tensor_sync_count": 0,
            "boundary_sync_count": 0,
            "full_sync_count": 0,
            "sync_bytes": 0,
            "boundary_transfer_bytes": 0,
            "boundary_syncs": (),
            "boundary_protocols": (),
            "local_simulation": False,
            "development_sharded_states": (),
            "full_mps_reconstruction_count": 0,
            "unsupported_instruction_count": 0,
        }

    autograd_state = None
    needs_autograd_replay = (
        context is not None
        and context.initialized
        and _program_requires_grad(circuit_or_ir)
    )
    if needs_autograd_replay and gradient_fastpath:
        autograd_state = run_mps(circuit_or_ir, **_mps_run_options(options))
        local = autograd_state
        mps_stats = _replicated_mps_stats(circuit_or_ir)
    else:
        local, mps_stats = _run_once(options)
    if needs_autograd_replay and autograd_state is None:
        autograd_state = run_mps(circuit_or_ir, **_mps_run_options(options))
    adaptive_initial_plan = None
    adaptive_final_plan = None
    adaptive_rerun = False
    if adaptive:
        budget = 0.0 if global_error_budget is None else float(global_error_budget)
        adaptive_initial_plan = local.adaptive_bond_plan(
            global_error_budget=budget,
            growth_factor=growth_factor,
        )
        adaptive_initial_plan = _allreduce_adaptive_plan(
            adaptive_initial_plan, context=context
        )
        adaptive_final_plan = adaptive_initial_plan
        if not adaptive_initial_plan.budget_satisfied:
            suggested = int(adaptive_initial_plan.suggested_max_bond)
            if max_bond_cap is not None:
                suggested = min(int(max_bond_cap), suggested)
            rerun_options = dict(options)
            rerun_options["max_bond"] = suggested
            local, mps_stats = _run_once(rerun_options)
            adaptive_final_plan = local.adaptive_bond_plan(
                global_error_budget=budget,
                growth_factor=growth_factor,
            )
            adaptive_final_plan = _allreduce_adaptive_plan(
                adaptive_final_plan, context=context
            )
            adaptive_rerun = True
    shards = _mps_shards(local.n_wires, world_size)
    active_rank = context.rank if context is not None else 0
    local_shard_tensors = {
        wire: local.tensors[wire]
        for shard in shards
        if shard.rank == active_rank
        for wire in shard.wires
    }
    sharded_state = ShardedMPSState(
        n_wires=local.n_wires,
        bsz=local.bsz,
        config=local.config,
        local_tensors=local_shard_tensors,
        shards=shards,
        context=context,
    )
    jax_local_world_size = (
        context.local_world_size
        if context is not None
        else (
            max(1, min(int(world_size), int(backend_policy.local_world_size)))
            if backend_policy.local_world_size > 1
            else int(world_size)
        )
    )
    jax_distributed_plan = plan_jax_distributed_quantum_backend(
        circuit_or_ir,
        mode="mps",
        world_size=world_size,
        local_world_size=jax_local_world_size,
        bsz=local.bsz,
        max_bond=options.get("max_bond"),
        distributed_backend_policy=backend_policy,
    ).summary()
    return DistributedMPSState(
        local,
        world_size=world_size,
        shards=shards,
        context=context,
        local_shard_tensors=local_shard_tensors,
        sharded_state=sharded_state,
        owned_instruction_count=mps_stats["owned_instruction_count"],
        sharded_kernel_count=mps_stats["sharded_kernel_count"],
        tensor_sync_count=mps_stats["tensor_sync_count"],
        boundary_sync_count=mps_stats["boundary_sync_count"],
        full_sync_count=mps_stats["full_sync_count"],
        sync_bytes=mps_stats["sync_bytes"],
        boundary_transfer_bytes=mps_stats["boundary_transfer_bytes"],
        boundary_syncs=mps_stats["boundary_syncs"],
        boundary_protocols=mps_stats["boundary_protocols"],
        adaptive_initial_plan=adaptive_initial_plan,
        adaptive_final_plan=adaptive_final_plan,
        adaptive_rerun=adaptive_rerun,
        autograd_state=autograd_state,
        backend_policy=backend_policy,
        local_simulation=bool(mps_stats.get("local_simulation", False)),
        development_sharded_states=mps_stats.get("development_sharded_states", ()),
        full_mps_reconstruction_count=int(
            mps_stats.get("full_mps_reconstruction_count", 0)
        ),
        unsupported_instruction_count=int(
            mps_stats.get("unsupported_instruction_count", 0)
        ),
        strict_sharded=strict_sharded,
        jax_distributed_plan=jax_distributed_plan,
    )


def _as_ir(program: Any) -> CircuitIR:
    return ensure_circuit_ir(program)


def _value_requires_grad(value: Any) -> bool:
    if isinstance(value, torch.Tensor):
        return bool(value.requires_grad)
    if isinstance(value, Mapping):
        return any(_value_requires_grad(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return any(_value_requires_grad(item) for item in value)
    return False


def _program_requires_grad(program: Any) -> bool:
    ir = _as_ir(program)
    return any(
        _value_requires_grad(instruction.params)
        or _value_requires_grad(instruction.matrix)
        for instruction in ir
    )


def _mps_run_options(options: Mapping[str, Any]) -> dict[str, Any]:
    allowed = {
        "bsz",
        "device",
        "dtype",
        "max_bond",
        "cutoff",
        "fuse_single_qubit",
        "dense_observable_wires",
    }
    return {key: value for key, value in options.items() if key in allowed}


def _replicated_mps_stats(circuit_or_ir: Any) -> dict[str, Any]:
    return {
        "owned_instruction_count": len(tuple(_as_ir(circuit_or_ir))),
        "sharded_kernel_count": 0,
        "tensor_sync_count": 0,
        "boundary_sync_count": 0,
        "full_sync_count": 0,
        "sync_bytes": 0,
        "boundary_transfer_bytes": 0,
        "boundary_syncs": (),
        "boundary_protocols": (),
        "local_simulation": False,
        "development_sharded_states": (),
        "full_mps_reconstruction_count": 0,
        "unsupported_instruction_count": 0,
    }


def _initial_mps(
    circuit_or_ir: Any,
    *,
    bsz: int = 1,
    device: torch.device | str = "cpu",
    dtype: torch.dtype | None = None,
    max_bond: int | None = None,
    cutoff: float = 0.0,
) -> MPSState:
    ir = _as_ir(circuit_or_ir)
    config = MPSConfig(max_bond=max_bond, cutoff=cutoff)
    if (
        hasattr(circuit_or_ir, "initial_state")
        and getattr(circuit_or_ir, "_inputs", None) is not None
    ):
        return MPSState.from_statevector(
            circuit_or_ir.initial_state().to(device=device), ir.n_wires, config=config
        )
    return MPSState.zero(
        ir.n_wires,
        bsz=getattr(circuit_or_ir, "bsz", bsz),
        device=device,
        dtype=getattr(circuit_or_ir, "dtype", dtype),
        config=config,
    )


def _allreduce_adaptive_plan(
    plan: MPSAdaptiveBondPlan,
    *,
    context: TorchDistributedContext | None,
) -> MPSAdaptiveBondPlan:
    if context is None or not context.initialized or context.world_size <= 1:
        return plan
    value = torch.tensor(
        [plan.observed_error], dtype=torch.float64, device=context.device
    )
    dist.all_reduce(value, op=dist.ReduceOp.MAX)
    observed = float(value.item())
    budget_satisfied = (
        observed <= float(plan.global_error_budget)
        if plan.global_error_budget is not None
        else observed == 0.0
    )
    return MPSAdaptiveBondPlan(
        current_max_bond=plan.current_max_bond,
        suggested_max_bond=plan.suggested_max_bond,
        global_error_budget=plan.global_error_budget,
        observed_error=observed,
        budget_satisfied=budget_satisfied,
        hot_bonds=plan.hot_bonds,
        per_bond_suggestions=plan.per_bond_suggestions,
    )


def _rank_for_wire(wire: int, shards: Sequence[DistributedShardPlan]) -> int:
    for shard in shards:
        if int(wire) in shard.wires:
            return shard.rank
    return 0


def _instruction_owner(
    instruction: Instruction, shards: Sequence[DistributedShardPlan]
) -> int:
    if not instruction.wires:
        return 0
    return _rank_for_wire(min(int(wire) for wire in instruction.wires), shards)


def _is_one_qubit_unitary(instruction: Instruction) -> bool:
    return len(instruction.wires) == 1 and not instruction.metadata.get("is_channel")


def _truncation_record_from_split_info(
    split_info: Mapping[str, Any],
    *,
    bond: int,
    config: MPSConfig,
) -> MPSTruncationRecord | None:
    if split_info.get("method") != "svd":
        return None
    return MPSTruncationRecord(
        bond=int(bond),
        kept_rank=int(split_info["rank"]),
        original_rank=int(split_info["original_rank"]),
        discarded_weight=float(split_info["discarded_weight"]),
        max_bond=config.max_bond,
        cutoff=config.cutoff,
        source="distributed_two_site",
    )


def _refresh_sharded_from_mps(sharded: ShardedMPSState, mps: MPSState) -> None:
    for wire in tuple(sharded.local_tensors):
        sharded.local_tensors[wire] = mps.tensors[wire]


def _instruction_is_site_local(
    instruction: Instruction, shards: Sequence[DistributedShardPlan]
) -> bool:
    wires = tuple(int(wire) for wire in instruction.wires)
    if instruction.metadata.get("is_channel"):
        return False
    if len(wires) == 1:
        return True
    if len(wires) == 2 and abs(wires[0] - wires[1]) == 1:
        return _rank_for_wire(wires[0], shards) == _rank_for_wire(wires[1], shards)
    return False


def _instruction_is_boundary_local(
    instruction: Instruction, shards: Sequence[DistributedShardPlan]
) -> bool:
    wires = tuple(int(wire) for wire in instruction.wires)
    if instruction.metadata.get("is_channel"):
        return False
    if len(wires) != 2 or abs(wires[0] - wires[1]) != 1:
        return False
    return _rank_for_wire(wires[0], shards) != _rank_for_wire(wires[1], shards)


def _boundary_sync_record(
    instruction: Instruction,
    shards: Sequence[DistributedShardPlan],
) -> DistributedBoundarySync:
    wires = tuple(sorted(int(wire) for wire in instruction.wires))
    if len(wires) != 2:
        raise ValueError("Boundary MPS sync requires a two-wire instruction.")
    left_wire, right_wire = wires
    left_rank = _rank_for_wire(left_wire, shards)
    right_rank = _rank_for_wire(right_wire, shards)
    return DistributedBoundarySync(
        left_wire=left_wire,
        right_wire=right_wire,
        left_rank=left_rank,
        right_rank=right_rank,
        owner_rank=left_rank,
    )


def _mps_tensors_nbytes(mps: MPSState, wires: Sequence[int]) -> int:
    return sum(_tensor_nbytes(mps.tensors[int(wire)]) for wire in wires)


def _boundary_protocol_record(
    mps: MPSState,
    record: DistributedBoundarySync,
    *,
    transport: str = "auto",
) -> DistributedBoundaryProtocol:
    tensor_bytes = _tensor_nbytes(mps.tensors[record.left_wire]) + _tensor_nbytes(
        mps.tensors[record.right_wire]
    )
    recipients = tuple(sorted({record.left_rank, record.right_rank}))
    normalized_transport = _resolve_boundary_transport(transport)
    if normalized_transport in {"async_p2p", "p2p"}:
        send_stage = (
            "isend_boundary_to_owner"
            if normalized_transport == "async_p2p"
            else "send_boundary_to_owner"
        )
        update_stage = (
            "isend_updated_boundary_shards"
            if normalized_transport == "async_p2p"
            else "send_updated_boundary_shards"
        )
        stages = (send_stage, "apply_two_site_update", update_stage)
        estimated_transfer_bytes = tensor_bytes * 2
        point_to_point_messages = 2
        collective_messages = 0
    else:
        stages = ("gather_boundary", "apply_two_site_update", "scatter_boundary_shards")
        estimated_transfer_bytes = tensor_bytes * max(1, len(recipients))
        point_to_point_messages = 0
        collective_messages = 2
    return DistributedBoundaryProtocol(
        sync=record,
        stages=stages,
        tensor_bytes=tensor_bytes,
        recipient_ranks=recipients,
        transport=normalized_transport,
        estimated_transfer_bytes=estimated_transfer_bytes,
        point_to_point_messages=point_to_point_messages,
        collective_messages=collective_messages,
    )


def _resolve_boundary_transport(boundary_transport: str) -> str:
    transport = str(boundary_transport).lower()
    if transport in {"auto", "async", "async_p2p", "isend_irecv"}:
        return "async_p2p"
    if transport == "p2p":
        return "p2p"
    if transport in {"broadcast", "object_broadcast"}:
        return "broadcast"
    raise ValueError(
        "boundary_transport must be 'auto', 'async_p2p', 'p2p', or 'broadcast'."
    )


def _site_local_touched_wires(instruction: Instruction) -> tuple[int, ...]:
    wires = tuple(int(wire) for wire in instruction.wires)
    if len(wires) == 1:
        return wires
    return tuple(sorted(wires))


def _boundary_touched_wires(instruction: Instruction) -> tuple[int, ...]:
    return tuple(sorted(int(wire) for wire in instruction.wires))


def _mps_payload(mps: MPSState) -> dict[str, Any]:
    return {
        "local_swap_count": mps.local_swap_count,
        "truncation_errors": tuple(mps.truncation_errors),
        "truncation_records": tuple(
            record.summary() for record in mps.truncation_records
        ),
        "orthogonality_center": mps.orthogonality_center,
    }


def _load_mps_payload(mps: MPSState, payload: Mapping[str, Any]) -> None:
    mps.local_swap_count = int(payload["local_swap_count"])
    mps.truncation_errors = [float(value) for value in payload["truncation_errors"]]
    mps.truncation_records = [
        MPSTruncationRecord(**record)
        for record in payload.get("truncation_records", ())
    ]
    mps.orthogonality_center = payload.get("orthogonality_center")


def _broadcast_mps_metadata(
    payload: Mapping[str, Any] | None,
    *,
    src: int,
    context: TorchDistributedContext,
) -> Mapping[str, Any]:
    """Broadcast MPS metadata without pickle or NumPy-backed object collectives."""
    transport_device = (
        context.device
        if backend_uses_accelerator_tensors(context.backend)
        else torch.device("cpu")
    )
    encoded = (
        json.dumps(payload, separators=(",", ":")).encode("utf-8")
        if context.rank == src
        else b""
    )
    length = torch.tensor(
        [len(encoded)],
        dtype=torch.int64,
        device=transport_device,
    )
    dist.broadcast(length, src=src)
    byte_count = int(length.item())
    if context.rank == src:
        buffer = torch.tensor(
            list(encoded),
            dtype=torch.uint8,
            device=transport_device,
        )
    else:
        buffer = torch.empty(
            byte_count,
            dtype=torch.uint8,
            device=transport_device,
        )
    dist.broadcast(buffer, src=src)
    decoded = json.loads(bytes(buffer.cpu().tolist()).decode("utf-8"))
    if not isinstance(decoded, dict):
        raise RuntimeError("Distributed MPS metadata must decode to a mapping.")
    return decoded


def _broadcast_mps(
    mps: MPSState, *, src: int, context: TorchDistributedContext
) -> None:
    payload = _broadcast_mps_metadata(
        _mps_payload(mps) if context.rank == src else None,
        src=src,
        context=context,
    )
    if context.rank != src:
        _load_mps_payload(mps, payload)
    mps.tensors = [
        _broadcast_mps_site_tensor(
            tensor if context.rank == src else None,
            src=src,
            context=context,
            wire=wire,
        ).to(device=context.device)
        for wire, tensor in enumerate(mps.tensors)
    ]


def _broadcast_mps_tensors(
    mps: MPSState,
    wires: Sequence[int],
    *,
    src: int,
    context: TorchDistributedContext,
) -> None:
    wire_tuple = tuple(dict.fromkeys(int(wire) for wire in wires))
    for wire in wire_tuple:
        mps.tensors[wire] = _broadcast_mps_site_tensor(
            mps.tensors[wire] if context.rank == src else None,
            src=src,
            context=context,
            wire=wire,
        ).to(device=context.device)


def _sync_boundary_mps_tensors(
    mps: MPSState,
    record: DistributedBoundarySync,
    *,
    instruction: Instruction,
    context: TorchDistributedContext,
    transport: str = "auto",
) -> None:
    """Synchronize a two-site MPS update across an adjacent shard boundary."""

    normalized_transport = _resolve_boundary_transport(transport)
    if normalized_transport in {"async_p2p", "p2p"} and context.world_size > 1:
        _sync_boundary_mps_tensors_p2p(
            mps,
            record,
            instruction=instruction,
            context=context,
            async_transfer=normalized_transport == "async_p2p",
        )
        return

    gathered = _gather_boundary_mps_tensors(mps, record, context=context)
    if context.rank == record.owner_rank:
        mps.tensors[record.left_wire] = gathered[record.left_wire].to(
            device=context.device
        )
        mps.tensors[record.right_wire] = gathered[record.right_wire].to(
            device=context.device
        )
        mps.apply_instruction(instruction)
    _scatter_boundary_mps_tensors(mps, record, context=context)


def _sync_boundary_mps_tensors_p2p(
    mps: MPSState,
    record: DistributedBoundarySync,
    *,
    instruction: Instruction,
    context: TorchDistributedContext,
    async_transfer: bool = False,
) -> None:
    """Synchronize an adjacent cross-shard MPS update with rank-to-rank tensor messages."""

    if context.rank == record.right_rank:
        _send = _send_tensor_async_p2p if async_transfer else _send_tensor_p2p
        _recv = _recv_tensor_async_p2p if async_transfer else _recv_tensor_p2p
        _send(mps.tensors[record.right_wire], dst=record.owner_rank)
        mps.tensors[record.right_wire] = _recv(
            src=record.owner_rank,
            reference=mps.tensors[record.right_wire],
        )
        return

    if context.rank == record.owner_rank:
        _send = _send_tensor_async_p2p if async_transfer else _send_tensor_p2p
        _recv = _recv_tensor_async_p2p if async_transfer else _recv_tensor_p2p
        if record.right_rank != record.owner_rank:
            mps.tensors[record.right_wire] = _recv(
                src=record.right_rank,
                reference=mps.tensors[record.right_wire],
            )
        mps.apply_instruction(instruction)
        if record.right_rank != record.owner_rank:
            _send(mps.tensors[record.right_wire], dst=record.right_rank)


def _gather_boundary_mps_tensors(
    mps: MPSState,
    record: DistributedBoundarySync,
    *,
    context: TorchDistributedContext,
) -> dict[int, torch.Tensor]:
    return {
        record.left_wire: _broadcast_mps_site_tensor(
            (
                mps.tensors[record.left_wire]
                if context.rank == record.left_rank
                else None
            ),
            src=record.left_rank,
            context=context,
            wire=record.left_wire,
        ),
        record.right_wire: _broadcast_mps_site_tensor(
            (
                mps.tensors[record.right_wire]
                if context.rank == record.right_rank
                else None
            ),
            src=record.right_rank,
            context=context,
            wire=record.right_wire,
        ),
    }


def _scatter_boundary_mps_tensors(
    mps: MPSState,
    record: DistributedBoundarySync,
    *,
    context: TorchDistributedContext,
) -> None:
    left = _broadcast_mps_site_tensor(
        mps.tensors[record.left_wire] if context.rank == record.owner_rank else None,
        src=record.owner_rank,
        context=context,
        wire=record.left_wire,
    )
    right = _broadcast_mps_site_tensor(
        mps.tensors[record.right_wire] if context.rank == record.owner_rank else None,
        src=record.owner_rank,
        context=context,
        wire=record.right_wire,
    )
    if context.rank == record.left_rank:
        mps.tensors[record.left_wire] = left.to(device=context.device)
    if context.rank == record.right_rank:
        mps.tensors[record.right_wire] = right.to(device=context.device)


def _sync_mps_from_sharded(
    mps: MPSState,
    sharded: ShardedMPSState,
    *,
    context: TorchDistributedContext,
) -> None:
    gathered = sharded.gather_tensors()
    missing = [wire for wire in range(mps.n_wires) if wire not in gathered]
    if missing:
        raise RuntimeError(
            f"Cannot synchronize full MPS from shards; missing wires {missing}."
        )
    mps.tensors = [
        gathered[wire].to(device=context.device) for wire in range(mps.n_wires)
    ]


def _mps_from_rank_tensors(
    rank_tensors: Mapping[int, Mapping[int, torch.Tensor]],
    *,
    n_wires: int,
    config: MPSConfig,
    truncation_errors: Sequence[float] = (),
    truncation_records: Sequence[MPSTruncationRecord] = (),
) -> MPSState:
    tensors_by_wire: dict[int, torch.Tensor] = {}
    for payload in rank_tensors.values():
        tensors_by_wire.update({int(wire): tensor for wire, tensor in payload.items()})
    missing = [wire for wire in range(int(n_wires)) if wire not in tensors_by_wire]
    if missing:
        raise RuntimeError(
            f"Cannot reconstruct MPS from rank shards; missing wires {missing}."
        )
    out = MPSState(
        [tensors_by_wire[wire] for wire in range(int(n_wires))], config=config
    )
    out.truncation_errors = [float(value) for value in truncation_errors]
    out.truncation_records = list(truncation_records)
    return out


def _rank_tensors_from_mps(
    mps: MPSState,
    shards: Sequence[DistributedShardPlan],
) -> dict[int, dict[int, torch.Tensor]]:
    return {
        shard.rank: {wire: mps.tensors[wire] for wire in shard.wires}
        for shard in shards
    }


def _development_sharded_states(
    rank_tensors: Mapping[int, Mapping[int, torch.Tensor]],
    *,
    n_wires: int,
    bsz: int,
    config: MPSConfig,
    shards: Sequence[DistributedShardPlan],
) -> tuple[ShardedMPSState, ...]:
    return tuple(
        ShardedMPSState(
            n_wires=n_wires,
            bsz=bsz,
            config=config,
            local_tensors=rank_tensors.get(shard.rank, {}),
            shards=shards,
            rank=shard.rank,
        )
        for shard in shards
    )


def _run_mps_site_sharded_local(
    circuit_or_ir: Any,
    *,
    world_size: int,
    bsz: int = 1,
    device: torch.device | str = "cpu",
    dtype: torch.dtype | None = None,
    max_bond: int | None = None,
    cutoff: float = 0.0,
    boundary_transport: str = "auto",
    strict_sharded: bool = False,
    **_: Any,
) -> tuple[MPSState, dict[str, Any]]:
    """Run a single-process MPS development simulator with rank-owned tensors."""

    ir = _as_ir(circuit_or_ir)
    seed_mps = _initial_mps(
        circuit_or_ir,
        bsz=bsz,
        device=device,
        dtype=dtype,
        max_bond=max_bond,
        cutoff=cutoff,
    )
    shards = _mps_shards(ir.n_wires, world_size)
    rank_tensors = _rank_tensors_from_mps(seed_mps, shards)
    config = seed_mps.config
    truncation_errors: list[float] = []
    owned_instruction_count = 0
    sharded_kernel_count = 0
    boundary_sync_count = 0
    full_sync_count = 0
    sync_bytes = 0
    boundary_transfer_bytes = 0
    boundary_syncs: list[DistributedBoundarySync] = []
    boundary_protocols: list[DistributedBoundaryProtocol] = []
    truncation_records: list[MPSTruncationRecord] = []
    full_mps_reconstruction_count = 0
    unsupported_instruction_count = 0
    normalized_boundary_transport = _resolve_boundary_transport(boundary_transport)

    for instruction in ir:
        owner = _instruction_owner(instruction, shards)
        site_local = _instruction_is_site_local(instruction, shards)
        boundary_local = _instruction_is_boundary_local(instruction, shards)
        if _is_one_qubit_unitary(instruction) and site_local:
            wire = int(instruction.wires[0])
            matrix = _instruction_matrix_for_mps(
                instruction,
                bsz=seed_mps.bsz,
                device=device,
                dtype=seed_mps.dtype,
            )
            rank_tensors[owner][wire] = _apply_one_mps_tensor(
                rank_tensors[owner][wire], matrix
            )
            owned_instruction_count += 1
            sharded_kernel_count += 1
            continue
        if (
            site_local
            and len(instruction.wires) == 2
            and abs(int(instruction.wires[0]) - int(instruction.wires[1])) == 1
        ):
            first, second = int(instruction.wires[0]), int(instruction.wires[1])
            left_wire = min(first, second)
            matrix = _instruction_matrix_for_mps(
                instruction,
                bsz=seed_mps.bsz,
                device=device,
                dtype=seed_mps.dtype,
            )
            left, right, split_info = _apply_two_mps_tensors_with_info(
                rank_tensors[owner][left_wire],
                rank_tensors[owner][left_wire + 1],
                matrix,
                config,
                reverse=first > second,
            )
            rank_tensors[owner][left_wire] = left
            rank_tensors[owner][left_wire + 1] = right
            step_error = float(split_info["discarded_weight"])
            if step_error > 0:
                truncation_errors.append(step_error)
            record = _truncation_record_from_split_info(
                split_info, bond=left_wire, config=config
            )
            if record is not None:
                truncation_records.append(record)
            owned_instruction_count += 1
            sharded_kernel_count += 1
            continue
        if boundary_local:
            first, second = int(instruction.wires[0]), int(instruction.wires[1])
            record = _boundary_sync_record(instruction, shards)
            tensor_bytes = _tensor_nbytes(
                rank_tensors[record.left_rank][record.left_wire]
            ) + _tensor_nbytes(rank_tensors[record.right_rank][record.right_wire])
            protocol = DistributedBoundaryProtocol(
                sync=record,
                stages=(
                    "local_sim_exchange_boundary_tensors",
                    "apply_two_site_update",
                    "local_sim_scatter_boundary_tensors",
                ),
                tensor_bytes=tensor_bytes,
                recipient_ranks=tuple(sorted({record.left_rank, record.right_rank})),
                transport=normalized_boundary_transport,
                estimated_transfer_bytes=tensor_bytes * 2,
                point_to_point_messages=2,
                collective_messages=0,
            )
            matrix = _instruction_matrix_for_mps(
                instruction,
                bsz=seed_mps.bsz,
                device=device,
                dtype=seed_mps.dtype,
            )
            left, right, split_info = _apply_two_mps_tensors_with_info(
                rank_tensors[record.left_rank][record.left_wire],
                rank_tensors[record.right_rank][record.right_wire],
                matrix,
                config,
                reverse=first > second,
            )
            rank_tensors[record.left_rank][record.left_wire] = left
            rank_tensors[record.right_rank][record.right_wire] = right
            step_error = float(split_info["discarded_weight"])
            if step_error > 0:
                truncation_errors.append(step_error)
            truncation_record = _truncation_record_from_split_info(
                split_info, bond=record.left_wire, config=config
            )
            if truncation_record is not None:
                truncation_records.append(truncation_record)
            boundary_syncs.append(record)
            boundary_protocols.append(protocol)
            boundary_sync_count += 1
            boundary_transfer_bytes += protocol.estimated_transfer_bytes
            sync_bytes += protocol.estimated_transfer_bytes
            owned_instruction_count += 1
            sharded_kernel_count += 1
            continue

        unsupported_instruction_count += 1
        if strict_sharded:
            raise RuntimeError(
                "Strict distributed MPS only supports one-qubit gates, adjacent two-qubit gates "
                "within a shard, and adjacent two-qubit gates across a shard boundary. "
                f"Unsupported instruction {instruction.name!r} on wires {tuple(instruction.wires)}."
            )
        fallback_mps = _mps_from_rank_tensors(
            rank_tensors,
            n_wires=ir.n_wires,
            config=config,
            truncation_errors=truncation_errors,
            truncation_records=truncation_records,
        )
        fallback_mps.apply_instruction(instruction)
        truncation_errors = list(fallback_mps.truncation_errors)
        truncation_records = list(fallback_mps.truncation_records)
        rank_tensors = _rank_tensors_from_mps(fallback_mps, shards)
        full_sync_count += 1
        full_mps_reconstruction_count += 1
        sync_bytes += _mps_tensors_nbytes(
            fallback_mps, range(fallback_mps.n_wires)
        ) * max(0, int(world_size) - 1)

    local = _mps_from_rank_tensors(
        rank_tensors,
        n_wires=ir.n_wires,
        config=config,
        truncation_errors=truncation_errors,
        truncation_records=truncation_records,
    )
    return local, {
        "owned_instruction_count": owned_instruction_count,
        "sharded_kernel_count": sharded_kernel_count,
        "tensor_sync_count": 0,
        "boundary_sync_count": boundary_sync_count,
        "full_sync_count": full_sync_count,
        "sync_bytes": sync_bytes,
        "boundary_transfer_bytes": boundary_transfer_bytes,
        "boundary_syncs": tuple(boundary_syncs),
        "boundary_protocols": tuple(boundary_protocols),
        "local_simulation": True,
        "development_sharded_states": _development_sharded_states(
            rank_tensors,
            n_wires=ir.n_wires,
            bsz=local.bsz,
            config=config,
            shards=shards,
        ),
        "full_mps_reconstruction_count": full_mps_reconstruction_count,
        "unsupported_instruction_count": unsupported_instruction_count,
    }


def _run_mps_site_sharded_sync(
    circuit_or_ir: Any,
    *,
    context: TorchDistributedContext,
    world_size: int,
    bsz: int = 1,
    device: torch.device | str = "cpu",
    dtype: torch.dtype | None = None,
    max_bond: int | None = None,
    cutoff: float = 0.0,
    boundary_transport: str = "auto",
    **_: Any,
) -> tuple[MPSState, dict[str, int]]:
    ir = _as_ir(circuit_or_ir)
    mps = _initial_mps(
        circuit_or_ir,
        bsz=bsz,
        device=device,
        dtype=dtype,
        max_bond=max_bond,
        cutoff=cutoff,
    )
    shards = _mps_shards(ir.n_wires, world_size)
    local_sharded = ShardedMPSState(
        n_wires=mps.n_wires,
        bsz=mps.bsz,
        config=mps.config,
        local_tensors={
            wire: mps.tensors[wire]
            for shard in shards
            if shard.rank == context.rank
            for wire in shard.wires
        },
        shards=shards,
        context=context,
    )
    owned_instruction_count = 0
    sharded_kernel_count = 0
    tensor_sync_count = 0
    boundary_sync_count = 0
    full_sync_count = 0
    sync_bytes = 0
    boundary_transfer_bytes = 0
    boundary_syncs: list[DistributedBoundarySync] = []
    boundary_protocols: list[DistributedBoundaryProtocol] = []
    normalized_boundary_transport = _resolve_boundary_transport(boundary_transport)
    for instruction in ir:
        owner = _instruction_owner(instruction, shards)
        site_local = _instruction_is_site_local(instruction, shards)
        boundary_local = _instruction_is_boundary_local(instruction, shards)
        use_sharded_one = _is_one_qubit_unitary(instruction) and context.rank == owner
        use_sharded_two = (
            site_local
            and len(instruction.wires) == 2
            and abs(int(instruction.wires[0]) - int(instruction.wires[1])) == 1
            and context.rank == owner
        )
        if not site_local and not boundary_local:
            _sync_mps_from_sharded(mps, local_sharded, context=context)
        if use_sharded_one:
            wire = int(instruction.wires[0])
            matrix = _instruction_matrix_for_mps(
                instruction,
                bsz=mps.bsz,
                device=context.device,
                dtype=mps.dtype,
            )
            local_sharded.apply_one_local(matrix, wire)
            mps.tensors[wire] = local_sharded.local_tensors[wire]
            owned_instruction_count += 1
            sharded_kernel_count += 1
        elif use_sharded_two:
            first, second = int(instruction.wires[0]), int(instruction.wires[1])
            left_wire = min(first, second)
            matrix = _instruction_matrix_for_mps(
                instruction,
                bsz=mps.bsz,
                device=context.device,
                dtype=mps.dtype,
            )
            step_error = local_sharded.apply_two_local(
                matrix, left_wire, reverse=first > second
            )
            mps.tensors[left_wire] = local_sharded.local_tensors[left_wire]
            mps.tensors[left_wire + 1] = local_sharded.local_tensors[left_wire + 1]
            if step_error > 0:
                mps.truncation_errors.append(step_error)
            owned_instruction_count += 1
            sharded_kernel_count += 1
        elif context.rank == owner and not boundary_local:
            mps.apply_instruction(instruction)
            owned_instruction_count += 1
        if site_local:
            touched_wires = _site_local_touched_wires(instruction)
            _broadcast_mps_tensors(
                mps,
                touched_wires,
                src=owner,
                context=context,
            )
            _refresh_sharded_from_mps(local_sharded, mps)
            tensor_sync_count += 1
            sync_bytes += _mps_tensors_nbytes(mps, touched_wires) * max(
                0, context.world_size - 1
            )
        elif boundary_local:
            record = _boundary_sync_record(instruction, shards)
            protocol = _boundary_protocol_record(
                mps, record, transport=normalized_boundary_transport
            )
            boundary_syncs.append(record)
            boundary_protocols.append(protocol)
            _sync_boundary_mps_tensors(
                mps,
                record,
                instruction=instruction,
                context=context,
                transport=normalized_boundary_transport,
            )
            _refresh_sharded_from_mps(local_sharded, mps)
            if context.rank == record.owner_rank:
                owned_instruction_count += 1
            boundary_sync_count += 1
            boundary_transfer_bytes += protocol.estimated_transfer_bytes
            sync_bytes += protocol.estimated_transfer_bytes
        else:
            _broadcast_mps(mps, src=owner, context=context)
            _refresh_sharded_from_mps(local_sharded, mps)
            full_sync_count += 1
            sync_bytes += _mps_tensors_nbytes(mps, range(mps.n_wires)) * max(
                0, context.world_size - 1
            )
    return mps, {
        "owned_instruction_count": owned_instruction_count,
        "sharded_kernel_count": sharded_kernel_count,
        "tensor_sync_count": tensor_sync_count,
        "boundary_sync_count": boundary_sync_count,
        "full_sync_count": full_sync_count,
        "sync_bytes": sync_bytes,
        "boundary_transfer_bytes": boundary_transfer_bytes,
        "boundary_syncs": tuple(boundary_syncs),
        "boundary_protocols": tuple(boundary_protocols),
    }


__all__ = [
    "DistributedIdentity",
    "DistributedIdentityError",
    "DistributedBoundaryProtocol",
    "DistributedBoundarySync",
    "DistributedMPSState",
    "DistributedShardPlan",
    "ShardedMPSState",
    "TorchDistributedContext",
    "destroy_torch_distributed",
    "init_torch_distributed",
    "require_verified_flagcx",
    "run_distributed_mps",
    "torch_distributed_is_available",
]


def __getattr__(name: str) -> Any:
    """Preserve private transport probes after distributed decomposition."""
    if name.startswith("__"):
        raise AttributeError(name)
    module = import_module("flagquantum.runtime.distributed.mps_transport")
    try:
        return getattr(module, name)
    except AttributeError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

"""Multi-step owner-sharded training over the native distributed MPS runtime."""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from contextlib import AbstractContextManager, nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Literal, Mapping, Sequence

import torch
import torch.distributed as dist
from torch.profiler import record_function

from ....compute import get_platform_runtime
from ....core.ir import CircuitIR, ensure_circuit_ir
from . import checkpointing as _checkpointing
from .device_resolution import resolve_distributed_mps_device
from .errors import MPSTrainingError
from .metadata_transport import all_gather_json
from .records import MPSReverseCheckpointPolicy
from .reverse import execute_torch_distributed_mps_reverse
from .reverse_planning import build_mps_parameter_layout
from .state import (
    initial_mps_ownership,
    topology_aware_mps_ownership,
    validate_mps_ownership,
)
from .training_records import (
    MPSOptimizerOwnership,
    MPSStepMetrics,
    ShardedMPSTrainingResult,
)
from .transport import warmup_mps_neighbor_communicators

_parameter_layout = build_mps_parameter_layout


def _nvtx_phase(name: str, device: torch.device) -> AbstractContextManager[None]:
    return torch.cuda.nvtx.range(name) if device.type == "cuda" else nullcontext()


def _initial_state_contract(
    ir: Any,
    initial_mps_tensors: Mapping[int, torch.Tensor] | None,
    initial_bond_dimension: int,
    site_ownership: tuple[tuple[int, ...], ...],
) -> tuple[str, dict[int, tuple[int, ...]]]:
    digest_cache: dict[tuple[int, int], str] = {}

    def tensor_digest(tensor: torch.Tensor) -> str:
        cache_key = (id(tensor), int(tensor._version))
        cached = digest_cache.get(cache_key)
        if cached is not None:
            return cached
        digest = hashlib.sha256()
        value = tensor.detach().contiguous()
        # PyTorch 2.13 rejects a direct complex-to-byte view for some otherwise
        # contiguous MPS layouts because the complex element stride is not a
        # byte stride.  Expose real/imag channels first so checkpoint identity
        # remains a byte-exact digest for both real and complex tensors.
        if value.is_complex():
            value = torch.view_as_real(value)
        raw = value.contiguous().view(torch.uint8).reshape(-1)
        chunk_bytes = 8 * 1024 * 1024
        for start in range(0, raw.numel(), chunk_bytes):
            chunk = raw[start : start + chunk_bytes].cpu().numpy()
            digest.update(chunk.tobytes())
        result = digest.hexdigest()
        digest_cache[cache_key] = result
        return result

    if initial_mps_tensors is None:
        local_shapes: dict[int, tuple[int, ...]] = {
            wire: (
                int(ir.metadata.get("batch_size", 1)),
                1 if wire == 0 else initial_bond_dimension,
                2,
                1 if wire == ir.n_wires - 1 else initial_bond_dimension,
            )
            for wire in range(ir.n_wires)
            if wire in site_ownership[dist.get_rank()]
        }
        local_descriptors = {
            wire: {
                "shape": shape,
                "initializer": "flagquantum_rank_owned_mps_v1",
            }
            for wire, shape in local_shapes.items()
        }
    else:
        local_shapes = {
            int(wire): tuple(int(value) for value in tensor.shape)
            for wire, tensor in initial_mps_tensors.items()
        }
        local_descriptors = {
            int(wire): {
                "shape": local_shapes[int(wire)],
                "dtype": str(tensor.dtype),
                "sha256": tensor_digest(tensor),
            }
            for wire, tensor in initial_mps_tensors.items()
        }
    gathered = all_gather_json(local_descriptors)
    global_descriptors = {
        int(wire): descriptor
        for payload in gathered
        for wire, descriptor in payload.items()
    }
    content = json.dumps(global_descriptors, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(content.encode()).hexdigest(), local_shapes


def _contract(
    ir: Any,
    optimizer: str,
    lr: float,
    lr_decay: float,
    max_bond: int | None,
    cutoff: float,
    *,
    world_size: int,
    initial_state_fingerprint: str,
    gradient_policy: str,
    gradient_tolerance: float,
    initial_mps_left_canonical: bool,
    reverse_checkpoint_policy: MPSReverseCheckpointPolicy,
    gradient_bucket_bytes: int,
    compile_site_kernels: bool,
    prefetch_layer_halos: bool,
    compile_observables: bool,
    fuse_local_reverse: bool,
    canonicalization_policy: str,
    site_ownership: tuple[tuple[int, ...], ...],
) -> dict[str, Any]:
    return {
        "schema": "sharded_mps_training_checkpoint_v1",
        "ir_hash": ir.content_hash,
        "optimizer": optimizer,
        "learning_rate": float(lr),
        "learning_rate_decay": float(lr_decay),
        "max_bond": max_bond,
        "cutoff": float(cutoff),
        "world_size": world_size,
        "initial_state_fingerprint": initial_state_fingerprint,
        "gradient_policy": gradient_policy,
        "gradient_tolerance": float(gradient_tolerance),
        "initial_mps_left_canonical": bool(initial_mps_left_canonical),
        "reverse_checkpoint_policy": asdict(reverse_checkpoint_policy),
        "gradient_bucket_bytes": int(gradient_bucket_bytes),
        "gradient_route": "bounded_dtype_owner_reduce_v1",
        "compile_site_kernels": bool(compile_site_kernels),
        "prefetch_layer_halos": bool(prefetch_layer_halos),
        "compile_observables": bool(compile_observables),
        "fuse_local_reverse": bool(fuse_local_reverse),
        "canonicalization_policy": canonicalization_policy,
        "site_ownership": site_ownership,
    }


def _resolve_compile_site_kernels(
    requested: bool | None,
    *,
    ir: Any,
    device: torch.device,
    steps: int,
    checkpointing: bool = False,
) -> tuple[bool, str]:
    """Choose compiled MPS site buckets without changing unsupported circuits."""
    if requested is not None:
        return bool(requested), "explicit_compiled" if requested else "explicit_eager"
    if checkpointing:
        return False, "auto_checkpoint_compatibility_uses_eager"
    if device.type != "cuda":
        return False, "auto_cpu_eager_fast_path"
    if steps < 3:
        return False, "auto_short_training_avoids_compile_warmup"
    reversed_rotations = any(
        instruction.name in {"rxx", "ryy", "rzz"}
        and tuple(map(int, instruction.wires))
        != tuple(sorted(map(int, instruction.wires)))
        for instruction in ir.instructions
    )
    if reversed_rotations:
        return False, "auto_incompatible_wire_order_uses_eager"
    return True, "auto_cuda_repeated_training"


@dataclass(frozen=True)
class _ParameterAllGatherBucket:
    indices_by_owner: tuple[tuple[int, ...], ...]
    input_buffer: torch.Tensor
    output_buffer: torch.Tensor
    pack_targets: tuple[torch.Tensor, ...]
    pack_sources: tuple[torch.Tensor, ...]
    unpack_targets: tuple[torch.Tensor, ...]
    unpack_sources: tuple[torch.Tensor, ...]
    payload_bytes: int


def _allocate_parameter_all_gather_bucket(
    parameters: tuple[torch.Tensor, ...],
    indices_by_owner: tuple[tuple[int, ...], ...],
    *,
    rank: int,
    dtype: torch.dtype,
    device: torch.device,
    element_size: int,
) -> _ParameterAllGatherBucket:
    owner_elements = [
        sum(parameters[index].numel() for index in indices)
        for indices in indices_by_owner
    ]
    stride = max(owner_elements)
    input_buffer = torch.empty(stride, dtype=dtype, device=device)
    output_buffer = torch.empty(
        len(indices_by_owner) * stride, dtype=dtype, device=device
    )
    pack_targets = []
    pack_sources = []
    offset = 0
    for index in indices_by_owner[rank]:
        parameter = parameters[index]
        count = parameter.numel()
        pack_targets.append(input_buffer[offset : offset + count].reshape_as(parameter))
        pack_sources.append(parameter)
        offset += count
    unpack_targets = []
    unpack_sources = []
    for owner, owner_indices in enumerate(indices_by_owner):
        if rank == owner:
            continue
        offset = 0
        for index in owner_indices:
            parameter = parameters[index]
            count = parameter.numel()
            start = owner * stride + offset
            unpack_targets.append(parameter)
            unpack_sources.append(
                output_buffer[start : start + count].reshape_as(parameter)
            )
            offset += count
    return _ParameterAllGatherBucket(
        indices_by_owner=indices_by_owner,
        input_buffer=input_buffer,
        output_buffer=output_buffer,
        pack_targets=tuple(pack_targets),
        pack_sources=tuple(pack_sources),
        unpack_targets=tuple(unpack_targets),
        unpack_sources=tuple(unpack_sources),
        payload_bytes=sum(owner_elements) * element_size,
    )


def _chunk_parameter_indices_by_owner(
    indices_by_owner: list[list[int]],
    parameters: tuple[torch.Tensor, ...],
    *,
    element_size: int,
    max_bucket_bytes: int,
) -> list[list[tuple[int, ...]]]:
    owner_chunks: list[list[tuple[int, ...]]] = []
    for indices in indices_by_owner:
        chunks: list[tuple[int, ...]] = []
        current: list[int] = []
        current_bytes = 0
        for index in indices:
            parameter_bytes = parameters[index].numel() * element_size
            if current and current_bytes + parameter_bytes > max_bucket_bytes:
                chunks.append(tuple(current))
                current = []
                current_bytes = 0
            current.append(index)
            current_bytes += parameter_bytes
        if current:
            chunks.append(tuple(current))
        owner_chunks.append(chunks)
    return owner_chunks


def _parameter_broadcast_buckets(
    parameters: tuple[torch.Tensor, ...],
    owners: tuple[int, ...],
    *,
    max_bucket_bytes: int,
    world_size: int | None = None,
    rank: int = 0,
) -> tuple[_ParameterAllGatherBucket, ...]:
    """Allocate reusable dtype buckets for one-collective owner synchronization."""

    if len(parameters) != len(owners):
        raise ValueError("parameters and owners must have equal length")
    world_size = world_size or (max(owners, default=-1) + 1)
    if any(owner < 0 or owner >= world_size for owner in owners):
        raise ValueError("parameter owner is outside world_size")
    if rank < 0 or rank >= world_size:
        raise ValueError("rank is outside world_size")
    grouped: dict[tuple[torch.dtype, torch.device], list[list[int]]] = {}
    for index, (parameter, owner) in enumerate(zip(parameters, owners)):
        key = (parameter.dtype, parameter.device)
        if key not in grouped:
            grouped[key] = [[] for _ in range(world_size)]
        grouped[key][owner].append(index)
    buckets: list[_ParameterAllGatherBucket] = []
    for (dtype, device), indices_by_owner in grouped.items():
        element_size = torch.empty((), dtype=dtype).element_size()
        owner_chunks = _chunk_parameter_indices_by_owner(
            indices_by_owner,
            parameters,
            element_size=element_size,
            max_bucket_bytes=max_bucket_bytes,
        )
        for round_index in range(max(map(len, owner_chunks), default=0)):
            round_indices = tuple(
                chunks[round_index] if round_index < len(chunks) else ()
                for chunks in owner_chunks
            )
            buckets.append(
                _allocate_parameter_all_gather_bucket(
                    parameters,
                    round_indices,
                    rank=rank,
                    dtype=dtype,
                    device=device,
                    element_size=element_size,
                )
            )
    return tuple(buckets)


def _broadcast_parameters(
    parameters: tuple[torch.Tensor, ...],
    buckets: tuple[_ParameterAllGatherBucket, ...],
) -> tuple[int, int]:
    """Synchronize owner shards with bounded, reusable flat all-gathers."""

    byte_count = 0
    with torch.no_grad():
        for bucket in buckets:
            buffer = bucket.input_buffer
            buffer.zero_()
            if bucket.pack_targets:
                torch._foreach_copy_(bucket.pack_targets, bucket.pack_sources)
            dist.all_gather_single(bucket.output_buffer, buffer)
            byte_count += bucket.payload_bytes
            if bucket.unpack_targets:
                torch._foreach_copy_(bucket.unpack_targets, bucket.unpack_sources)
    return len(buckets), byte_count


def _distributed_dot(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    value = torch.dot(left, right)
    dist.all_reduce(value, op=dist.ReduceOp.SUM)
    return value


def _parameter_gradient_values(
    parameters: Sequence[torch.Tensor], owned_indices: Sequence[int]
) -> tuple[tuple[int, float], ...]:
    """Snapshot owner gradients for diagnostics, rejecting missing values."""
    values = []
    for index in owned_indices:
        gradient = parameters[index].grad
        if gradient is None:
            raise MPSTrainingError(
                f"cannot record gradient for owned parameter {index}: gradient is missing"
            )
        values.append((index, float(gradient.detach().cpu())))
    return tuple(values)


def _owner_flatten(tensors: Sequence[torch.Tensor], *, gradients: bool) -> torch.Tensor:
    values = []
    for tensor in tensors:
        value = tensor.grad if gradients else tensor.detach()
        if value is None:
            value = torch.zeros_like(tensor)
        values.append(value.reshape(-1))
    if values:
        return torch.cat(values)
    return torch.empty(0, device=tensors[0].device if tensors else "cpu")


def _owner_add_(parameters: Sequence[torch.Tensor], update: torch.Tensor) -> None:
    offset = 0
    with torch.no_grad():
        for parameter in parameters:
            count = parameter.numel()
            parameter.add_(update[offset : offset + count].reshape_as(parameter))
            offset += count


def _sharded_lbfgs_direction(
    gradient: torch.Tensor,
    history: Sequence[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
) -> torch.Tensor:
    """Apply distributed L-BFGS two-loop recursion to owner-local shards."""

    q = gradient.clone()
    alphas: list[torch.Tensor] = []
    for step_delta, gradient_delta, inverse_curvature in reversed(history):
        alpha = inverse_curvature * _distributed_dot(step_delta, q)
        alphas.append(alpha)
        q = q - alpha * gradient_delta
    if history:
        step_delta, gradient_delta, _ = history[-1]
        yy = _distributed_dot(gradient_delta, gradient_delta).clamp_min(1e-30)
        scale = _distributed_dot(step_delta, gradient_delta) / yy
        result = scale * q
    else:
        result = q
    for (step_delta, gradient_delta, inverse_curvature), alpha in zip(
        history, reversed(alphas)
    ):
        beta = inverse_curvature * _distributed_dot(gradient_delta, result)
        result = result + step_delta * (alpha - beta)
    return -result


def _validate_optimizer_options(
    *,
    steps: int,
    optimizer: str,
    lr: float,
    lr_decay: float,
    lbfgs_start_step: int | None,
    lbfgs_lr: float,
    lbfgs_history_size: int,
) -> int:
    if steps <= 0 or lr <= 0:
        raise ValueError("steps and lr must be positive")
    if not math.isfinite(lr_decay) or not 0.0 < lr_decay <= 1.0:
        raise ValueError("lr_decay must be finite and in (0, 1]")
    if optimizer not in {"sgd", "adam", "adam_lbfgs"}:
        raise ValueError("optimizer must be 'sgd', 'adam', or 'adam_lbfgs'")
    transition_step = steps
    if optimizer == "adam_lbfgs":
        if lbfgs_start_step is None or not 0 < lbfgs_start_step < steps:
            raise ValueError("adam_lbfgs requires 0 < lbfgs_start_step < steps")
        transition_step = int(lbfgs_start_step)
    if lbfgs_lr <= 0 or lbfgs_history_size <= 0:
        raise ValueError("lbfgs_lr and lbfgs_history_size must be positive")
    return transition_step


def _validate_checkpoint_training_options(
    *,
    optimizer: str,
    checkpoint_dir: str | Path | None,
    checkpoint_interval: int,
    checkpoint_retention_generations: int | None,
    checkpoint_free_space_reserve_bytes: int,
    checkpoint_writer_lease_stale_seconds: float,
    resume: bool,
) -> None:
    if checkpoint_interval <= 0:
        raise ValueError("checkpoint_interval must be positive")
    if (
        checkpoint_retention_generations is not None
        and checkpoint_retention_generations <= 0
    ):
        raise ValueError("checkpoint_retention_generations must be positive or None")
    if checkpoint_free_space_reserve_bytes < 0:
        raise ValueError("checkpoint_free_space_reserve_bytes must be non-negative")
    if checkpoint_writer_lease_stale_seconds <= 0:
        raise ValueError("checkpoint_writer_lease_stale_seconds must be positive")
    if optimizer == "adam_lbfgs" and (checkpoint_dir is not None or resume):
        raise MPSTrainingError(
            "adam_lbfgs checkpoint/resume requires versioned L-BFGS history support"
        )


def _validate_training_runtime_options(
    *,
    memory_leak_tolerance_bytes: int,
    memory_warmup_steps: int,
    gradient_bucket_bytes: int,
    canonicalization_policy: str,
    site_ownership_policy: str,
) -> None:
    if memory_leak_tolerance_bytes < 0:
        raise ValueError("memory_leak_tolerance_bytes must be non-negative")
    if memory_warmup_steps < 0:
        raise ValueError("memory_warmup_steps must be non-negative")
    if gradient_bucket_bytes <= 0:
        raise ValueError("gradient_bucket_bytes must be positive")
    if canonicalization_policy not in {"none", "dirty", "full"}:
        raise ValueError("canonicalization_policy must be none, dirty or full")
    if site_ownership_policy not in {"balanced", "topology_aware"}:
        raise ValueError("site_ownership_policy must be balanced or topology_aware")


def _resolve_training_site_ownership(
    ir: Any,
    *,
    world_size: int,
    local_world_size: int,
    site_ownership: Sequence[Sequence[int]] | None,
    site_ownership_policy: str,
    initial_bond_dimension: int,
    inter_node_cut_multiplier: int,
    ownership_maximum_load_ratio: float,
) -> tuple[tuple[tuple[int, ...], ...], str]:
    if site_ownership is not None:
        return (
            validate_mps_ownership(site_ownership, ir.n_wires, world_size),
            "explicit",
        )
    if site_ownership_policy != "topology_aware" or world_size == 1:
        return initial_mps_ownership(ir.n_wires, world_size), "balanced"
    boundary_penalties = [1] * max(0, ir.n_wires - 1)
    for instruction in ir.instructions:
        wires = tuple(int(wire) for wire in instruction.wires)
        if len(wires) == 2 and abs(wires[0] - wires[1]) == 1:
            boundary_penalties[min(wires)] += 1
    bonds = (
        1,
        *([int(initial_bond_dimension)] * max(0, ir.n_wires - 1)),
        1,
    )
    ownership = topology_aware_mps_ownership(
        bonds,
        world_size,
        local_world_size=local_world_size,
        boundary_penalties=boundary_penalties,
        inter_node_multiplier=inter_node_cut_multiplier,
        maximum_load_ratio=ownership_maximum_load_ratio,
    )
    return ownership, "topology_aware"


def _resolve_training_device(
    device: torch.device | str | None,
) -> tuple[torch.device, Any]:
    backend = str(dist.get_backend()).strip().lower()
    if device is None and backend == "nccl":
        device = torch.device("cuda", torch.cuda.current_device())
    resolved = resolve_distributed_mps_device(backend, device)
    return resolved, get_platform_runtime(resolved.type)


def _create_owned_optimizer(
    parameters: Sequence[torch.Tensor],
    *,
    optimizer: str,
    lr: float,
) -> torch.optim.Optimizer | None:
    if not parameters:
        return None
    if optimizer == "sgd":
        return torch.optim.SGD(parameters, lr=lr)
    return torch.optim.Adam(parameters, lr=lr)


def _step_circuit_ir(
    circuit_factory: Callable[[], Any] | None,
    base_ir: CircuitIR,
    parameters: Sequence[torch.Tensor],
) -> CircuitIR:
    if circuit_factory is None:
        return base_ir
    step_ir = ensure_circuit_ir(circuit_factory())
    step_parameters, _ = _parameter_layout(step_ir)
    if tuple(map(id, step_parameters)) != tuple(map(id, parameters)):
        raise MPSTrainingError("circuit_factory must reuse trainable leaf tensors")
    return step_ir


def _owned_reverse_work(
    records: Sequence[Any], rank: int
) -> tuple[int, int, int, int, float, int, int]:
    owned_site = 0
    owned_bond = 0
    two_site_splits = 0
    truncated_splits = 0
    discarded_weight = 0.0
    qr_activity = 0
    largest_realized_bond = 1
    for record in records:
        if record.compute_owner != rank:
            continue
        if record.kind == "one_site":
            owned_site += 1
        else:
            owned_bond += 1
        if record.kind == "two_site":
            two_site_splits += 1
            discarded_weight += record.discarded_weight
            if record.kept_rank is not None:
                largest_realized_bond = max(
                    largest_realized_bond, int(record.kept_rank)
                )
                if (
                    record.original_rank is not None
                    and record.kept_rank < record.original_rank
                ):
                    truncated_splits += 1
        if record.kind.startswith("canonicalize"):
            qr_activity += 1
    return (
        owned_site,
        owned_bond,
        two_site_splits,
        truncated_splits,
        discarded_weight,
        qr_activity,
        largest_realized_bond,
    )


def _boundary_reverse_work(
    records: Sequence[Any], rank: int, element_size: int
) -> tuple[int, int, tuple[dict[str, Any], ...]]:
    boundaries = 0
    boundary_bytes = 0
    bond_updates = []
    for record in records:
        if record.communication_peer is None or rank not in record.owner_ranks:
            continue
        boundaries += 1
        record_bytes = sum(
            math.prod(shape) * element_size
            for shape in record.input_shapes + record.output_shapes
        )
        boundary_bytes += record_bytes
        if record.kind != "two_site":
            continue
        bond_updates.append(
            {
                "operation_id": record.operation_id,
                "bond": min(record.wires),
                "owner_ranks": record.owner_ranks,
                "compute_owner": record.compute_owner,
                "original_rank": record.original_rank,
                "kept_rank": record.kept_rank,
                "discarded_weight": record.discarded_weight,
                "communication_peer": record.communication_peer,
                "communication_sequence": record.communication_sequence,
                "input_shapes": record.input_shapes,
                "output_shapes": record.output_shapes,
                "payload_bytes": record_bytes,
                "forward_transport": "batched_isend_irecv",
                "reverse_transport": "batched_isend_irecv",
            }
        )
    return boundaries, boundary_bytes, tuple(bond_updates)


def train_distributed_mps(
    circuit_or_ir: Any | Callable[[], Any],
    *,
    steps: int,
    observable: Mapping[int, str] | None = None,
    observable_terms: (
        Sequence[tuple[Mapping[int, str], torch.Tensor | float]] | None
    ) = None,
    hamiltonian_terms: (
        Sequence[tuple[Mapping[int, str], torch.Tensor | float]] | None
    ) = None,
    optimizer: Literal["sgd", "adam", "adam_lbfgs"] = "sgd",
    lr: float = 0.05,
    lr_decay: float = 1.0,
    lbfgs_start_step: int | None = None,
    lbfgs_lr: float = 0.8,
    lbfgs_history_size: int = 10,
    device: torch.device | str | None = None,
    max_bond: int | None = None,
    cutoff: float = 0.0,
    gradient_policy: Literal["exact", "approximate"] = "exact",
    gradient_tolerance: float = 0.0,
    checkpoint_dir: str | Path | None = None,
    checkpoint_interval: int = 1,
    checkpoint_retention_generations: int | None = 2,
    checkpoint_free_space_reserve_bytes: int = 1 << 30,
    allow_checkpoint_overwrite: bool = False,
    allow_checkpoint_writer_lease_break: bool = False,
    checkpoint_writer_lease_stale_seconds: float = 3600.0,
    resume: bool = False,
    memory_leak_tolerance_bytes: int = 0,
    initial_bond_dimension: int = 1,
    initial_mps_tensors: Mapping[int, torch.Tensor] | None = None,
    initial_mps_left_canonical: bool = False,
    site_ownership: Sequence[Sequence[int]] | None = None,
    site_ownership_policy: Literal["balanced", "topology_aware"] = "balanced",
    inter_node_cut_multiplier: int = 8,
    ownership_maximum_load_ratio: float = 1.25,
    reverse_checkpoint_policy: MPSReverseCheckpointPolicy | None = None,
    memory_warmup_steps: int = 1,
    gradient_bucket_bytes: int = 25 * 1024 * 1024,
    compile_site_kernels: bool | None = None,
    prefetch_layer_halos: bool = True,
    compile_observables: bool = False,
    fuse_local_reverse: bool = True,
    canonicalization_policy: Literal["none", "dirty", "full"] = "dirty",
    svd_driver: str | None = "gesvd",
    record_parameter_gradients: bool = False,
) -> ShardedMPSTrainingResult:
    """Train one rank-sharded MPS with native PyTorch SGD or Adam.

    ``hamiltonian_terms`` is a linear energy objective.  Its optimized path
    currently accepts on-site Z fields and adjacent XX, YY, or ZZ couplings and
    contracts them as one five-channel Heisenberg MPO.  ``observable_terms``
    retains its existing multi-target MSE semantics.

    The caller owns process-group initialization and cleanup. This keeps the
    function composable under torchrun and makes failures visible to the
    launcher rather than silently replacing distributed execution.
    """

    if not dist.is_initialized():
        raise MPSTrainingError("train_distributed_mps requires torch.distributed")
    lbfgs_transition_step = _validate_optimizer_options(
        steps=steps,
        optimizer=optimizer,
        lr=lr,
        lr_decay=lr_decay,
        lbfgs_start_step=lbfgs_start_step,
        lbfgs_lr=lbfgs_lr,
        lbfgs_history_size=lbfgs_history_size,
    )
    _validate_checkpoint_training_options(
        optimizer=optimizer,
        checkpoint_dir=checkpoint_dir,
        checkpoint_interval=checkpoint_interval,
        checkpoint_retention_generations=checkpoint_retention_generations,
        checkpoint_free_space_reserve_bytes=checkpoint_free_space_reserve_bytes,
        checkpoint_writer_lease_stale_seconds=(checkpoint_writer_lease_stale_seconds),
        resume=resume,
    )
    _validate_training_runtime_options(
        memory_leak_tolerance_bytes=memory_leak_tolerance_bytes,
        memory_warmup_steps=memory_warmup_steps,
        gradient_bucket_bytes=gradient_bucket_bytes,
        canonicalization_policy=canonicalization_policy,
        site_ownership_policy=site_ownership_policy,
    )
    circuit_factory = circuit_or_ir if callable(circuit_or_ir) else None
    ir = ensure_circuit_ir(
        circuit_factory() if circuit_factory is not None else circuit_or_ir
    )
    if circuit_factory is not None and (checkpoint_dir is not None or resume):
        raise MPSTrainingError(
            "circuit_factory training does not yet support checkpoint/resume"
        )
    rank, world_size = dist.get_rank(), dist.get_world_size()
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", world_size))
    resolved_site_ownership, resolved_site_ownership_policy = (
        _resolve_training_site_ownership(
            ir,
            world_size=world_size,
            local_world_size=local_world_size,
            site_ownership=site_ownership,
            site_ownership_policy=site_ownership_policy,
            initial_bond_dimension=initial_bond_dimension,
            inter_node_cut_multiplier=inter_node_cut_multiplier,
            ownership_maximum_load_ratio=ownership_maximum_load_ratio,
        )
    )
    resolved_device, platform = _resolve_training_device(device)
    resolved_compile_site_kernels, site_kernel_selection_reason = (
        _resolve_compile_site_kernels(
            compile_site_kernels,
            ir=ir,
            device=resolved_device,
            steps=steps,
            checkpointing=checkpoint_dir is not None or resume,
        )
    )

    parameters, _ = _parameter_layout(ir)
    owners = tuple(index % world_size for index in range(len(parameters)))
    owned_indices = tuple(index for index, owner in enumerate(owners) if owner == rank)
    owned = [parameters[index] for index in owned_indices]
    parameter_broadcast_buckets = _parameter_broadcast_buckets(
        parameters,
        owners,
        max_bucket_bytes=gradient_bucket_bytes,
        world_size=world_size,
        rank=rank,
    )
    optimizer_obj = _create_owned_optimizer(owned, optimizer=optimizer, lr=lr)
    initial_state_fingerprint, local_bond_layout = _initial_state_contract(
        ir, initial_mps_tensors, initial_bond_dimension, resolved_site_ownership
    )
    selected_reverse_policy = reverse_checkpoint_policy or MPSReverseCheckpointPolicy()
    contract = _contract(
        ir,
        optimizer,
        lr,
        lr_decay,
        max_bond,
        cutoff,
        world_size=world_size,
        initial_state_fingerprint=initial_state_fingerprint,
        gradient_policy=gradient_policy,
        gradient_tolerance=gradient_tolerance,
        initial_mps_left_canonical=initial_mps_left_canonical,
        reverse_checkpoint_policy=selected_reverse_policy,
        gradient_bucket_bytes=gradient_bucket_bytes,
        compile_site_kernels=resolved_compile_site_kernels,
        prefetch_layer_halos=prefetch_layer_halos,
        compile_observables=compile_observables,
        fuse_local_reverse=fuse_local_reverse,
        canonicalization_policy=canonicalization_policy,
        site_ownership=resolved_site_ownership,
    )
    contract_fingerprint = hashlib.sha256(
        json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    communication_setup_seconds = warmup_mps_neighbor_communicators(resolved_device)
    root, checkpoint_storage_probe_seconds = _checkpointing._start_checkpoint_session(
        checkpoint_dir,
        rank=rank,
        world_size=world_size,
        contract_fingerprint=contract_fingerprint,
        resume=resume,
        allow_overwrite=allow_checkpoint_overwrite,
        allow_lease_break=allow_checkpoint_writer_lease_break,
        lease_stale_seconds=checkpoint_writer_lease_stale_seconds,
    )
    start_step = 0
    if resume:
        if root is None:
            raise ValueError("resume requires checkpoint_dir")
        start_step = _checkpointing._restore_training_checkpoint(
            root,
            rank=rank,
            world_size=world_size,
            steps=steps,
            device=resolved_device,
            contract_fingerprint=contract_fingerprint,
            contract=contract,
            bond_layout=local_bond_layout,
            owned_indices=owned_indices,
            parameters=parameters,
            optimizer=optimizer_obj,
        )
        _broadcast_parameters(parameters, parameter_broadcast_buckets)

    if resolved_device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(resolved_device)
    losses: list[float] = []
    metrics: list[MPSStepMetrics] = []
    checkpoints: list[str] = []
    pruned_checkpoint_files: list[str] = []
    checkpoint_write_seconds = 0.0
    checkpoint_commit_seconds = 0.0
    checkpoint_prune_seconds = 0.0
    checkpoint_bytes_written = 0
    checkpoint_min_free_bytes_observed = 0
    checkpoint_estimated_generation_bytes = 0
    checkpoint_writer_lease_heartbeat_count = 0
    checkpoint_writer_lease_heartbeat_seconds = 0.0
    memories: list[int] = []
    lbfgs_history: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    previous_lbfgs_state: tuple[torch.Tensor, torch.Tensor] | None = None
    for step in range(start_step, steps):
        if root is not None:
            heartbeat_started = time.perf_counter()
            _checkpointing._refresh_checkpoint_writer_lease(
                root,
                rank=rank,
                contract_fingerprint=contract_fingerprint,
            )
            checkpoint_writer_lease_heartbeat_seconds += (
                time.perf_counter() - heartbeat_started
            )
            checkpoint_writer_lease_heartbeat_count += 1
        if resolved_device.type == "cuda":
            platform.synchronize(resolved_device)
            torch.cuda.reset_peak_memory_stats(resolved_device)
        step_started = time.perf_counter()
        forward_started = step_started
        step_ir = _step_circuit_ir(circuit_factory, ir, parameters)
        with (
            record_function("flagquantum::mps::forward"),
            _nvtx_phase("flagquantum::mps::forward", resolved_device),
        ):
            reverse = execute_torch_distributed_mps_reverse(
                step_ir,
                observable=observable,
                observable_terms=observable_terms,
                hamiltonian_terms=hamiltonian_terms,
                device=resolved_device,
                max_bond=max_bond,
                cutoff=cutoff,
                gradient_policy=gradient_policy,
                gradient_tolerance=gradient_tolerance,
                checkpoint_policy=selected_reverse_policy,
                initial_bond_dimension=initial_bond_dimension,
                initial_mps_tensors=initial_mps_tensors,
                initial_mps_left_canonical=initial_mps_left_canonical,
                site_ownership=resolved_site_ownership,
                gradient_owner_ranks=owners,
                gradient_bucket_bytes=gradient_bucket_bytes,
                compile_site_kernels=resolved_compile_site_kernels,
                prefetch_layer_halos=prefetch_layer_halos,
                compile_observables=compile_observables,
                fuse_local_reverse=fuse_local_reverse,
                canonicalization_policy=canonicalization_policy,
                svd_driver=svd_driver,
            )
        if resolved_device.type == "cuda":
            platform.synchronize(resolved_device)
            forward_peak_memory = int(torch.cuda.max_memory_allocated(resolved_device))
            torch.cuda.reset_peak_memory_stats(resolved_device)
        else:
            forward_peak_memory = 0
        forward_seconds = time.perf_counter() - forward_started
        for parameter in parameters:
            parameter.grad = None
        reverse_started = time.perf_counter()
        with (
            record_function("flagquantum::mps::reverse_vjp"),
            _nvtx_phase("flagquantum::mps::reverse_vjp", resolved_device),
        ):
            reverse.backward()
        if resolved_device.type == "cuda":
            platform.synchronize(resolved_device)
            reverse_peak_memory = int(torch.cuda.max_memory_allocated(resolved_device))
        else:
            reverse_peak_memory = 0
        reverse_seconds = time.perf_counter() - reverse_started
        diagnostic_parameter_values = (
            tuple(
                (index, float(parameters[index].detach().cpu()))
                for index in owned_indices
            )
            if record_parameter_gradients
            else ()
        )
        optimizer_started = time.perf_counter()
        optimizer_stage = (
            "lbfgs"
            if optimizer == "adam_lbfgs" and step >= lbfgs_transition_step
            else ("adam" if optimizer == "adam_lbfgs" else optimizer)
        )
        step_learning_rate = (
            lbfgs_lr
            if optimizer_stage == "lbfgs"
            else (
                float(optimizer_obj.param_groups[0]["lr"])
                if optimizer_obj is not None
                else float(lr * lr_decay**step)
            )
        )
        with (
            record_function("flagquantum::mps::optimizer"),
            _nvtx_phase("flagquantum::mps::optimizer", resolved_device),
        ):
            optimizer_collective_count = 0
            optimizer_collective_bytes = 0
            if optimizer_stage == "lbfgs":
                current_parameters = _owner_flatten(owned, gradients=False)
                current_gradient = _owner_flatten(owned, gradients=True)
                if previous_lbfgs_state is not None:
                    previous_lbfgs_parameters, previous_lbfgs_gradient = (
                        previous_lbfgs_state
                    )
                    step_delta = current_parameters - previous_lbfgs_parameters
                    gradient_delta = current_gradient - previous_lbfgs_gradient
                    curvature = _distributed_dot(step_delta, gradient_delta)
                    optimizer_collective_count += 1
                    if float(curvature.detach().cpu()) > 1e-14:
                        lbfgs_history.append(
                            (step_delta, gradient_delta, curvature.reciprocal())
                        )
                        del lbfgs_history[:-lbfgs_history_size]
                direction = _sharded_lbfgs_direction(current_gradient, lbfgs_history)
                optimizer_collective_count += 2 * len(lbfgs_history) + (
                    2 if lbfgs_history else 0
                )
                previous_lbfgs_state = (
                    current_parameters.clone(),
                    current_gradient.clone(),
                )
                _owner_add_(owned, lbfgs_lr * direction)
            elif optimizer_obj is not None:
                optimizer_obj.step()
                for parameter_group in optimizer_obj.param_groups:
                    parameter_group["lr"] *= lr_decay
            optimizer_collective_bytes += (
                optimizer_collective_count
                * torch.empty((), dtype=parameters[0].dtype).element_size()
            )
            parameter_collectives, parameter_bytes = _broadcast_parameters(
                parameters, parameter_broadcast_buckets
            )
            optimizer_collective_count += parameter_collectives
            optimizer_collective_bytes += parameter_bytes
        if resolved_device.type == "cuda":
            platform.synchronize(resolved_device)
        optimizer_seconds = time.perf_counter() - optimizer_started
        diagnostics_started = time.perf_counter()
        training_compute_seconds = diagnostics_started - step_started
        if resolved_device.type == "cuda":
            platform.synchronize(resolved_device)
            memory = max(forward_peak_memory, reverse_peak_memory)
            memory_snapshot = platform.memory_snapshot(resolved_device)
            allocated_memory = int(memory_snapshot.allocated_bytes or 0)
            reserved_memory = int(memory_snapshot.reserved_bytes or 0)
        else:
            memory = sum(
                parameter.numel() * parameter.element_size() for parameter in owned
            )
            allocated_memory = memory
            reserved_memory = memory
        memories.append(allocated_memory)
        element_size = torch.empty((), dtype=getattr(torch, ir.dtype)).element_size()
        (
            owned_site,
            owned_bond,
            two_site_splits,
            truncated_splits,
            discarded_weight,
            qr_activity,
            largest_realized_bond,
        ) = _owned_reverse_work(reverse.tape.records, rank)
        boundaries, boundary_bytes, bond_updates = _boundary_reverse_work(
            reverse.tape.records, rank, element_size
        )
        optimizer_memory = (
            0
            if optimizer_obj is None
            else sum(
                value.numel() * value.element_size()
                for state in optimizer_obj.state.values()
                for value in state.values()
                if isinstance(value, torch.Tensor)
            )
        )
        optimizer_memory += sum(
            value.numel() * value.element_size()
            for item in lbfgs_history
            for value in item
        )
        optimizer_memory += sum(
            value.numel() * value.element_size()
            for value in (previous_lbfgs_state or ())
        )
        loss = float(reverse.value.detach().cpu())
        losses.append(loss)
        diagnostics_seconds = time.perf_counter() - diagnostics_started
        metrics.append(
            MPSStepMetrics(
                step=step,
                loss=loss,
                forward_seconds=forward_seconds,
                reverse_seconds=reverse_seconds,
                optimizer_seconds=optimizer_seconds,
                end_to_end_seconds=time.perf_counter() - step_started,
                training_compute_seconds=training_compute_seconds,
                diagnostics_seconds=diagnostics_seconds,
                static_qr_metadata_records=reverse.static_qr_metadata_records,
                dynamic_metadata_broadcasts=reverse.dynamic_metadata_broadcasts,
                reverse_segment_cache_hit=reverse.reverse_segment_cache_hit,
                gradient_bucket_cache_hit=reverse.gradient_bucket_cache_hit,
                layer_halo_message_count=reverse.layer_halo_message_count,
                layer_halo_payload_bytes=reverse.layer_halo_payload_bytes,
                layer_halo_intra_node_bytes=reverse.layer_halo_intra_node_bytes,
                layer_halo_inter_node_bytes=reverse.layer_halo_inter_node_bytes,
                layer_halo_wait_seconds=reverse.layer_halo_wait_seconds,
                owned_site_work=owned_site,
                owned_bond_work=owned_bond,
                boundary_exchanges=boundaries,
                svd_activity=two_site_splits,
                qr_activity=qr_activity,
                peak_memory_bytes=memory,
                useful_work_completed=(owned_site + owned_bond) > 0,
                two_site_splits=two_site_splits,
                truncated_splits=truncated_splits,
                discarded_weight=discarded_weight,
                boundary_forward_exchanges=boundaries,
                boundary_reverse_exchanges=boundaries,
                boundary_bytes=boundary_bytes,
                bond_updates=bond_updates,
                allocated_memory_bytes=allocated_memory,
                reserved_memory_bytes=reserved_memory,
                tape_memory_bytes=reverse.tape.saved_tensor_bytes,
                optimizer_memory_bytes=optimizer_memory,
                communication_buffer_bytes=boundary_bytes,
                gradient_collective_count=reverse.gradient_collective_count,
                gradient_collective_bytes=reverse.gradient_collective_bytes,
                gradient_bucket_count=reverse.gradient_bucket_count,
                gradient_bucket_fill_ratio=reverse.gradient_bucket_fill_ratio,
                reverse_tape_segments=reverse.reverse_tape_segments,
                fused_reverse_segments=reverse.fused_reverse_segments,
                reverse_autograd_grad_invocations=(
                    reverse.reverse_autograd_grad_invocations
                ),
                reverse_python_dispatches=reverse.reverse_python_dispatches,
                qr_factorization_count=reverse.qr_factorization_count,
                svd_factorization_count=reverse.svd_factorization_count,
                objective_scan_pairs=reverse.objective_scan_pairs,
                objective_scan_forward_messages=reverse.objective_scan_forward_messages,
                objective_scan_reverse_messages=reverse.objective_scan_reverse_messages,
                objective_execution=reverse.objective_execution,
                learning_rate=step_learning_rate,
                largest_realized_bond=largest_realized_bond,
                forward_peak_memory_bytes=forward_peak_memory,
                reverse_peak_memory_bytes=reverse_peak_memory,
                optimizer_stage=optimizer_stage,
                objective_evaluations=1,
                parameter_gradients=(
                    _parameter_gradient_values(parameters, owned_indices)
                    if record_parameter_gradients
                    else ()
                ),
                parameter_values=diagnostic_parameter_values,
                optimizer_collective_count=optimizer_collective_count,
                optimizer_collective_bytes=optimizer_collective_bytes,
                lbfgs_history_length=len(lbfgs_history),
            )
        )
        if root is not None and (step + 1) % checkpoint_interval == 0:
            free_bytes, estimated_bytes = _checkpointing._checkpoint_storage_preflight(
                root,
                rank=rank,
                owned_indices=owned_indices,
                parameters=parameters,
                optimizer=optimizer_obj,
                reserve_bytes=checkpoint_free_space_reserve_bytes,
            )
            checkpoint_min_free_bytes_observed = (
                free_bytes
                if checkpoint_min_free_bytes_observed == 0
                else min(checkpoint_min_free_bytes_observed, free_bytes)
            )
            checkpoint_estimated_generation_bytes = max(
                checkpoint_estimated_generation_bytes, estimated_bytes
            )
            checkpoint_started = time.perf_counter()
            path = _checkpointing._save_checkpoint(
                root,
                rank=rank,
                world_size=world_size,
                step=step + 1,
                owned_indices=owned_indices,
                parameters=parameters,
                optimizer=optimizer_obj,
                contract=contract,
                bond_layout=local_bond_layout,
            )
            checkpoint_write_seconds += time.perf_counter() - checkpoint_started
            checkpoint_bytes_written += path.stat().st_size
            checkpoint_bytes_written += (
                _checkpointing._checkpoint_checksum_path(path).stat().st_size
            )
            checkpoints.append(str(path))
            commit_started = time.perf_counter()
            _checkpointing._commit_checkpoint_generation(
                root,
                rank=rank,
                world_size=world_size,
                step=step + 1,
                path=path,
                contract_fingerprint=contract_fingerprint,
            )
            checkpoint_commit_seconds += time.perf_counter() - commit_started
            prune_started = time.perf_counter()
            pruned_checkpoint_files.extend(
                _checkpointing._prune_checkpoint_generations_collective(
                    root,
                    rank=rank,
                    committed_step=step + 1,
                    keep_generations=checkpoint_retention_generations,
                )
            )
            checkpoint_prune_seconds += time.perf_counter() - prune_started

    stable_memories = memories[min(memory_warmup_steps, len(memories) - 1) :]
    growth = (
        max(0, stable_memories[-1] - stable_memories[0])
        if len(stable_memories) > 1
        else 0
    )
    ownership = tuple(
        MPSOptimizerOwnership(
            parameter_index=index,
            owner_rank=owner,
            optimizer_state_local=owner == rank,
            gradient_route="bounded_dtype_owner_reduce",
            update_route="owner_step_then_bounded_dtype_all_gather",
        )
        for index, owner in enumerate(owners)
    )
    checkpoint_retained_bytes = 0
    if root is not None:
        for path in root.glob(f"rank-{rank}-step-*.pt"):
            checkpoint_retained_bytes += path.stat().st_size
            checksum = _checkpointing._checkpoint_checksum_path(path)
            if checksum.is_file():
                checkpoint_retained_bytes += checksum.stat().st_size
    result = ShardedMPSTrainingResult(
        losses=tuple(losses),
        completed_steps=steps,
        start_step=start_step,
        rank=rank,
        world_size=world_size,
        local_world_size=local_world_size,
        optimizer=optimizer,
        ownership=ownership,
        steps=tuple(metrics),
        checkpoint_files=tuple(path for path in checkpoints if Path(path).is_file()),
        memory_growth_bytes=growth,
        suspected_memory_leak=growth > memory_leak_tolerance_bytes,
        checkpoint_contract_fingerprint=contract_fingerprint,
        memory_warmup_steps=memory_warmup_steps,
        communication_setup_seconds=communication_setup_seconds,
        site_kernel_execution=(
            "compiled" if resolved_compile_site_kernels else "eager"
        ),
        site_kernel_selection_reason=site_kernel_selection_reason,
        site_ownership=resolved_site_ownership,
        site_ownership_policy=resolved_site_ownership_policy,
        checkpoint_retention_generations=checkpoint_retention_generations,
        checkpoint_pruned_files=tuple(pruned_checkpoint_files),
        checkpoint_write_seconds=checkpoint_write_seconds,
        checkpoint_commit_seconds=checkpoint_commit_seconds,
        checkpoint_prune_seconds=checkpoint_prune_seconds,
        checkpoint_bytes_written=checkpoint_bytes_written,
        checkpoint_retained_bytes=checkpoint_retained_bytes,
        checkpoint_free_space_reserve_bytes=checkpoint_free_space_reserve_bytes,
        checkpoint_min_free_bytes_observed=checkpoint_min_free_bytes_observed,
        checkpoint_estimated_generation_bytes=checkpoint_estimated_generation_bytes,
        checkpoint_storage_semantics=(
            "shared_filesystem_verified" if world_size > 1 else "local_filesystem"
        ),
        checkpoint_storage_probe_seconds=checkpoint_storage_probe_seconds,
        checkpoint_overwrite_allowed=allow_checkpoint_overwrite,
        checkpoint_writer_lease_break_allowed=allow_checkpoint_writer_lease_break,
        checkpoint_writer_lease_stale_seconds=checkpoint_writer_lease_stale_seconds,
        checkpoint_writer_lease_heartbeat_count=(
            checkpoint_writer_lease_heartbeat_count
        ),
        checkpoint_writer_lease_heartbeat_seconds=(
            checkpoint_writer_lease_heartbeat_seconds
        ),
    )
    if root is not None:
        _checkpointing._release_checkpoint_writer_lease(root, rank=rank)
    return result


__all__ = ("train_distributed_mps",)

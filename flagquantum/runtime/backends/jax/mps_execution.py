"""Sharded MPS forward execution and parameterized tensor construction."""

from __future__ import annotations

from typing import Any, Sequence

from ...distributed.backend_policy import DistributedBackendPolicy
from .array_conversions import (
    _gate_matrix_as_jax,
    _jax_split_record,
    _parameterized_gate_matrix_as_jax,
)
from .backend_dispatch import plan_jax_distributed_quantum_backend
from .mps_canonicalization import _jax_mps_boundary_protocol
from .mps_kernels import (
    _apply_one_jax_mps_tensor,
    _apply_two_jax_mps_tensors,
    _initialize_jax_mps_rank_tensors,
    _rank_shards_from_jax_mps_tensors,
)
from .mps_result import JAXShardedMPSResult
from .planning_core import _as_ir
from .runtime_environment import (
    _jax_array_nbytes,
    _jax_complex_dtype,
    _jnp_device_put,
    _require_torch,
    _resolve_jax_device,
    _resolve_local_world_size,
    _resolve_policy,
    _resolve_world_size,
    _torch_complex_dtype,
)


def run_jax_sharded_mps(
    circuit_or_ir: Any,
    *,
    world_size: int | None = None,
    local_world_size: int | None = None,
    bsz: int = 1,
    complex_bytes: int | None = None,
    dtype: Any | None = None,
    max_bond: int | None = None,
    cutoff: float = 0.0,
    distributed_backend_policy: DistributedBackendPolicy | None = None,
    distributed_profile: str | None = None,
    jax_backend: str | None = None,
    torch_backend: str | None = None,
    device: str | None = None,
    strict_sharded: bool = True,
) -> JAXShardedMPSResult:
    """Execute a JAX MPS site-sharded development backend.

    The executor keeps one logical MPS partitioned by contiguous site ranges.
    It supports one-site gates, adjacent two-site gates inside a rank, and
    adjacent two-site gates crossing a shard boundary. Unsupported gates fail
    closed so this path never silently falls back to dense statevector or a full
    replicated MPS.
    """

    from ...distributed.engine import (
        _boundary_sync_record,
        _instruction_is_boundary_local,
        _instruction_is_site_local,
        _instruction_owner,
        _mps_shards,
    )

    torch = _require_torch()
    policy = _resolve_policy(
        distributed_backend_policy=distributed_backend_policy,
        distributed_profile=distributed_profile,
        jax_backend=jax_backend,
        torch_backend=torch_backend,
    )
    resolved_world_size = _resolve_world_size(world_size, policy)
    resolved_local_world_size = _resolve_local_world_size(
        local_world_size,
        world_size=resolved_world_size,
        policy=policy,
    )
    if complex_bytes is None:
        complex_bytes = 16 if dtype == torch.complex128 else 8
    ir = _as_ir(circuit_or_ir)
    jax_dtype = _jax_complex_dtype(complex_bytes)
    torch_dtype = _torch_complex_dtype(complex_bytes)
    jax_device = _resolve_jax_device(device)
    shard_plans = _mps_shards(ir.n_wires, resolved_world_size)
    rank_tensors = _initialize_jax_mps_rank_tensors(
        n_wires=ir.n_wires,
        bsz=bsz,
        shard_plans=shard_plans,
        dtype=jax_dtype,
        device=jax_device,
    )
    jax_plan = plan_jax_distributed_quantum_backend(
        ir,
        mode="mps",
        world_size=resolved_world_size,
        local_world_size=resolved_local_world_size,
        bsz=bsz,
        complex_bytes=complex_bytes,
        max_bond=max_bond,
        distributed_backend_policy=policy,
    )

    owned_instruction_count = 0
    sharded_kernel_count = 0
    boundary_sync_count = 0
    boundary_transfer_bytes = 0
    unsupported_instruction_count = 0
    truncation_error = 0.0
    max_truncation_error = 0.0
    truncation_records: list[dict[str, Any]] = []
    boundary_protocols: list[dict[str, Any]] = []

    for instruction in ir:
        owner = _instruction_owner(instruction, shard_plans)
        site_local = _instruction_is_site_local(instruction, shard_plans)
        boundary_local = _instruction_is_boundary_local(instruction, shard_plans)
        if len(instruction.wires) == 1 and site_local:
            matrix, _ = _gate_matrix_as_jax(
                instruction, torch_dtype=torch_dtype, jax_dtype=jax_dtype
            )
            matrix = _jnp_device_put(matrix, jax_device)
            wire = int(instruction.wires[0])
            rank_tensors[owner][wire] = _apply_one_jax_mps_tensor(
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
            matrix, _ = _gate_matrix_as_jax(
                instruction, torch_dtype=torch_dtype, jax_dtype=jax_dtype
            )
            matrix = _jnp_device_put(matrix, jax_device)
            first, second = int(instruction.wires[0]), int(instruction.wires[1])
            left_wire = min(first, second)
            left, right, split_info = _apply_two_jax_mps_tensors(
                rank_tensors[owner][left_wire],
                rank_tensors[owner][left_wire + 1],
                matrix,
                max_bond=max_bond,
                cutoff=cutoff,
                reverse=first > second,
            )
            rank_tensors[owner][left_wire] = left
            rank_tensors[owner][left_wire + 1] = right
            step_error = float(split_info["discarded_weight"])
            truncation_error += step_error
            max_truncation_error = max(max_truncation_error, step_error)
            record = _jax_split_record(
                split_info, bond=left_wire, max_bond=max_bond, cutoff=cutoff
            )
            if record is not None:
                truncation_records.append(record)
            owned_instruction_count += 1
            sharded_kernel_count += 1
            continue
        if boundary_local:
            matrix, _ = _gate_matrix_as_jax(
                instruction, torch_dtype=torch_dtype, jax_dtype=jax_dtype
            )
            matrix = _jnp_device_put(matrix, jax_device)
            first, second = int(instruction.wires[0]), int(instruction.wires[1])
            boundary = _boundary_sync_record(instruction, shard_plans)
            tensor_bytes = _jax_array_nbytes(
                rank_tensors[boundary.left_rank][boundary.left_wire]
            )
            tensor_bytes += _jax_array_nbytes(
                rank_tensors[boundary.right_rank][boundary.right_wire]
            )
            left, right, split_info = _apply_two_jax_mps_tensors(
                rank_tensors[boundary.left_rank][boundary.left_wire],
                rank_tensors[boundary.right_rank][boundary.right_wire],
                matrix,
                max_bond=max_bond,
                cutoff=cutoff,
                reverse=first > second,
            )
            rank_tensors[boundary.left_rank][boundary.left_wire] = left
            rank_tensors[boundary.right_rank][boundary.right_wire] = right
            step_error = float(split_info["discarded_weight"])
            truncation_error += step_error
            max_truncation_error = max(max_truncation_error, step_error)
            record = _jax_split_record(
                split_info, bond=boundary.left_wire, max_bond=max_bond, cutoff=cutoff
            )
            if record is not None:
                truncation_records.append(record)
            protocol = _jax_mps_boundary_protocol(
                left_wire=boundary.left_wire,
                right_wire=boundary.right_wire,
                left_rank=boundary.left_rank,
                right_rank=boundary.right_rank,
                local_world_size=resolved_local_world_size,
                tensor_bytes=tensor_bytes,
            )
            boundary_protocols.append(protocol)
            boundary_transfer_bytes += int(protocol["estimated_transfer_bytes"])
            boundary_sync_count += 1
            owned_instruction_count += 1
            sharded_kernel_count += 1
            continue

        unsupported_instruction_count += 1
        message = (
            "JAX sharded MPS does not support this instruction without a full-MPS/statevector fallback. "
            "Supported gates are one-site gates, adjacent two-site gates within one shard, and adjacent "
            f"two-site gates across a shard boundary. Unsupported instruction {instruction.name!r} "
            f"on wires {tuple(instruction.wires)}."
        )
        if strict_sharded:
            raise RuntimeError(message)
        raise RuntimeError(message)

    return JAXShardedMPSResult(
        rank_shards=_rank_shards_from_jax_mps_tensors(rank_tensors, shard_plans),
        shard_plans=tuple(shard_plans),
        jax_plan=jax_plan,
        backend_policy=policy,
        n_wires=ir.n_wires,
        bsz=bsz,
        complex_bytes=complex_bytes,
        max_bond=max_bond,
        cutoff=float(cutoff),
        owned_instruction_count=owned_instruction_count,
        sharded_kernel_count=sharded_kernel_count,
        boundary_sync_count=boundary_sync_count,
        boundary_transfer_bytes=boundary_transfer_bytes,
        unsupported_instruction_count=unsupported_instruction_count,
        truncation_error=float(truncation_error),
        max_truncation_error=float(max_truncation_error),
        truncation_records=tuple(truncation_records),
        boundary_protocols=tuple(boundary_protocols),
    )


def _jax_parameterized_mps_rank_tensors(
    circuit: Any,
    *,
    n_wires: int,
    bsz: int,
    shard_plans: Sequence[Any],
    complex_bytes: int,
    max_bond: int | None,
    cutoff: float,
    local_world_size: int,
    device: Any | None,
) -> tuple[
    dict[int, dict[int, Any]],
    int,
    int,
    int,
    int,
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    from ...distributed.engine import (
        _boundary_sync_record,
        _instruction_is_boundary_local,
        _instruction_is_site_local,
        _instruction_owner,
    )

    jax_dtype = _jax_complex_dtype(complex_bytes)
    rank_tensors = _initialize_jax_mps_rank_tensors(
        n_wires=int(n_wires),
        bsz=int(bsz),
        shard_plans=shard_plans,
        dtype=jax_dtype,
        device=device,
    )
    owned_instruction_count = 0
    sharded_kernel_count = 0
    boundary_sync_count = 0
    boundary_transfer_bytes = 0
    truncation_records: list[dict[str, Any]] = []
    boundary_protocols: list[dict[str, Any]] = []

    for instruction in circuit.to_ir():
        owner = _instruction_owner(instruction, shard_plans)
        site_local = _instruction_is_site_local(instruction, shard_plans)
        boundary_local = _instruction_is_boundary_local(instruction, shard_plans)
        if len(instruction.wires) == 1 and site_local:
            matrix, _ = _parameterized_gate_matrix_as_jax(
                instruction, complex_bytes=complex_bytes
            )
            matrix = _jnp_device_put(matrix, device)
            wire = int(instruction.wires[0])
            rank_tensors[owner][wire] = _apply_one_jax_mps_tensor(
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
            matrix, _ = _parameterized_gate_matrix_as_jax(
                instruction, complex_bytes=complex_bytes
            )
            matrix = _jnp_device_put(matrix, device)
            first, second = int(instruction.wires[0]), int(instruction.wires[1])
            left_wire = min(first, second)
            left, right, split_info = _apply_two_jax_mps_tensors(
                rank_tensors[owner][left_wire],
                rank_tensors[owner][left_wire + 1],
                matrix,
                max_bond=max_bond,
                cutoff=cutoff,
                reverse=first > second,
            )
            rank_tensors[owner][left_wire] = left
            rank_tensors[owner][left_wire + 1] = right
            record = _jax_split_record(
                split_info, bond=left_wire, max_bond=max_bond, cutoff=cutoff
            )
            if record is not None:
                truncation_records.append(record)
            owned_instruction_count += 1
            sharded_kernel_count += 1
            continue
        if boundary_local:
            matrix, _ = _parameterized_gate_matrix_as_jax(
                instruction, complex_bytes=complex_bytes
            )
            matrix = _jnp_device_put(matrix, device)
            first, second = int(instruction.wires[0]), int(instruction.wires[1])
            boundary = _boundary_sync_record(instruction, shard_plans)
            tensor_bytes = _jax_array_nbytes(
                rank_tensors[boundary.left_rank][boundary.left_wire]
            )
            tensor_bytes += _jax_array_nbytes(
                rank_tensors[boundary.right_rank][boundary.right_wire]
            )
            left, right, split_info = _apply_two_jax_mps_tensors(
                rank_tensors[boundary.left_rank][boundary.left_wire],
                rank_tensors[boundary.right_rank][boundary.right_wire],
                matrix,
                max_bond=max_bond,
                cutoff=cutoff,
                reverse=first > second,
            )
            rank_tensors[boundary.left_rank][boundary.left_wire] = left
            rank_tensors[boundary.right_rank][boundary.right_wire] = right
            record = _jax_split_record(
                split_info, bond=boundary.left_wire, max_bond=max_bond, cutoff=cutoff
            )
            if record is not None:
                truncation_records.append(record)
            protocol = _jax_mps_boundary_protocol(
                left_wire=boundary.left_wire,
                right_wire=boundary.right_wire,
                left_rank=boundary.left_rank,
                right_rank=boundary.right_rank,
                local_world_size=local_world_size,
                tensor_bytes=tensor_bytes,
            )
            boundary_protocols.append(protocol)
            boundary_transfer_bytes += int(protocol["estimated_transfer_bytes"])
            boundary_sync_count += 1
            owned_instruction_count += 1
            sharded_kernel_count += 1
            continue
        raise RuntimeError(
            "JAX sharded MPS parameter reverse mode does not support this instruction without a "
            f"full-MPS/statevector fallback: {instruction.name!r} on wires {tuple(instruction.wires)}."
        )
    return (
        rank_tensors,
        owned_instruction_count,
        sharded_kernel_count,
        boundary_sync_count,
        boundary_transfer_bytes,
        truncation_records,
        boundary_protocols,
    )

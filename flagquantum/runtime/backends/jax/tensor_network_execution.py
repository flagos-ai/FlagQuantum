"""Tensor-network slice records, contraction kernels, and forward execution."""

from __future__ import annotations

from typing import Any, Sequence

from ...distributed.backend_policy import DistributedBackendPolicy
from .array_conversions import _jax_nodes_from_torch_nodes
from .backend_dispatch import plan_jax_distributed_quantum_backend
from .common import node_count as _node_count
from .runtime_environment import (
    _jax_complex_dtype,
    _jnp_device_put,
    _require_torch,
    _resolve_collective_backend,
    _resolve_jax_device,
    _resolve_local_world_size,
    _resolve_policy,
    _resolve_world_size,
    _torch_complex_dtype,
)
from .tensor_network_contraction import _jax_contract_tensor_slices_by_backend
from .tensor_network_planning import _tn_tasks
from .tensor_network_records import (
    JAXShardedTensorNetworkResult,
    JAXTNSliceRankState,
)


def _jax_tn_task_summary(
    task: tuple[int, tuple[tuple[int, int], ...]], task_index: int
) -> dict[str, Any]:
    rank, assignments = task
    return {
        "task_index": int(task_index),
        "rank": int(rank),
        "assignments": tuple((int(label), int(value)) for label, value in assignments),
    }


def _resolve_tn_compute_backend(
    compute_backend: str, policy: DistributedBackendPolicy
) -> str:
    backend = str(compute_backend).lower()
    if backend in {"auto", ""}:
        if policy.profile == "production" and policy.jax_backend in {
            "pmap",
            "jax_pmap",
        }:
            return "pmap"
        if policy.profile == "production" and policy.jax_backend in {
            "shard_map",
            "jax_shard_map",
        }:
            return "shard_map"
        return "local_simulated"
    if backend in {"local", "local_simulated", "local_simulated_slice_compute"}:
        return "local_simulated"
    if backend in {"pmap", "jax_pmap", "jax_pmap_slice_contraction"}:
        return "pmap"
    if backend in {"shard_map", "jax_shard_map"}:
        return "shard_map"
    raise ValueError(
        "compute_backend must be 'auto', 'local_simulated', 'pmap', or 'shard_map'."
    )


def run_jax_sharded_tensor_network(
    circuit_or_ir: Any,
    *,
    world_size: int | None = None,
    local_world_size: int | None = None,
    bsz: int = 1,
    complex_bytes: int | None = None,
    dtype: Any | None = None,
    max_intermediate_size: int | None = None,
    sliced_labels: Sequence[int] | None = None,
    distributed_backend_policy: DistributedBackendPolicy | None = None,
    distributed_profile: str | None = None,
    jax_backend: str | None = None,
    torch_backend: str | None = None,
    device: str | None = None,
    compute_backend: str = "auto",
    collective_backend: str = "auto",
) -> JAXShardedTensorNetworkResult:
    """Execute JAX tensor-network contraction with slice tasks sharded by rank."""

    from ....simulation.tensor_network.entrypoints import build_tensor_network

    torch = _require_torch()
    policy = _resolve_policy(
        distributed_backend_policy=distributed_backend_policy,
        distributed_profile=distributed_profile,
        jax_backend=jax_backend,
        torch_backend=torch_backend,
    )
    resolved_collective_backend = _resolve_collective_backend(
        collective_backend, policy
    )
    resolved_compute_backend = _resolve_tn_compute_backend(compute_backend, policy)
    if (
        resolved_compute_backend == "pmap"
        and resolved_collective_backend == "local_simulated"
    ):
        resolved_collective_backend = "pmap"
    resolved_world_size = _resolve_world_size(world_size, policy)
    resolved_local_world_size = _resolve_local_world_size(
        local_world_size,
        world_size=resolved_world_size,
        policy=policy,
    )
    if complex_bytes is None:
        complex_bytes = 16 if dtype == torch.complex128 else 8
    torch_dtype = _torch_complex_dtype(complex_bytes)
    jax_dtype = _jax_complex_dtype(complex_bytes)
    jax_device = _resolve_jax_device(device)
    plan = build_tensor_network(
        circuit_or_ir,
        bsz=bsz,
        device="cpu",
        dtype=torch_dtype,
    )
    slicing = plan.slicing_plan(
        max_intermediate_size=max_intermediate_size, sliced_labels=sliced_labels
    )
    tasks = _tn_tasks(slicing.sliced_labels, slicing.slice_shape, resolved_world_size)
    active_ranks = tuple(sorted({int(rank) for rank, _ in tasks}))
    if resolved_world_size > 1 and (int(slicing.n_slices) < 2 or len(active_ranks) < 2):
        raise RuntimeError(
            "JAX sharded tensor-network execution requires at least two slice tasks assigned to multiple ranks. "
            "Pass sliced_labels or max_intermediate_size so the contraction is actually partitioned; "
            "FlagQuantum will not label an unsliced full contraction as distributed."
        )
    jax_nodes = _jax_nodes_from_torch_nodes(
        plan.nodes, dtype=jax_dtype, device=jax_device
    )
    task_summaries = tuple(
        _jax_tn_task_summary(task, index) for index, task in enumerate(tasks)
    )
    partial_values, reduced, compute_execution, collective_execution = (
        _jax_contract_tensor_slices_by_backend(
            jax_nodes,
            plan.output_labels,
            tasks,
            world_size=resolved_world_size,
            compute_backend=resolved_compute_backend,
            collective_backend=resolved_collective_backend,
        )
    )
    rank_partials = []
    for rank, partial in enumerate(partial_values):
        rank_summary_by_assignment = tuple(
            task_summaries[index]
            for index, task in enumerate(tasks)
            if int(task[0]) == rank
        )
        rank_partials.append(
            JAXTNSliceRankState(
                rank=rank, tasks=rank_summary_by_assignment, partial=partial
            )
        )
    reduced = _jnp_device_put(reduced, jax_device)
    jax_plan = plan_jax_distributed_quantum_backend(
        circuit_or_ir,
        mode="tensor_network",
        world_size=resolved_world_size,
        local_world_size=resolved_local_world_size,
        bsz=plan.bsz,
        complex_bytes=complex_bytes,
        max_intermediate_size=max_intermediate_size,
        sliced_labels=slicing.sliced_labels,
        distributed_backend_policy=policy,
    )
    return JAXShardedTensorNetworkResult(
        rank_partials=tuple(rank_partials),
        reduced_output=reduced,
        slicing=slicing,
        jax_plan=jax_plan,
        backend_policy=policy,
        n_wires=plan.n_wires,
        bsz=plan.bsz,
        complex_bytes=complex_bytes,
        local_world_size=resolved_local_world_size,
        node_count=_node_count(resolved_world_size, resolved_local_world_size),
        compute_backend=resolved_compute_backend,
        compute_execution=compute_execution,
        collective_backend=resolved_collective_backend,
        collective_execution=collective_execution,
    )

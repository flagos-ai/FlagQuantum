"""Tensor-network slice-task and representation planning."""

from __future__ import annotations

from itertools import product
from typing import Any, Sequence

from ....core.ir import CircuitIR
from ...distributed.backend_policy import DistributedBackendPolicy
from .common import node_count as _node_count
from .planning_core import JAXDistributedQuantumPlan


def _tn_tasks(
    sliced_labels: Sequence[int],
    slice_shape: Sequence[int],
    world_size: int,
) -> tuple[tuple[int, tuple[tuple[int, int], ...]], ...]:
    labels = tuple(int(label) for label in sliced_labels)
    ranges = [range(int(size)) for size in slice_shape]
    tasks = []
    for task_index, values in enumerate(product(*ranges) if ranges else [()]):
        tasks.append(
            (
                task_index % max(1, int(world_size)),
                tuple(zip(labels, tuple(int(value) for value in values))),
            )
        )
    return tuple(tasks)


def _tasks_by_rank_from_slicing(slicing: Any, world_size: int) -> tuple[int, ...]:
    tasks = _tn_tasks(slicing.sliced_labels, slicing.slice_shape, int(world_size))
    return tuple(
        sum(1 for rank, _ in tasks if int(rank) == target)
        for target in range(int(world_size))
    )


def _tensor_network_plan(
    ir: CircuitIR,
    *,
    policy: DistributedBackendPolicy,
    world_size: int,
    local_world_size: int,
    bsz: int,
    complex_bytes: int,
    max_intermediate_size: int | None,
    sliced_labels: Sequence[int] | None,
) -> JAXDistributedQuantumPlan:
    from ....simulation.tensor import build_tensor_network

    contraction_plan = build_tensor_network(ir, bsz=bsz, device="cpu")
    slicing = contraction_plan.slicing_plan(
        max_intermediate_size=max_intermediate_size,
        sliced_labels=sliced_labels,
    )
    tasks = _tn_tasks(slicing.sliced_labels, slicing.slice_shape, world_size)
    tasks_by_rank = {
        rank: sum(1 for task_rank, _ in tasks if task_rank == rank)
        for rank in range(world_size)
    }
    output_bytes = int(bsz) * (2 ** int(ir.n_wires)) * int(complex_bytes)
    local_memory = tuple(
        int(tasks_by_rank[rank] * output_bytes) for rank in range(world_size)
    )
    rank_ownership = tuple(
        {
            "rank": rank,
            "state_partition": "tensor_network_slices",
            "slice_task_count": tasks_by_rank[rank],
            "local_memory_bytes": local_memory[rank],
        }
        for rank in range(world_size)
    )
    blockers = (
        "jax_pmap_tensor_network_slice_executor_pending",
        "rank_local_jax_kernel_is_not_capacity_scaling",
        "tn_output_state_reduction_still_materializes_full_output",
    )
    gradient_blockers = ("jax_sharded_tensor_network_reverse_contraction_pending",)
    return JAXDistributedQuantumPlan(
        mode="tensor_network",
        n_wires=ir.n_wires,
        world_size=world_size,
        local_world_size=local_world_size,
        node_count=_node_count(world_size, local_world_size),
        backend_policy=policy,
        rank_ownership=rank_ownership,
        communication_tiers={
            "model": "jax_pmap_tensor_network_slice_reduce_planned",
            "collective": "psum",
            "reduction_tensor_bytes": output_bytes,
            "inter_node_collective_possible": bool(
                _node_count(world_size, local_world_size) > 1 and world_size > 1
            ),
            "note": "Exact physical bytes depend on XLA collective lowering.",
        },
        local_memory_bytes_by_rank=local_memory,
        blockers=blockers,
        gradient_blockers=gradient_blockers,
        task_summary={
            "sliced_labels": slicing.sliced_labels,
            "slice_shape": slicing.slice_shape,
            "slice_task_count": len(tasks),
            "tasks_by_rank": tasks_by_rank,
            "max_intermediate_size": max_intermediate_size,
        },
    )

"""Distributed tensor-network reduction execution.

This module provides FlagQuantum-native distributed result objects without
depending on an external graph or tensor-network package. Development backends
must preserve the same rank ownership and communication semantics as production
backends, while production backends are expected to execute one logical workload
across rank-local shards instead of replicating the full circuit per rank.
"""

from __future__ import annotations

from typing import Any, Sequence

import torch
import torch.distributed as dist

from ...simulation.tensor import (
    TensorNetworkNode,
    _contract_nodes_greedy,
    _label_dims,
    _slice_nodes,
    build_tensor_network,
    run_tensor_network,
)
from ..backends.jax import plan_jax_distributed_quantum_backend
from ..backends.mps.operations import (
    tensor_nbytes as _tensor_nbytes,
)
from .models import (
    DistributedSliceTask,
    DistributedTensorNetworkState,
    _resolve_backend_policy,
    _should_use_torch_distributed,
    _tensor_slice_tasks,
    init_torch_distributed,
)


class _DistributedAllReduceSum(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, tensor: torch.Tensor) -> torch.Tensor:
        out = tensor.clone()
        dist.all_reduce(out, op=dist.ReduceOp.SUM)
        return out

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor]:
        return (grad_output,)


def _all_reduce_sum_autograd(tensor: torch.Tensor) -> torch.Tensor:
    if not tensor.requires_grad:
        out = tensor.clone()
        dist.all_reduce(out, op=dist.ReduceOp.SUM)
        return out
    return _DistributedAllReduceSum.apply(tensor)


def _zero_for_output(
    nodes: Sequence[TensorNetworkNode], output_labels: Sequence[int]
) -> torch.Tensor:
    dims = _label_dims(nodes)
    reference = nodes[0].tensor if nodes else torch.empty((), dtype=torch.complex64)
    return torch.zeros(
        tuple(dims[label] for label in output_labels),
        dtype=reference.dtype,
        device=reference.device,
    )


def _contract_assigned_tensor_slices(
    nodes: Sequence[TensorNetworkNode],
    output_labels: Sequence[int],
    tasks: Sequence[DistributedSliceTask],
) -> torch.Tensor:
    partial = _zero_for_output(nodes, output_labels)
    if not tasks:
        return partial
    for task in tasks:
        subnodes = _slice_nodes(nodes, dict(task.assignments))
        subtotal, _ = _contract_nodes_greedy(subnodes, output_labels)
        partial = partial + subtotal
    return partial


def run_distributed_tensor_network(
    circuit_or_ir: Any,
    *,
    world_size: int = 1,
    max_intermediate_size: int | None = None,
    sliced_labels: Sequence[int] | None = None,
    distributed_executor: str = "auto",
    backend: str | None = None,
    init_method: str = "env://",
    rank: int | None = None,
    local_rank: int | None = None,
    **options: Any,
) -> DistributedTensorNetworkState:
    """Run tensor-network contraction with torch.distributed slice parallelism."""

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

    plan = build_tensor_network(
        circuit_or_ir,
        bsz=int(options.get("bsz", 1)),
        device=options.get("device", "cpu"),
        dtype=options.get("dtype"),
    )
    slicing = plan.slicing_plan(
        max_intermediate_size=max_intermediate_size, sliced_labels=sliced_labels
    )
    tasks = _tensor_slice_tasks(slicing.sliced_labels, slicing.slice_shape, world_size)
    state_cache = None
    local_simulation = False
    rank_partial_bytes: dict[int, int] = {}
    if context is not None and context.initialized:
        local_tasks = tuple(task for task in tasks if task.rank == context.rank)
        partial = _contract_assigned_tensor_slices(
            plan.nodes, plan.output_labels, local_tasks
        )
        rank_partial_bytes[context.rank] = _tensor_nbytes(partial)
        partial = _all_reduce_sum_autograd(partial)
        state_cache = partial.reshape(plan.bsz, 2**plan.n_wires)
    elif world_size > 1 and backend_policy.torch_backend == "local_tensor":
        partials = []
        for task_rank in range(world_size):
            local_tasks = tuple(task for task in tasks if task.rank == task_rank)
            partial = _contract_assigned_tensor_slices(
                plan.nodes, plan.output_labels, local_tasks
            )
            rank_partial_bytes[task_rank] = _tensor_nbytes(partial)
            partials.append(partial)
        if partials:
            total = partials[0]
            for partial in partials[1:]:
                total = total + partial
        else:
            total = _zero_for_output(plan.nodes, plan.output_labels)
        state_cache = total.reshape(plan.bsz, 2**plan.n_wires)
        local_simulation = True
    local = run_tensor_network(
        circuit_or_ir,
        contraction_strategy="sliced",
        max_intermediate_size=max_intermediate_size,
        sliced_labels=slicing.sliced_labels,
        **options,
    )
    if state_cache is not None:
        local._state_cache = state_cache
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
        mode="tensor_network",
        world_size=world_size,
        local_world_size=jax_local_world_size,
        bsz=local.bsz,
        max_intermediate_size=max_intermediate_size,
        sliced_labels=slicing.sliced_labels,
        distributed_backend_policy=backend_policy,
    ).summary()
    return DistributedTensorNetworkState(
        local,
        world_size=world_size,
        tasks=tasks,
        context=context,
        state_cache=state_cache,
        backend_policy=backend_policy,
        local_simulation=local_simulation,
        rank_partial_bytes=rank_partial_bytes,
        jax_distributed_plan=jax_distributed_plan,
    )


__all__ = ["run_distributed_tensor_network"]

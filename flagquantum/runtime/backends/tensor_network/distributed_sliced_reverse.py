"""Real torch.distributed execution of rank-owned TN slices."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Sequence

import torch
import torch.distributed as dist

from ....simulation.tensor_models import (
    TensorNetworkContractionPlan,
    TensorNetworkExpectationPlan,
    TensorNetworkSlicingPlan,
)
from .distributed_optimizer import _packed_owner_layout, plan_tn_parameter_owners
from .sliced_reverse import execute_sliced_tn_explicit_reverse
from .sliced_tasks import DistributedTNSliceTaskPlan


@dataclass(frozen=True)
class DistributedSlicedTNReverseResult:
    """Rank-local work and globally reduced sliced-TN result."""

    value: torch.Tensor
    parameter_gradients: tuple[torch.Tensor, ...]
    task_plan_identity: str
    rank: int
    world_size: int
    local_task_count: int
    local_slice_batch_size: int
    local_forward_operation_count: int
    local_reverse_operation_count: int
    local_execution_seconds: float
    collective_seconds: float
    collective_count: int
    collective_payload_bytes_per_rank: int
    nonfinite_cotangent_count: int
    nonfinite_parameter_gradient_count: int
    local_saved_tape_bytes: int
    local_rematerialized_operation_count: int
    gradient_aggregation_semantics: str
    parameter_gradient_owner_ranks: tuple[int, ...]

    def summary(self) -> dict[str, Any]:
        return {
            "task_plan_identity": self.task_plan_identity,
            "rank": self.rank,
            "world_size": self.world_size,
            "local_task_count": self.local_task_count,
            "local_slice_batch_size": self.local_slice_batch_size,
            "local_slice_batch_count": self.local_task_count
            // self.local_slice_batch_size,
            "local_forward_operation_count": self.local_forward_operation_count,
            "local_reverse_operation_count": self.local_reverse_operation_count,
            "local_execution_seconds": self.local_execution_seconds,
            "collective_seconds": self.collective_seconds,
            "collective_count": self.collective_count,
            "collective_payload_bytes_per_rank": (
                self.collective_payload_bytes_per_rank
            ),
            "nonfinite_cotangent_count": self.nonfinite_cotangent_count,
            "nonfinite_parameter_gradient_count": (
                self.nonfinite_parameter_gradient_count
            ),
            "local_saved_tape_bytes": self.local_saved_tape_bytes,
            "local_rematerialized_operation_count": (
                self.local_rematerialized_operation_count
            ),
            "distribution_semantics": "sharded_across_ranks",
            "forward_distribution_semantics": "sharded_across_ranks",
            "backward_distribution_semantics": "sharded_across_ranks",
            "gradient_distribution_semantics": (
                "sharded_across_ranks"
                if self.gradient_aggregation_semantics == "reduce_to_parameter_owner"
                else "replicated_after_all_reduce"
            ),
            "local_gradient_contribution_semantics": "rank_owned_slice_contributions",
            "gradient_aggregation_semantics": self.gradient_aggregation_semantics,
            "parameter_gradient_owner_ranks": self.parameter_gradient_owner_ranks,
            "scalability_claim_allowed": False,
            "scalability_blockers": (
                "accelerator_capacity_evidence_pending",
                "production_benchmark_audit_pending",
            ),
        }


def execute_distributed_sliced_tn_explicit_reverse(
    plan: TensorNetworkContractionPlan | TensorNetworkExpectationPlan,
    slicing: TensorNetworkSlicingPlan,
    tasks: DistributedTNSliceTaskPlan,
    parameters: Sequence[torch.Tensor],
    *,
    process_group: Any | None = None,
    output_cotangent: torch.Tensor | None = None,
    checkpoint_budget_bytes: int | None = None,
    compiled_reverse: bool = False,
    deferred_parameter_pullback: bool = False,
    slice_batch_size: int = 1,
    gradient_reduction: str = "all_reduce",
) -> DistributedSlicedTNReverseResult:
    """Execute only rank-owned slices, then reduce value and all gradients."""

    if not dist.is_initialized():
        raise RuntimeError(
            "distributed sliced TN reverse requires an initialized process group"
        )
    tasks.validate()
    world_size = int(dist.get_world_size(group=process_group))
    rank = int(dist.get_rank(group=process_group))
    if gradient_reduction not in {"all_reduce", "owner_reduce"}:
        raise ValueError("gradient_reduction must be 'all_reduce' or 'owner_reduce'")
    if tasks.world_size != world_size:
        raise RuntimeError("TN slice task plan world size does not match process group")
    if (
        tasks.slicing_labels != slicing.sliced_labels
        or tasks.slice_shape != slicing.slice_shape
    ):
        raise ValueError("TN slice task plan does not match the slicing plan")
    local_tasks = tuple(task for task in tasks.tasks if task.owner_rank == rank)
    if not local_tasks:
        raise RuntimeError(
            "distributed sliced TN reverse requires at least one task per rank"
        )

    if torch.cuda.is_available() and any(parameter.is_cuda for parameter in parameters):
        torch.cuda.synchronize()
    local_start = perf_counter()
    local = execute_sliced_tn_explicit_reverse(
        plan,
        slicing,
        parameters,
        output_cotangent=output_cotangent,
        checkpoint_budget_bytes=checkpoint_budget_bytes,
        compiled_reverse=compiled_reverse,
        deferred_parameter_pullback=deferred_parameter_pullback,
        slice_batch_size=slice_batch_size,
        _task_assignments=tuple(task.assignments for task in local_tasks),
    )
    if torch.cuda.is_available() and local.value.is_cuda:
        torch.cuda.synchronize(local.value.device)
    local_seconds = perf_counter() - local_start

    gradients = tuple(
        torch.zeros_like(parameter) if gradient is None else gradient
        for parameter, gradient in zip(parameters, local.parameter_gradients)
    )
    value = local.value.clone()
    if value.is_cuda:
        torch.cuda.synchronize(value.device)
    collective_start = perf_counter()
    dist.all_reduce(value, op=dist.ReduceOp.SUM, group=process_group)
    gradient_owners = (
        plan_tn_parameter_owners(len(gradients), world_size) if gradients else ()
    )
    if gradient_reduction == "all_reduce":
        for gradient in gradients:
            dist.all_reduce(gradient, op=dist.ReduceOp.SUM, group=process_group)
        gradient_collective_count = len(gradients)
    elif gradients:
        reference = gradients[0]
        if any(
            item.device != reference.device or item.dtype != reference.dtype
            for item in gradients
        ):
            raise ValueError(
                "packed owner gradient reduction requires one dtype and device"
            )
        chunk_size, offsets = _packed_owner_layout(
            gradients, gradient_owners, world_size
        )
        packed = torch.zeros(
            world_size * chunk_size,
            dtype=reference.dtype,
            device=reference.device,
        )
        for index, (gradient, owner) in enumerate(zip(gradients, gradient_owners)):
            start = owner * chunk_size + offsets[index]
            packed[start : start + gradient.numel()].copy_(gradient.reshape(-1))
        # One packed all-reduce is supported consistently by both the NCCL
        # production path and the Gloo semantic-test path.  Only owner chunks
        # are materialized back into gradient tensors below.
        dist.all_reduce(packed, op=dist.ReduceOp.SUM, group=process_group)
        owned_chunk = packed[rank * chunk_size : (rank + 1) * chunk_size]
        for index, (gradient, owner) in enumerate(zip(gradients, gradient_owners)):
            if owner == rank:
                start = offsets[index]
                gradient.copy_(
                    owned_chunk[start : start + gradient.numel()].view_as(gradient)
                )
            else:
                gradient.zero_()
        gradient_collective_count = 1
    else:
        gradient_collective_count = 0
    if value.is_cuda:
        torch.cuda.synchronize(value.device)
    collective_seconds = perf_counter() - collective_start
    payload_bytes = int(value.numel()) * int(value.element_size()) + sum(
        int(gradient.numel()) * int(gradient.element_size()) for gradient in gradients
    )
    return DistributedSlicedTNReverseResult(
        value=value,
        parameter_gradients=gradients,
        task_plan_identity=tasks.identity,
        rank=rank,
        world_size=world_size,
        local_task_count=len(local_tasks),
        local_slice_batch_size=int(slice_batch_size),
        local_forward_operation_count=local.forward_operation_count,
        local_reverse_operation_count=local.reverse_operation_count,
        local_execution_seconds=local_seconds,
        collective_seconds=collective_seconds,
        collective_count=1 + gradient_collective_count,
        collective_payload_bytes_per_rank=payload_bytes,
        nonfinite_cotangent_count=local.nonfinite_cotangent_count,
        nonfinite_parameter_gradient_count=(local.nonfinite_parameter_gradient_count),
        local_saved_tape_bytes=local.saved_tape_bytes,
        local_rematerialized_operation_count=(local.rematerialized_operation_count),
        gradient_aggregation_semantics=(
            "reduce_to_parameter_owner"
            if gradient_reduction == "owner_reduce"
            else "all_reduce_replicated_result"
        ),
        parameter_gradient_owner_ranks=gradient_owners,
    )

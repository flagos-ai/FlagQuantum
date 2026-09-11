"""Rank-owned optimizer updates for distributed tensor-network training."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Sequence

import torch
import torch.distributed as dist

from ....compute import get_platform_runtime


@dataclass(frozen=True)
class DistributedTNOptimizerStepResult:
    """Evidence from one rank-owned SGD update and parameter synchronization."""

    rank: int
    world_size: int
    learning_rate: float
    owned_parameter_indices: tuple[int, ...]
    ownership: tuple[dict[str, Any], ...]
    collective_count: int
    collective_payload_bytes_per_rank: int
    execution_seconds: float

    def summary(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "world_size": self.world_size,
            "optimizer_name": "rank_owned_sgd",
            "learning_rate": self.learning_rate,
            "training_step_count": 1,
            "owned_parameter_indices": self.owned_parameter_indices,
            "parameter_ownership_semantics": "sharded_across_ranks",
            "gradient_ownership_semantics": "rank_owned_optimizer_consumption",
            "optimizer_update_semantics": "sharded_across_ranks",
            "optimizer_update_ownership_semantics": "sharded_across_ranks",
            "optimizer_update_ownership": self.ownership,
            "parameter_distribution_after_step": "replicated_for_next_forward",
            "parameter_writeback_route": "owner_update_then_packed_all_gather",
            "optimizer_state_distribution": "rank_owned",
            "collective_count": self.collective_count,
            "collective_payload_bytes_per_rank": (
                self.collective_payload_bytes_per_rank
            ),
            "execution_seconds": self.execution_seconds,
        }


def plan_tn_parameter_owners(
    parameter_count: int,
    world_size: int,
) -> tuple[int, ...]:
    """Assign contiguous, balanced parameter ranges to ranks."""

    count = int(parameter_count)
    ranks = int(world_size)
    if count <= 0:
        raise ValueError("parameter_count must be positive")
    if ranks <= 0:
        raise ValueError("world_size must be positive")
    return tuple(min(ranks - 1, index * ranks // count) for index in range(count))


def _packed_owner_layout(
    tensors: Sequence[torch.Tensor], owners: Sequence[int], world_size: int
) -> tuple[int, tuple[int, ...]]:
    """Return a fixed-width rank chunk and each tensor's offset in that chunk."""

    used = [0] * world_size
    offsets: list[int] = []
    for tensor, owner in zip(tensors, owners):
        offsets.append(used[owner])
        used[owner] += int(tensor.numel())
    return max(used), tuple(offsets)


def execute_rank_owned_tn_sgd_step(
    parameters: Sequence[torch.Tensor],
    gradients: Sequence[torch.Tensor],
    *,
    learning_rate: float,
    process_group: Any | None = None,
) -> DistributedTNOptimizerStepResult:
    """Apply each SGD update on one owner rank, then broadcast parameter values.

    Gradient tensors may be replicated outputs of the sliced reverse all-reduce.
    Only the assigned owner consumes a given gradient and mutates that parameter.
    Broadcasting the updated values is necessary because every rank needs the
    circuit parameters for its distinct slices in the next forward pass.
    """

    if not dist.is_initialized():
        raise RuntimeError("rank-owned TN optimizer requires a process group")
    if len(parameters) != len(gradients):
        raise ValueError("parameters and gradients must have equal length")
    if not parameters:
        raise ValueError("rank-owned TN optimizer requires parameters")
    step_size = float(learning_rate)
    if not torch.isfinite(torch.tensor(step_size)) or step_size <= 0.0:
        raise ValueError("learning_rate must be finite and positive")

    world_size = int(dist.get_world_size(group=process_group))
    rank = int(dist.get_rank(group=process_group))
    owners = plan_tn_parameter_owners(len(parameters), world_size)
    owned = tuple(index for index, owner in enumerate(owners) if owner == rank)
    ownership = tuple(
        {
            "rank": owner,
            "parameter_indices": tuple(
                index
                for index, assigned_owner in enumerate(owners)
                if assigned_owner == owner
            ),
            "ownership": "rank_owned_optimizer_update",
            "writeback_route": "owner_update_then_packed_all_gather",
        }
        for owner in range(world_size)
    )

    for index, (parameter, gradient) in enumerate(zip(parameters, gradients)):
        if parameter.shape != gradient.shape:
            raise ValueError(f"gradient shape mismatch for parameter {index}")
        if parameter.device != gradient.device:
            raise ValueError(f"gradient device mismatch for parameter {index}")
        if parameter.dtype != gradient.dtype:
            raise ValueError(f"gradient dtype mismatch for parameter {index}")
    if not bool(
        torch.stack(tuple(torch.isfinite(item).all() for item in gradients)).all()
    ):
        raise ValueError("optimizer gradients contain a nonfinite value")

    reference = parameters[0]
    if any(
        parameter.device != reference.device or parameter.dtype != reference.dtype
        for parameter in parameters
    ):
        raise ValueError(
            "packed rank-owned SGD requires one parameter dtype and device"
        )
    chunk_size, offsets = _packed_owner_layout(parameters, owners, world_size)
    platform = get_platform_runtime(reference.device.type)

    if reference.is_cuda:
        platform.synchronize(reference.device)
    started = perf_counter()
    with torch.no_grad():
        for index in owned:
            parameters[index].add_(gradients[index], alpha=-step_size)
        local_chunk = torch.zeros(
            chunk_size, dtype=reference.dtype, device=reference.device
        )
        for index in owned:
            start = offsets[index]
            local_chunk[start : start + parameters[index].numel()].copy_(
                parameters[index].reshape(-1)
            )
        gathered = torch.empty(
            world_size * chunk_size,
            dtype=reference.dtype,
            device=reference.device,
        )
        dist.all_gather_single(gathered, local_chunk, group=process_group)
        for index, (parameter, owner) in enumerate(zip(parameters, owners)):
            start = owner * chunk_size + offsets[index]
            parameter.copy_(
                gathered[start : start + parameter.numel()].view_as(parameter)
            )
        if not bool(
            torch.stack(tuple(torch.isfinite(item).all() for item in parameters)).all()
        ):
            raise RuntimeError("parameters became nonfinite after owner all-gather")
    if reference.is_cuda:
        platform.synchronize(reference.device)
    elapsed = perf_counter() - started
    return DistributedTNOptimizerStepResult(
        rank=rank,
        world_size=world_size,
        learning_rate=step_size,
        owned_parameter_indices=owned,
        ownership=ownership,
        collective_count=1,
        collective_payload_bytes_per_rank=chunk_size * reference.element_size(),
        execution_seconds=elapsed,
    )

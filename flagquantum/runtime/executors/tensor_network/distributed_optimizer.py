"""Rank-owned optimizer updates for distributed tensor-network training."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from time import perf_counter
from typing import Any

import torch
import torch.distributed as dist

from ....compute import get_platform_runtime
from ...distributed.context import resolve_local_world_size, resolve_node_count

#: How each sliced-reverse gradient reduction leaves the reduced gradient
#: distributed. `owner_reduce` zeroes every non-owned gradient, so the reduced
#: contribution for a parameter exists only on the rank that owns it;
#: `all_reduce` leaves the full reduced vector on every rank. Only the first is
#: an ownership statement, and it is read from the reduction that ran rather
#: than asserted beside it.
_GRADIENT_DISTRIBUTION_BY_REDUCTION: dict[str, str] = {
    "owner_reduce": "sharded_across_ranks",
    "all_reduce": "replicated_after_all_reduce",
}

#: The gradient reductions the sliced reverse executes. One definition, so the
#: executor that runs the collective and the evidence that describes it cannot
#: disagree about which reductions exist.
GRADIENT_REDUCTION_MODES = frozenset(_GRADIENT_DISTRIBUTION_BY_REDUCTION)


@dataclass(frozen=True)
class DistributedTNOptimizerStepResult:
    """Evidence from one rank-owned SGD update and parameter synchronization."""

    rank: int
    world_size: int
    local_world_size: int
    learning_rate: float
    owned_parameter_indices: tuple[int, ...]
    ownership: tuple[dict[str, Any], ...]
    gradient_reduction: str
    collective_count: int
    collective_payload_bytes_per_rank: int
    execution_seconds: float

    @property
    def gradient_distribution_semantics(self) -> str:
        """How the reduction left the reduced gradient across the ranks."""

        return _GRADIENT_DISTRIBUTION_BY_REDUCTION[self.gradient_reduction]

    @property
    def gradient_ownership_semantics(self) -> str:
        """Whether a reduced gradient is owned by one rank or held by all."""

        return self.gradient_distribution_semantics

    @property
    def gradient_ownership(self) -> tuple[dict[str, Any], ...]:
        """The per-rank gradient ownership map, empty when gradients replicate.

        A replicated gradient has no owner to report: publishing the update
        ownership here instead would describe a different tensor.
        """

        if self.gradient_ownership_semantics != "sharded_across_ranks":
            return ()
        return self.ownership

    def summary(self) -> dict[str, Any]:
        node_count = resolve_node_count(self.world_size, self.local_world_size)
        return {
            "rank": self.rank,
            "world_size": self.world_size,
            "local_world_size": self.local_world_size,
            "node_count": node_count,
            "rank_placement": {
                "rank": self.rank,
                "local_rank": self.rank % max(1, self.local_world_size),
                "world_size": self.world_size,
                "local_world_size": self.local_world_size,
                "node_rank": self.rank // max(1, self.local_world_size),
                "node_count": node_count,
            },
            "optimizer_name": "rank_owned_sgd",
            "learning_rate": self.learning_rate,
            "training_step_count": 1,
            "owned_parameter_indices": self.owned_parameter_indices,
            "parameter_ownership_semantics": "sharded_across_ranks",
            "gradient_ownership_semantics": self.gradient_ownership_semantics,
            "gradient_distribution_semantics": self.gradient_distribution_semantics,
            "gradient_reduction": self.gradient_reduction,
            "gradient_ownership": self.gradient_ownership,
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


def plan_tn_parameter_ownership(
    parameter_count: int,
    world_size: int,
) -> tuple[dict[str, Any], ...]:
    """One ownership record per rank, covering every parameter exactly once.

    Parameter ownership, gradient ownership, and update ownership are the same
    partition on this path: the rank that owns a parameter is the rank whose
    gradient survives the reduction and the rank that applies the update. The
    record is built here once so the three maps a result publishes cannot drift
    apart, and it carries ``owned_parameter_count`` so a consumer can check the
    partition covers the parameter list without parsing index tuples.
    """

    owners = plan_tn_parameter_owners(parameter_count, world_size)
    return tuple(
        {
            "rank": owner,
            "parameter_indices": tuple(
                index for index, assigned in enumerate(owners) if assigned == owner
            ),
            "owned_parameter_count": sum(1 for assigned in owners if assigned == owner),
            "ownership": "rank_owned_optimizer_update",
            "writeback_route": "owner_update_then_packed_all_gather",
        }
        for owner in range(int(world_size))
    )


def _packed_owner_layout(
    tensors: Sequence[torch.Tensor], owners: Sequence[int], world_size: int
) -> tuple[int, tuple[int, ...]]:
    """Return a fixed-width rank chunk and each tensor's offset in that chunk."""

    used = [0] * world_size
    offsets: list[int] = []
    for tensor, owner in zip(tensors, owners, strict=True):
        offsets.append(used[owner])
        used[owner] += int(tensor.numel())
    return max(used), tuple(offsets)


def execute_rank_owned_tn_sgd_step(
    parameters: Sequence[torch.Tensor],
    gradients: Sequence[torch.Tensor],
    *,
    learning_rate: float,
    gradient_reduction: str,
    process_group: Any | None = None,
) -> DistributedTNOptimizerStepResult:
    """Apply each SGD update on one owner rank, then broadcast parameter values.

    ``gradient_reduction`` names the sliced-reverse reduction that produced
    ``gradients``, and it decides what this step may report about them: an
    owner-scoped reduction leaves only the owned gradient materialized, while a
    full all-reduce leaves the whole reduced vector on every rank. Only the
    assigned owner consumes a given gradient and mutates that parameter.
    Broadcasting the updated values is necessary because every rank needs the
    circuit parameters for its distinct slices in the next forward pass.
    """

    # Argument checks precede the process-group check: a reduction this step
    # cannot describe is knowable before any resource is required, and the
    # evidence it would publish is the reason it exists.
    if gradient_reduction not in GRADIENT_REDUCTION_MODES:
        raise ValueError(
            "gradient_reduction must be one of "
            f"{sorted(GRADIENT_REDUCTION_MODES)}, got {gradient_reduction!r}"
        )
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
    ownership = plan_tn_parameter_ownership(len(parameters), world_size)

    for index, (parameter, gradient) in enumerate(
        zip(parameters, gradients, strict=True)
    ):
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
        dist.all_gather_into_tensor(gathered, local_chunk, group=process_group)
        for index, (parameter, owner) in enumerate(
            zip(parameters, owners, strict=True)
        ):
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
        local_world_size=resolve_local_world_size(world_size),
        learning_rate=step_size,
        owned_parameter_indices=owned,
        ownership=ownership,
        gradient_reduction=gradient_reduction,
        collective_count=1,
        collective_payload_bytes_per_rank=chunk_size * reference.element_size(),
        execution_seconds=elapsed,
    )

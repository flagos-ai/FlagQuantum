"""Rank-local pair contractions for sharded tensor-network values."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Sequence

import torch
import torch.distributed as dist

from ....simulation.tensor_contraction import _einsum_pair_by_labels
from .distributed_dag import (
    DistributedTNContractionDAG,
    DistributedTNContractionRecord,
    DistributedTNShard,
)


@dataclass(frozen=True)
class DistributedTNShardedPairResult:
    """One rank's output from a pre-sharded pair contraction."""

    value: torch.Tensor
    mode: Literal["retained_output_shard", "contracted_partial_reduce"]
    rank: int
    world_size: int
    shard_label: int
    shard: DistributedTNShard
    collective_count: int
    collective_bytes: int
    full_input_materialized: bool = False
    full_output_materialized: bool = False

    def summary(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "rank": self.rank,
            "world_size": self.world_size,
            "shard_label": self.shard_label,
            "shard_range": (self.shard.start, self.shard.stop),
            "local_output_shape": tuple(int(dim) for dim in self.value.shape),
            "local_output_bytes": int(self.value.numel())
            * int(self.value.element_size()),
            "collective_count": self.collective_count,
            "collective_bytes": self.collective_bytes,
            "full_input_materialized": self.full_input_materialized,
            "full_output_materialized": self.full_output_materialized,
            "distribution_semantics": "rank_group_sharded_intermediate",
            "scalability_claim_allowed": False,
            "scalability_blockers": ("multi_gpu_capacity_evidence_pending",),
        }


def select_sharded_pair_mode(
    dag: DistributedTNContractionDAG,
    operation: DistributedTNContractionRecord,
) -> tuple[
    Literal["retained_output_shard", "contracted_partial_reduce"],
    int,
]:
    """Select a supported sharded pair mode or fail before execution."""

    dag.validate()
    values = {value.value_id: value for value in dag.values}
    output = values[operation.output_value_id]
    inputs = tuple(values[value_id] for value_id in operation.input_value_ids)
    if output.semantics == "sharded":
        if output.shard_label not in output.labels:
            raise ValueError("sharded TN output label is not retained")
        return "retained_output_shard", int(output.shard_label)
    sharded_inputs = tuple(value for value in inputs if value.semantics == "sharded")
    if not sharded_inputs:
        raise ValueError("distributed TN operation has no sharded value")
    labels = {value.shard_label for value in sharded_inputs}
    if len(labels) != 1:
        raise ValueError("distributed TN input shard labels are incompatible")
    label = int(next(iter(labels)))
    if label in output.labels:
        raise ValueError(
            "retained sharded input requires a correspondingly sharded output"
        )
    if not all(label in value.labels for value in inputs):
        raise ValueError(
            "contracted shard label must be shared by both operation inputs"
        )
    return "contracted_partial_reduce", label


def partition_tn_tensor_for_shard(
    tensor: torch.Tensor,
    labels: Sequence[int],
    *,
    shard_label: int,
    shard: DistributedTNShard,
) -> torch.Tensor:
    """Materialize only the contiguous range assigned to one input owner."""

    normalized = tuple(int(label) for label in labels)
    label = int(shard_label)
    if label not in normalized:
        raise ValueError("TN input shard label is absent from tensor labels")
    axis = normalized.index(label)
    if shard.stop > int(tensor.shape[axis]):
        raise ValueError("TN input shard range exceeds tensor extent")
    local = tensor.narrow(axis, shard.start, shard.stop - shard.start)
    expected = list(tensor.shape)
    expected[axis] = shard.stop - shard.start
    if tuple(local.shape) != tuple(expected):
        raise RuntimeError("TN input partition produced an invalid local shape")
    return local


def execute_pre_sharded_pair_contraction(
    left_local: torch.Tensor,
    left_labels: Sequence[int],
    right_local: torch.Tensor,
    right_labels: Sequence[int],
    output_labels: Sequence[int],
    *,
    mode: Literal["retained_output_shard", "contracted_partial_reduce"],
    shard_label: int,
    shard: DistributedTNShard,
    rank: int | None = None,
    world_size: int | None = None,
    process_group: Any | None = None,
) -> DistributedTNShardedPairResult:
    """Execute one pair using inputs already localized to this rank's shard."""

    actual_world_size = (
        int(dist.get_world_size(group=process_group))
        if dist.is_initialized()
        else int(1 if world_size is None else world_size)
    )
    actual_rank = (
        int(dist.get_rank(group=process_group))
        if dist.is_initialized() and rank is None
        else int(0 if rank is None else rank)
    )
    if not 0 <= actual_rank < actual_world_size:
        raise ValueError("pre-sharded TN rank is outside world size")
    if shard.rank != actual_rank:
        raise ValueError("pre-sharded TN input does not belong to the executing rank")
    if actual_world_size > 1 and not dist.is_initialized():
        raise RuntimeError(
            "multi-rank pre-sharded TN execution requires torch.distributed"
        )
    label = int(shard_label)
    if mode == "retained_output_shard":
        _validate_local_label_extent(left_local, left_labels, label=label, shard=shard)
        _validate_local_label_extent(
            right_local, right_labels, label=label, shard=shard
        )
        value = _einsum_pair_by_labels(
            left_local,
            tuple(int(item) for item in left_labels),
            right_local,
            tuple(int(item) for item in right_labels),
            tuple(int(item) for item in output_labels),
        )
        return DistributedTNShardedPairResult(
            value=value,
            mode=mode,
            rank=actual_rank,
            world_size=actual_world_size,
            shard_label=label,
            shard=shard,
            collective_count=0,
            collective_bytes=0,
        )
    if mode != "contracted_partial_reduce":
        raise ValueError("unsupported pre-sharded TN pair mode")
    if label in tuple(int(item) for item in output_labels):
        raise ValueError("contracted pre-sharded label cannot remain in the output")
    _validate_local_label_extent(left_local, left_labels, label=label, shard=shard)
    _validate_local_label_extent(right_local, right_labels, label=label, shard=shard)
    value = _einsum_pair_by_labels(
        left_local,
        tuple(int(item) for item in left_labels),
        right_local,
        tuple(int(item) for item in right_labels),
        tuple(int(item) for item in output_labels),
    )
    collective_bytes = 0
    collective_count = 0
    if actual_world_size > 1:
        dist.all_reduce(value, op=dist.ReduceOp.SUM, group=process_group)
        collective_count = 1
        collective_bytes = int(value.numel()) * int(value.element_size())
    return DistributedTNShardedPairResult(
        value=value,
        mode=mode,
        rank=actual_rank,
        world_size=actual_world_size,
        shard_label=label,
        shard=shard,
        collective_count=collective_count,
        collective_bytes=collective_bytes,
        full_output_materialized=mode == "contracted_partial_reduce",
    )


def contract_pair_for_output_shard(
    left: torch.Tensor,
    left_labels: Sequence[int],
    right: torch.Tensor,
    right_labels: Sequence[int],
    output_labels: Sequence[int],
    *,
    shard_label: int,
    shard: DistributedTNShard,
) -> torch.Tensor:
    """Contract one retained-label range without materializing the full output."""

    normalized_output = tuple(int(label) for label in output_labels)
    label = int(shard_label)
    if label not in normalized_output:
        raise ValueError("output-shard label must be retained by the contraction")
    sliced_left, sliced_left_labels = _slice_retained_label(
        left, left_labels, label=label, shard=shard
    )
    sliced_right, sliced_right_labels = _slice_retained_label(
        right, right_labels, label=label, shard=shard
    )
    if label not in sliced_left_labels and label not in sliced_right_labels:
        raise ValueError("output-shard label is absent from both contraction inputs")
    result = _einsum_pair_by_labels(
        sliced_left,
        sliced_left_labels,
        sliced_right,
        sliced_right_labels,
        normalized_output,
    )
    output_axis = normalized_output.index(label)
    if int(result.shape[output_axis]) != shard.stop - shard.start:
        raise RuntimeError("output-shard contraction produced an invalid local extent")
    return result


def contract_pair_for_contracted_shard(
    left: torch.Tensor,
    left_labels: Sequence[int],
    right: torch.Tensor,
    right_labels: Sequence[int],
    output_labels: Sequence[int],
    *,
    contracted_label: int,
    shard: DistributedTNShard,
) -> torch.Tensor:
    """Return one partial contraction over a sliced contracted-label range."""

    normalized_left = tuple(int(label) for label in left_labels)
    normalized_right = tuple(int(label) for label in right_labels)
    normalized_output = tuple(int(label) for label in output_labels)
    label = int(contracted_label)
    if label in normalized_output:
        raise ValueError("contracted-shard label cannot remain in the output")
    if label not in normalized_left or label not in normalized_right:
        raise ValueError("contracted-shard label must be shared by both inputs")
    left_axis = normalized_left.index(label)
    right_axis = normalized_right.index(label)
    if int(left.shape[left_axis]) != int(right.shape[right_axis]):
        raise ValueError("contracted-shard input extents do not match")
    if shard.stop > int(left.shape[left_axis]):
        raise ValueError("contracted-shard range exceeds the input extent")
    local_left = left.narrow(left_axis, shard.start, shard.stop - shard.start)
    local_right = right.narrow(right_axis, shard.start, shard.stop - shard.start)
    return _einsum_pair_by_labels(
        local_left,
        normalized_left,
        local_right,
        normalized_right,
        normalized_output,
    )


def combine_output_shards(
    partials: Sequence[torch.Tensor],
    output_labels: Sequence[int],
    *,
    shard_label: int,
) -> torch.Tensor:
    """Reconstruct a small validation output from ordered retained-label shards."""

    if not partials:
        raise ValueError("output-shard combination requires partial tensors")
    labels = tuple(int(label) for label in output_labels)
    label = int(shard_label)
    if label not in labels:
        raise ValueError("output-shard label is absent from output labels")
    return torch.cat(tuple(partials), dim=labels.index(label))


def combine_contracted_shards(
    partials: Sequence[torch.Tensor],
) -> torch.Tensor:
    """Sum local contracted-label partials like a distributed all-reduce."""

    if not partials:
        raise ValueError("contracted-shard combination requires partial tensors")
    result = partials[0]
    for partial in partials[1:]:
        if tuple(partial.shape) != tuple(result.shape):
            raise ValueError("contracted-shard partial shapes do not match")
        result = result + partial
    return result


def _slice_retained_label(
    tensor: torch.Tensor,
    labels: Sequence[int],
    *,
    label: int,
    shard: DistributedTNShard,
) -> tuple[torch.Tensor, tuple[int, ...]]:
    normalized = tuple(int(value) for value in labels)
    if label not in normalized:
        return tensor, normalized
    axis = normalized.index(label)
    if shard.stop > int(tensor.shape[axis]):
        raise ValueError("output-shard range exceeds the input extent")
    return (
        tensor.narrow(axis, shard.start, shard.stop - shard.start),
        normalized,
    )


def _validate_local_label_extent(
    tensor: torch.Tensor,
    labels: Sequence[int],
    *,
    label: int,
    shard: DistributedTNShard,
) -> None:
    normalized = tuple(int(item) for item in labels)
    if label not in normalized:
        return
    axis = normalized.index(label)
    expected = shard.stop - shard.start
    if int(tensor.shape[axis]) != expected:
        raise ValueError(
            "pre-sharded TN input extent does not match the assigned shard"
        )


__all__ = (
    "DistributedTNShardedPairResult",
    "combine_contracted_shards",
    "combine_output_shards",
    "contract_pair_for_contracted_shard",
    "contract_pair_for_output_shard",
    "execute_pre_sharded_pair_contraction",
    "partition_tn_tensor_for_shard",
    "select_sharded_pair_mode",
)

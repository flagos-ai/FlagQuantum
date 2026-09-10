"""Deterministic all-to-all plans between distributed TN shard layouts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Iterable

import torch
import torch.distributed as dist

from .distributed_dag import DistributedTNValueLayout
from .multi_axis_sharding import DistributedTNMultiAxisLayout

TN_REDISTRIBUTION_VERSION = "flagquantum.distributed_tn_redistribution.v1"
TN_MULTI_AXIS_REDISTRIBUTION_VERSION = (
    "flagquantum.distributed_tn_multi_axis_redistribution.v1"
)


@dataclass(frozen=True)
class DistributedTNRedistributionBlock:
    """One global hyper-rectangle transferred from a source to destination rank."""

    source_rank: int
    destination_rank: int
    global_slices: tuple[tuple[int, int], ...]
    element_count: int
    nbytes: int

    def __post_init__(self) -> None:
        if self.source_rank < 0 or self.destination_rank < 0:
            raise ValueError("TN redistribution ranks must be non-negative")
        if self.element_count <= 0 or self.nbytes <= 0:
            raise ValueError("TN redistribution blocks must be non-empty")
        if any(start < 0 or stop <= start for start, stop in self.global_slices):
            raise ValueError("TN redistribution slices must be non-empty")


@dataclass(frozen=True)
class DistributedTNRedistributionPlan:
    """Complete transfer partition between two layouts of one logical value."""

    version: str
    identity: str
    value_id: str
    shape: tuple[int, ...]
    source_shard_label: int
    destination_shard_label: int
    blocks: tuple[DistributedTNRedistributionBlock, ...]
    total_bytes: int
    planning_only: bool = True

    def summary(self) -> dict[str, object]:
        return {
            "version": self.version,
            "identity": self.identity,
            "value_id": self.value_id,
            "shape": self.shape,
            "source_shard_label": self.source_shard_label,
            "destination_shard_label": self.destination_shard_label,
            "blocks": tuple(asdict(block) for block in self.blocks),
            "block_count": len(self.blocks),
            "total_bytes": self.total_bytes,
            "self_transfer_bytes": sum(
                block.nbytes
                for block in self.blocks
                if block.source_rank == block.destination_rank
            ),
            "network_transfer_bytes": sum(
                block.nbytes
                for block in self.blocks
                if block.source_rank != block.destination_rank
            ),
            "planning_only": self.planning_only,
            "scalability_claim_allowed": False,
        }


@dataclass(frozen=True)
class DistributedTNRedistributionResult:
    """One rank's target shard and measured redistribution traffic."""

    local_tensor: torch.Tensor
    plan_identity: str
    rank: int
    world_size: int
    sent_bytes: int
    received_bytes: int
    self_transfer_bytes: int
    physical_message_count: int

    def summary(self) -> dict[str, object]:
        return {
            "plan_identity": self.plan_identity,
            "rank": self.rank,
            "world_size": self.world_size,
            "local_shape": tuple(int(dim) for dim in self.local_tensor.shape),
            "local_bytes": int(self.local_tensor.numel())
            * int(self.local_tensor.element_size()),
            "sent_bytes": self.sent_bytes,
            "received_bytes": self.received_bytes,
            "self_transfer_bytes": self.self_transfer_bytes,
            "physical_message_count": self.physical_message_count,
            "distribution_semantics": "rank_group_shard_redistribution",
            "scalability_claim_allowed": False,
            "scalability_blockers": ("multi_gpu_capacity_evidence_pending",),
        }


@dataclass(frozen=True)
class DistributedTNMultiAxisRedistributionPlan:
    """Intersection transfer plan between two Cartesian shard meshes."""

    version: str
    identity: str
    value_id: str
    shape: tuple[int, ...]
    source_layout_identity: str
    destination_layout_identity: str
    source_shard_labels: tuple[int, ...]
    destination_shard_labels: tuple[int, ...]
    blocks: tuple[DistributedTNRedistributionBlock, ...]
    total_bytes: int

    def summary(self) -> dict[str, object]:
        return {
            **asdict(self),
            "block_count": len(self.blocks),
            "self_transfer_bytes": sum(
                block.nbytes
                for block in self.blocks
                if block.source_rank == block.destination_rank
            ),
            "network_transfer_bytes": sum(
                block.nbytes
                for block in self.blocks
                if block.source_rank != block.destination_rank
            ),
            "distribution_semantics": "multi_axis_mesh_redistribution",
        }


def plan_multi_axis_tn_redistribution(
    source: DistributedTNMultiAxisLayout,
    destination: DistributedTNMultiAxisLayout,
) -> DistributedTNMultiAxisRedistributionPlan:
    """Intersect every source and destination Cartesian rank block."""

    _validate_multi_axis_compatible_layouts(source, destination)
    element_size = source.nbytes // _product(source.shape)
    blocks = []
    for source_shard in source.shards:
        for destination_shard in destination.shards:
            slices = tuple(
                (
                    max(source_start, destination_start),
                    min(source_stop, destination_stop),
                )
                for (source_start, source_stop), (
                    destination_start,
                    destination_stop,
                ) in zip(
                    source_shard.global_slices,
                    destination_shard.global_slices,
                )
            )
            if any(stop <= start for start, stop in slices):
                continue
            elements = _product(stop - start for start, stop in slices)
            blocks.append(
                DistributedTNRedistributionBlock(
                    source_rank=source_shard.rank,
                    destination_rank=destination_shard.rank,
                    global_slices=slices,
                    element_count=elements,
                    nbytes=elements * element_size,
                )
            )
    total_bytes = sum(block.nbytes for block in blocks)
    if total_bytes != source.nbytes:
        raise RuntimeError(
            "multi-axis TN redistribution does not partition the logical tensor"
        )
    payload = {
        "version": TN_MULTI_AXIS_REDISTRIBUTION_VERSION,
        "value_id": source.value_id,
        "shape": source.shape,
        "source_layout_identity": source.identity,
        "destination_layout_identity": destination.identity,
        "source_shard_labels": source.shard_labels,
        "destination_shard_labels": destination.shard_labels,
        "blocks": tuple(asdict(block) for block in blocks),
        "total_bytes": total_bytes,
    }
    return DistributedTNMultiAxisRedistributionPlan(
        version=TN_MULTI_AXIS_REDISTRIBUTION_VERSION,
        identity=_redistribution_identity(payload),
        value_id=source.value_id,
        shape=source.shape,
        source_layout_identity=source.identity,
        destination_layout_identity=destination.identity,
        source_shard_labels=source.shard_labels,
        destination_shard_labels=destination.shard_labels,
        blocks=tuple(blocks),
        total_bytes=total_bytes,
    )


def execute_multi_axis_tn_redistribution(
    local_source_tensor: torch.Tensor,
    source: DistributedTNMultiAxisLayout,
    destination: DistributedTNMultiAxisLayout,
    plan: DistributedTNMultiAxisRedistributionPlan,
    *,
    rank: int | None = None,
    process_group: Any | None = None,
) -> DistributedTNRedistributionResult:
    """Execute a Cartesian mesh change without materializing the logical tensor."""

    if not dist.is_initialized():
        raise RuntimeError(
            "multi-axis TN redistribution requires initialized torch.distributed"
        )
    actual_rank = int(dist.get_rank(group=process_group) if rank is None else rank)
    world_size = int(dist.get_world_size(group=process_group))
    if source.world_size != world_size or destination.world_size != world_size:
        raise RuntimeError("multi-axis redistribution world size mismatch")
    expected = plan_multi_axis_tn_redistribution(source, destination)
    if plan.identity != expected.identity:
        raise ValueError("multi-axis redistribution plan identity mismatch")
    source_shard = source.shards[actual_rank]
    if tuple(local_source_tensor.shape) != source_shard.local_shape:
        raise ValueError("multi-axis redistribution source local shape mismatch")

    received = []
    operations = []
    pending_sends = []
    sent_bytes = received_bytes = self_bytes = message_count = 0
    for block_index, block in enumerate(plan.blocks):
        if block.source_rank == actual_rank:
            local_slices = tuple(
                slice(
                    start - source_shard.global_slices[axis][0],
                    stop - source_shard.global_slices[axis][0],
                )
                for axis, (start, stop) in enumerate(block.global_slices)
            )
            packed = local_source_tensor[local_slices].contiguous()
            if int(packed.numel()) != block.element_count:
                raise RuntimeError("multi-axis redistribution packed size mismatch")
            if block.destination_rank == actual_rank:
                received.append((block, packed))
                self_bytes += block.nbytes
            else:
                pending_sends.append(packed)
                operations.append(
                    dist.P2POp(
                        dist.isend,
                        packed,
                        block.destination_rank,
                        process_group,
                        _multi_axis_redistribution_tag(block_index),
                    )
                )
                sent_bytes += block.nbytes
                message_count += 1
        elif block.destination_rank == actual_rank:
            shape = tuple(stop - start for start, stop in block.global_slices)
            buffer = torch.empty(
                shape,
                dtype=local_source_tensor.dtype,
                device=local_source_tensor.device,
            )
            received.append((block, buffer))
            operations.append(
                dist.P2POp(
                    dist.irecv,
                    buffer,
                    block.source_rank,
                    process_group,
                    _multi_axis_redistribution_tag(block_index),
                )
            )
            received_bytes += block.nbytes
            message_count += 1
    requests = dist.batch_isend_irecv(operations) if operations else ()
    for request in requests:
        request.wait()
    destination_shard = destination.shards[actual_rank]
    output = torch.empty(
        destination_shard.local_shape,
        dtype=local_source_tensor.dtype,
        device=local_source_tensor.device,
    )
    written = torch.zeros(
        destination_shard.local_shape,
        dtype=torch.bool,
        device=local_source_tensor.device,
    )
    for block, tensor in received:
        local_slices = tuple(
            slice(
                start - destination_shard.global_slices[axis][0],
                stop - destination_shard.global_slices[axis][0],
            )
            for axis, (start, stop) in enumerate(block.global_slices)
        )
        if bool(written[local_slices].any()):
            raise ValueError("multi-axis redistribution destination blocks overlap")
        output[local_slices] = tensor
        written[local_slices] = True
    if not bool(written.all()):
        raise ValueError("multi-axis redistribution destination is incomplete")
    return DistributedTNRedistributionResult(
        local_tensor=output,
        plan_identity=plan.identity,
        rank=actual_rank,
        world_size=world_size,
        sent_bytes=sent_bytes,
        received_bytes=received_bytes,
        self_transfer_bytes=self_bytes,
        physical_message_count=message_count,
    )


def plan_distributed_tn_redistribution(
    source: DistributedTNValueLayout,
    destination: DistributedTNValueLayout,
) -> DistributedTNRedistributionPlan:
    """Partition a logical tensor into source/destination shard intersections."""

    _validate_compatible_layouts(source, destination)
    source_axis = source.shard_axis
    destination_axis = destination.shard_axis
    source_label = source.shard_label
    destination_label = destination.shard_label
    if (
        source_axis is None
        or destination_axis is None
        or source_label is None
        or destination_label is None
    ):
        raise ValueError("TN redistribution requires shard axes and labels")
    source_axis = int(source_axis)
    destination_axis = int(destination_axis)
    source_label = int(source_label)
    destination_label = int(destination_label)
    element_size = source.nbytes // _product(source.shape)
    blocks = []
    for source_shard in source.shards:
        for destination_shard in destination.shards:
            slices = [(0, int(extent)) for extent in source.shape]
            if source_axis == destination_axis:
                start = max(source_shard.start, destination_shard.start)
                stop = min(source_shard.stop, destination_shard.stop)
                if stop <= start:
                    continue
                slices[source_axis] = (start, stop)
            else:
                slices[source_axis] = (source_shard.start, source_shard.stop)
                slices[destination_axis] = (
                    destination_shard.start,
                    destination_shard.stop,
                )
            element_count = _product(stop - start for start, stop in slices)
            blocks.append(
                DistributedTNRedistributionBlock(
                    source_rank=source_shard.rank,
                    destination_rank=destination_shard.rank,
                    global_slices=tuple(slices),
                    element_count=element_count,
                    nbytes=element_count * element_size,
                )
            )
    total_bytes = sum(block.nbytes for block in blocks)
    if total_bytes != source.nbytes:
        raise RuntimeError(
            "TN redistribution blocks do not partition the logical tensor exactly"
        )
    payload = {
        "version": TN_REDISTRIBUTION_VERSION,
        "value_id": source.value_id,
        "shape": source.shape,
        "source_shard_label": source_label,
        "destination_shard_label": destination_label,
        "blocks": tuple(asdict(block) for block in blocks),
        "total_bytes": total_bytes,
        "planning_only": True,
    }
    identity = hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode()
    ).hexdigest()
    return DistributedTNRedistributionPlan(
        version=TN_REDISTRIBUTION_VERSION,
        identity=identity,
        value_id=source.value_id,
        shape=source.shape,
        source_shard_label=source_label,
        destination_shard_label=destination_label,
        blocks=tuple(blocks),
        total_bytes=total_bytes,
    )


def pack_tn_redistribution_block(
    local_tensor: torch.Tensor,
    source: DistributedTNValueLayout,
    block: DistributedTNRedistributionBlock,
) -> torch.Tensor:
    """Extract one planned global block from its source-rank local tensor."""

    if source.semantics != "sharded":
        raise ValueError("TN redistribution packing requires a sharded source")
    source_shard = next(
        (shard for shard in source.shards if shard.rank == block.source_rank), None
    )
    if source_shard is None:
        raise ValueError("TN redistribution block source rank has no shard")
    if tuple(int(dim) for dim in local_tensor.shape) != source_shard.local_shape:
        raise ValueError("TN redistribution source local shape mismatch")
    slices = []
    for axis, (start, stop) in enumerate(block.global_slices):
        if axis == source.shard_axis:
            start -= source_shard.start
            stop -= source_shard.start
        if start < 0 or stop > int(local_tensor.shape[axis]) or stop <= start:
            raise ValueError("TN redistribution block is outside the source shard")
        slices.append(slice(start, stop))
    packed = local_tensor[tuple(slices)].contiguous()
    if int(packed.numel()) != block.element_count:
        raise RuntimeError("TN redistribution packed block element count mismatch")
    if int(packed.numel()) * int(packed.element_size()) != block.nbytes:
        raise RuntimeError("TN redistribution packed block byte count mismatch")
    return packed


def assemble_tn_redistribution_shard(
    destination: DistributedTNValueLayout,
    *,
    destination_rank: int,
    received_blocks: Iterable[tuple[DistributedTNRedistributionBlock, torch.Tensor]],
) -> torch.Tensor:
    """Assemble all received global blocks into one destination-rank shard."""

    if destination.semantics != "sharded":
        raise ValueError("TN redistribution assembly requires a sharded destination")
    destination_shard = next(
        (shard for shard in destination.shards if shard.rank == destination_rank),
        None,
    )
    if destination_shard is None:
        raise ValueError("TN redistribution destination rank has no shard")
    received = tuple(received_blocks)
    if not received:
        raise ValueError("TN redistribution assembly requires received blocks")
    reference = received[0][1]
    output = torch.empty(
        destination_shard.local_shape,
        dtype=reference.dtype,
        device=reference.device,
    )
    written = torch.zeros(
        destination_shard.local_shape,
        dtype=torch.bool,
        device=reference.device,
    )
    for block, tensor in received:
        if block.destination_rank != destination_rank:
            raise ValueError("TN redistribution block targets a different rank")
        expected_shape = tuple(stop - start for start, stop in block.global_slices)
        if tuple(int(dim) for dim in tensor.shape) != expected_shape:
            raise ValueError("TN redistribution received block shape mismatch")
        slices = []
        for axis, (start, stop) in enumerate(block.global_slices):
            if axis == destination.shard_axis:
                start -= destination_shard.start
                stop -= destination_shard.start
            if start < 0 or stop > int(output.shape[axis]) or stop <= start:
                raise ValueError(
                    "TN redistribution block is outside the destination shard"
                )
            slices.append(slice(start, stop))
        local_slices = tuple(slices)
        if bool(written[local_slices].any()):
            raise ValueError("TN redistribution destination blocks overlap")
        output[local_slices] = tensor
        written[local_slices] = True
    if not bool(written.all()):
        raise ValueError("TN redistribution destination shard has uncovered elements")
    return output


def execute_distributed_tn_redistribution(
    local_source_tensor: torch.Tensor,
    source: DistributedTNValueLayout,
    destination: DistributedTNValueLayout,
    plan: DistributedTNRedistributionPlan,
    *,
    rank: int | None = None,
    process_group: Any | None = None,
) -> DistributedTNRedistributionResult:
    """Execute a planned all-to-all with batched point-to-point operations."""

    if not dist.is_initialized():
        raise RuntimeError("TN redistribution requires initialized torch.distributed")
    actual_rank = int(dist.get_rank(group=process_group)) if rank is None else int(rank)
    world_size = int(dist.get_world_size(group=process_group))
    expected_ranks = tuple(range(world_size))
    if (
        source.owner_ranks != expected_ranks
        or destination.owner_ranks != expected_ranks
    ):
        raise RuntimeError(
            "TN redistribution v1 requires every process-group rank to own one shard"
        )
    if actual_rank not in expected_ranks:
        raise ValueError("TN redistribution rank is outside the process group")
    expected_plan = plan_distributed_tn_redistribution(source, destination)
    if plan.identity != expected_plan.identity:
        raise ValueError("TN redistribution plan identity does not match layouts")
    source_shard = source.shards[actual_rank]
    if tuple(int(dim) for dim in local_source_tensor.shape) != source_shard.local_shape:
        raise ValueError("TN redistribution source local shape mismatch")

    received: list[tuple[DistributedTNRedistributionBlock, torch.Tensor]] = []
    operations = []
    pending_sends: list[torch.Tensor] = []
    sent_bytes = received_bytes = self_bytes = message_count = 0
    for block_index, block in enumerate(plan.blocks):
        if block.source_rank == actual_rank:
            packed = pack_tn_redistribution_block(local_source_tensor, source, block)
            if block.destination_rank == actual_rank:
                received.append((block, packed))
                self_bytes += block.nbytes
            else:
                pending_sends.append(packed)
                operations.append(
                    dist.P2POp(
                        dist.isend,
                        packed,
                        block.destination_rank,
                        process_group,
                        _redistribution_tag(block_index),
                    )
                )
                sent_bytes += block.nbytes
                message_count += 1
        elif block.destination_rank == actual_rank:
            shape = tuple(stop - start for start, stop in block.global_slices)
            buffer = torch.empty(
                shape,
                dtype=local_source_tensor.dtype,
                device=local_source_tensor.device,
            )
            received.append((block, buffer))
            operations.append(
                dist.P2POp(
                    dist.irecv,
                    buffer,
                    block.source_rank,
                    process_group,
                    _redistribution_tag(block_index),
                )
            )
            received_bytes += block.nbytes
            message_count += 1
    requests = dist.batch_isend_irecv(operations) if operations else ()
    for request in requests:
        request.wait()
    local_destination = assemble_tn_redistribution_shard(
        destination,
        destination_rank=actual_rank,
        received_blocks=received,
    )
    return DistributedTNRedistributionResult(
        local_tensor=local_destination,
        plan_identity=plan.identity,
        rank=actual_rank,
        world_size=world_size,
        sent_bytes=sent_bytes,
        received_bytes=received_bytes,
        self_transfer_bytes=self_bytes,
        physical_message_count=message_count,
    )


def _validate_compatible_layouts(
    source: DistributedTNValueLayout,
    destination: DistributedTNValueLayout,
) -> None:
    if source.semantics != "sharded" or destination.semantics != "sharded":
        raise ValueError("TN redistribution requires two sharded layouts")
    for name in ("value_id", "labels", "shape", "dtype", "nbytes"):
        if getattr(source, name) != getattr(destination, name):
            raise ValueError(f"TN redistribution layout {name} mismatch")


def _validate_multi_axis_compatible_layouts(
    source: DistributedTNMultiAxisLayout,
    destination: DistributedTNMultiAxisLayout,
) -> None:
    for name in ("value_id", "labels", "shape", "dtype", "nbytes"):
        if getattr(source, name) != getattr(destination, name):
            raise ValueError(f"multi-axis TN redistribution {name} mismatch")
    if source.world_size != destination.world_size:
        raise ValueError("multi-axis TN redistribution world size mismatch")
    if source.shard_labels == destination.shard_labels:
        raise ValueError("multi-axis TN redistribution requires different mesh labels")


def _redistribution_identity(payload: dict[str, object]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _product(values: Iterable[int]) -> int:
    result = 1
    for value in values:
        result *= int(value)
    return int(result)


def _redistribution_tag(block_index: int) -> int:
    return 7_000_000 + int(block_index)


def _multi_axis_redistribution_tag(block_index: int) -> int:
    return 8_000_000 + int(block_index)


__all__ = (
    "assemble_tn_redistribution_shard",
    "DistributedTNRedistributionBlock",
    "DistributedTNMultiAxisRedistributionPlan",
    "DistributedTNRedistributionPlan",
    "DistributedTNRedistributionResult",
    "TN_REDISTRIBUTION_VERSION",
    "plan_distributed_tn_redistribution",
    "pack_tn_redistribution_block",
    "execute_distributed_tn_redistribution",
    "execute_multi_axis_tn_redistribution",
    "plan_multi_axis_tn_redistribution",
)

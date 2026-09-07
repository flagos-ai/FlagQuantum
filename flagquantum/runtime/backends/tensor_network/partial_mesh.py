"""Reverse contractions on meshes with partitioned and replicated dimensions."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from itertools import product
from math import prod
from typing import Any, Sequence

import torch
import torch.distributed as dist

from ....simulation.tensor_network.stages import einsum_pair_by_labels
from .distributed_dag import DistributedTNValueLayout

TN_PARTIAL_MESH_VERSION = "flagquantum.distributed_tn_partial_mesh.v1"


@dataclass(frozen=True)
class DistributedTNPartialMeshLayout:
    """One logical tensor distributed on only the mesh labels it contains."""

    version: str
    identity: str
    value_id: str
    labels: tuple[int, ...]
    logical_shape: tuple[int, ...]
    mesh_labels: tuple[int, ...]
    mesh_shape: tuple[int, ...]
    partitioned_mesh_labels: tuple[int, ...]
    replicated_mesh_labels: tuple[int, ...]
    local_shape: tuple[int, ...]
    logical_nbytes: int
    local_nbytes: int
    replication_factor: int

    @property
    def world_size(self) -> int:
        return prod(self.mesh_shape)

    def summary(self) -> dict[str, Any]:
        return asdict(self) | {
            "world_size": self.world_size,
            "distribution_semantics": "partial_mesh_partition_with_replication",
        }


@dataclass(frozen=True)
class DistributedTNPartialMeshReverseResult:
    """Two local input cotangents plus subgroup communication evidence."""

    left_cotangent: torch.Tensor
    right_cotangent: torch.Tensor
    reduced_mesh_labels: tuple[int, ...]
    subgroup_collective_count: int
    subgroup_collective_bytes: int


@dataclass(frozen=True)
class DistributedTNPartialMeshForwardResult:
    """One local pair contraction plus reductions for eliminated mesh axes."""

    value: torch.Tensor
    reduced_mesh_labels: tuple[int, ...]
    subgroup_collective_count: int
    subgroup_collective_bytes: int


@dataclass(frozen=True)
class DistributedTNPartialMeshTransferBlock:
    """One canonical-source block delivered to one destination replica."""

    source_rank: int
    destination_rank: int
    global_slices: tuple[tuple[int, int], ...]
    element_count: int
    nbytes: int


@dataclass(frozen=True)
class DistributedTNPartialMeshRedistributionPlan:
    """Transfer schedule between meshes that may replicate absent labels."""

    version: str
    identity: str
    value_id: str
    source_layout_identity: str
    destination_layout_identity: str
    blocks: tuple[DistributedTNPartialMeshTransferBlock, ...]
    canonical_source_rank_count: int
    delivered_bytes: int
    network_bytes: int
    self_bytes: int

    def summary(self) -> dict[str, Any]:
        return asdict(self) | {
            "block_count": len(self.blocks),
            "distribution_semantics": "canonical_partial_mesh_redistribution",
        }


@dataclass(frozen=True)
class DistributedTNPartialMeshRedistributionResult:
    """One rank-local partial-mesh target and communication evidence."""

    local_tensor: torch.Tensor
    plan_identity: str
    sent_bytes: int
    received_bytes: int
    self_bytes: int
    physical_message_count: int


class DistributedTNMeshGroupCache:
    """Collectively-created reusable process groups for every mesh axis."""

    def __init__(
        self,
        *,
        mesh_labels: Sequence[int],
        mesh_shape: Sequence[int],
    ) -> None:
        if not dist.is_initialized():
            raise RuntimeError("partial mesh group cache requires torch.distributed")
        self.mesh_labels = tuple(int(label) for label in mesh_labels)
        self.mesh_shape = tuple(int(extent) for extent in mesh_shape)
        if (
            not self.mesh_labels
            or len(self.mesh_labels) != len(self.mesh_shape)
            or len(set(self.mesh_labels)) != len(self.mesh_labels)
            or prod(self.mesh_shape) != dist.get_world_size()
        ):
            raise ValueError("partial mesh group cache metadata is inconsistent")
        self._groups: dict[tuple[int, tuple[int, ...]], Any] = {}
        self._closed = False
        for axis in range(len(self.mesh_shape)):
            fixed_axes = tuple(
                index for index in range(len(self.mesh_shape)) if index != axis
            )
            fixed_ranges = tuple(range(self.mesh_shape[index]) for index in fixed_axes)
            for fixed in product(*fixed_ranges):
                members = []
                for coordinate in range(self.mesh_shape[axis]):
                    candidate = [0] * len(self.mesh_shape)
                    candidate[axis] = coordinate
                    for index, value in zip(fixed_axes, fixed):
                        candidate[index] = value
                    candidate_rank = 0
                    for value, extent in zip(candidate, self.mesh_shape):
                        candidate_rank = candidate_rank * extent + value
                    members.append(candidate_rank)
                self._groups[(axis, tuple(fixed))] = dist.new_group(ranks=members)

    @property
    def group_count(self) -> int:
        return len(self._groups)

    @property
    def closed(self) -> bool:
        return self._closed

    def group_for(self, *, label: int, rank: int | None = None) -> Any:
        if self._closed:
            raise RuntimeError("partial mesh group cache is closed")
        axis = self.mesh_labels.index(int(label))
        actual_rank = dist.get_rank() if rank is None else int(rank)
        coordinates = _rank_coordinates(actual_rank, self.mesh_shape)
        fixed = tuple(
            coordinate for index, coordinate in enumerate(coordinates) if index != axis
        )
        return self._groups[(axis, fixed)]

    def close(self) -> None:
        if self._closed:
            return
        for group in self._groups.values():
            dist.destroy_process_group(group)
        self._groups.clear()
        self._closed = True

    def __enter__(self) -> DistributedTNMeshGroupCache:
        if self._closed:
            raise RuntimeError("partial mesh group cache is closed")
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


def plan_partial_mesh_tn_layout(
    value: DistributedTNValueLayout,
    *,
    mesh_labels: Sequence[int],
    mesh_shape: Sequence[int],
) -> DistributedTNPartialMeshLayout:
    """Plan tensor partitioning while replicating absent mesh dimensions."""

    labels = tuple(int(label) for label in mesh_labels)
    shape = tuple(int(extent) for extent in mesh_shape)
    if not labels or len(labels) != len(shape) or len(set(labels)) != len(labels):
        raise ValueError("partial TN mesh metadata is inconsistent")
    if any(extent <= 1 for extent in shape):
        raise ValueError("partial TN mesh extents must exceed one")
    for label, extent in zip(labels, shape):
        if label in value.labels:
            axis = value.labels.index(label)
            if value.shape[axis] != extent:
                raise ValueError("partial TN mesh label extent mismatch")
    partitioned = tuple(label for label in labels if label in value.labels)
    replicated = tuple(label for label in labels if label not in value.labels)
    local_shape = tuple(
        1 if label in partitioned else extent
        for label, extent in zip(value.labels, value.shape)
    )
    divisor = prod(shape[labels.index(label)] for label in partitioned)
    replication = prod(shape[labels.index(label)] for label in replicated)
    payload = {
        "version": TN_PARTIAL_MESH_VERSION,
        "value_id": value.value_id,
        "labels": value.labels,
        "logical_shape": value.shape,
        "mesh_labels": labels,
        "mesh_shape": shape,
        "partitioned_mesh_labels": partitioned,
        "replicated_mesh_labels": replicated,
        "local_shape": local_shape,
        "logical_nbytes": value.nbytes,
        "local_nbytes": value.nbytes // divisor,
        "replication_factor": replication,
    }
    return DistributedTNPartialMeshLayout(
        **payload,
        identity=hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    )


def partition_tn_tensor_for_partial_mesh(
    tensor: torch.Tensor,
    layout: DistributedTNPartialMeshLayout,
    *,
    rank: int,
) -> torch.Tensor:
    """Slice present mesh labels and replicate over absent mesh coordinates."""

    if tuple(tensor.shape) != layout.logical_shape:
        raise ValueError("partial TN mesh logical tensor shape mismatch")
    coordinates = _rank_coordinates(rank, layout.mesh_shape)
    index = [slice(None)] * tensor.ndim
    for label, coordinate in zip(layout.mesh_labels, coordinates):
        if label in layout.labels:
            index[layout.labels.index(label)] = slice(coordinate, coordinate + 1)
    local = tensor[tuple(index)]
    if tuple(local.shape) != layout.local_shape:
        raise RuntimeError("partial TN mesh partition produced invalid shape")
    return local


def plan_partial_mesh_tn_redistribution(
    source: DistributedTNPartialMeshLayout,
    destination: DistributedTNPartialMeshLayout,
) -> DistributedTNPartialMeshRedistributionPlan:
    """Use one canonical owner per duplicate source slice."""

    for name in ("value_id", "labels", "logical_shape", "logical_nbytes"):
        if getattr(source, name) != getattr(destination, name):
            raise ValueError(f"partial mesh redistribution {name} mismatch")
    if source.world_size != destination.world_size:
        raise ValueError("partial mesh redistribution world size mismatch")
    source_slices = {
        rank: _partial_mesh_global_slices(source, rank)
        for rank in range(source.world_size)
    }
    canonical = {}
    for rank, slices in source_slices.items():
        canonical.setdefault(slices, rank)
    element_size = source.logical_nbytes // prod(source.logical_shape)
    blocks = []
    for destination_rank in range(destination.world_size):
        destination_slices = _partial_mesh_global_slices(destination, destination_rank)
        for source_slice, source_rank in sorted(
            canonical.items(), key=lambda item: item[1]
        ):
            intersection = tuple(
                (max(a, c), min(b, d))
                for (a, b), (c, d) in zip(source_slice, destination_slices)
            )
            if any(stop <= start for start, stop in intersection):
                continue
            elements = prod(stop - start for start, stop in intersection)
            blocks.append(
                DistributedTNPartialMeshTransferBlock(
                    source_rank=source_rank,
                    destination_rank=destination_rank,
                    global_slices=intersection,
                    element_count=elements,
                    nbytes=elements * element_size,
                )
            )
    delivered = sum(block.nbytes for block in blocks)
    expected = destination.logical_nbytes * destination.replication_factor
    if delivered != expected:
        raise RuntimeError(
            "partial mesh redistribution does not cover destination replicas"
        )
    payload = {
        "version": TN_PARTIAL_MESH_VERSION,
        "value_id": source.value_id,
        "source_layout_identity": source.identity,
        "destination_layout_identity": destination.identity,
        "blocks": tuple(asdict(block) for block in blocks),
        "canonical_source_rank_count": len(canonical),
        "delivered_bytes": delivered,
        "network_bytes": sum(
            block.nbytes
            for block in blocks
            if block.source_rank != block.destination_rank
        ),
        "self_bytes": sum(
            block.nbytes
            for block in blocks
            if block.source_rank == block.destination_rank
        ),
    }
    return DistributedTNPartialMeshRedistributionPlan(
        version=TN_PARTIAL_MESH_VERSION,
        identity=hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        value_id=source.value_id,
        source_layout_identity=source.identity,
        destination_layout_identity=destination.identity,
        blocks=tuple(blocks),
        canonical_source_rank_count=len(canonical),
        delivered_bytes=delivered,
        network_bytes=payload["network_bytes"],
        self_bytes=payload["self_bytes"],
    )


def execute_partial_mesh_tn_redistribution(
    local_source: torch.Tensor,
    source: DistributedTNPartialMeshLayout,
    destination: DistributedTNPartialMeshLayout,
    plan: DistributedTNPartialMeshRedistributionPlan,
    *,
    process_group: Any | None = None,
) -> DistributedTNPartialMeshRedistributionResult:
    """Redistribute canonical source blocks and assemble destination replicas."""

    if not dist.is_initialized():
        raise RuntimeError("partial mesh redistribution requires torch.distributed")
    expected = plan_partial_mesh_tn_redistribution(source, destination)
    if plan.identity != expected.identity:
        raise ValueError("partial mesh redistribution plan identity mismatch")
    rank = dist.get_rank(group=process_group)
    if tuple(local_source.shape) != source.local_shape:
        raise ValueError("partial mesh redistribution local source shape mismatch")
    source_slices = _partial_mesh_global_slices(source, rank)
    destination_slices = _partial_mesh_global_slices(destination, rank)
    received = []
    operations = []
    pending_sends = []
    sent = received_bytes = self_bytes = messages = 0
    for block_index, block in enumerate(plan.blocks):
        if block.source_rank == rank:
            local_index = tuple(
                slice(start - source_slices[axis][0], stop - source_slices[axis][0])
                for axis, (start, stop) in enumerate(block.global_slices)
            )
            packed = local_source[local_index].contiguous()
            if int(packed.numel()) != block.element_count:
                raise RuntimeError("partial mesh packed block size mismatch")
            if block.destination_rank == rank:
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
                        9_000_000 + block_index,
                    )
                )
                sent += block.nbytes
                messages += 1
        elif block.destination_rank == rank:
            shape = tuple(stop - start for start, stop in block.global_slices)
            buffer = torch.empty(
                shape, dtype=local_source.dtype, device=local_source.device
            )
            received.append((block, buffer))
            operations.append(
                dist.P2POp(
                    dist.irecv,
                    buffer,
                    block.source_rank,
                    process_group,
                    9_000_000 + block_index,
                )
            )
            received_bytes += block.nbytes
            messages += 1
    requests = dist.batch_isend_irecv(operations) if operations else ()
    for request in requests:
        request.wait()
    output = torch.empty(
        destination.local_shape,
        dtype=local_source.dtype,
        device=local_source.device,
    )
    written = torch.zeros(
        destination.local_shape, dtype=torch.bool, device=local_source.device
    )
    for block, tensor in received:
        local_index = tuple(
            slice(
                start - destination_slices[axis][0],
                stop - destination_slices[axis][0],
            )
            for axis, (start, stop) in enumerate(block.global_slices)
        )
        if bool(written[local_index].any()):
            raise ValueError("partial mesh destination blocks overlap")
        output[local_index] = tensor
        written[local_index] = True
    if not bool(written.all()):
        raise ValueError("partial mesh destination has uncovered elements")
    return DistributedTNPartialMeshRedistributionResult(
        local_tensor=output,
        plan_identity=plan.identity,
        sent_bytes=sent,
        received_bytes=received_bytes,
        self_bytes=self_bytes,
        physical_message_count=messages,
    )


def execute_partial_mesh_reverse_pair(
    output_cotangent: torch.Tensor,
    output_layout: DistributedTNPartialMeshLayout,
    left: torch.Tensor,
    left_layout: DistributedTNPartialMeshLayout,
    right: torch.Tensor,
    right_layout: DistributedTNPartialMeshLayout,
    *,
    process_group: Any | None = None,
    group_cache: DistributedTNMeshGroupCache | None = None,
) -> DistributedTNPartialMeshReverseResult:
    """Execute one explicit pair VJP and reduce only lost mesh coordinates."""

    if not dist.is_initialized():
        raise RuntimeError("partial mesh reverse requires torch.distributed")
    layouts = (output_layout, left_layout, right_layout)
    if any(
        layout.mesh_labels != output_layout.mesh_labels
        or layout.mesh_shape != output_layout.mesh_shape
        for layout in layouts[1:]
    ):
        raise ValueError("partial mesh reverse layouts use different meshes")
    world_size = dist.get_world_size(group=process_group)
    if world_size != output_layout.world_size:
        raise RuntimeError("partial mesh reverse world size mismatch")
    if group_cache is not None and (
        group_cache.mesh_labels != output_layout.mesh_labels
        or group_cache.mesh_shape != output_layout.mesh_shape
        or group_cache.closed
    ):
        raise ValueError("partial mesh reverse group cache mismatches the mesh")
    for tensor, layout in zip((output_cotangent, left, right), layouts):
        if tuple(tensor.shape) != layout.local_shape:
            raise ValueError("partial mesh reverse local tensor shape mismatch")
    left_cotangent = einsum_pair_by_labels(
        output_cotangent,
        output_layout.labels,
        right.conj(),
        right_layout.labels,
        left_layout.labels,
    ).contiguous()
    right_cotangent = einsum_pair_by_labels(
        left.conj(),
        left_layout.labels,
        output_cotangent,
        output_layout.labels,
        right_layout.labels,
    ).contiguous()
    left_reduce = tuple(
        label
        for label in output_layout.mesh_labels
        if label not in left_layout.labels
        and (label in output_layout.labels or label in right_layout.labels)
    )
    right_reduce = tuple(
        label
        for label in output_layout.mesh_labels
        if label not in right_layout.labels
        and (label in left_layout.labels or label in output_layout.labels)
    )
    collective_count = 0
    collective_bytes = 0
    for tensor, reduce_labels in (
        (left_cotangent, left_reduce),
        (right_cotangent, right_reduce),
    ):
        for label in reduce_labels:
            group = (
                group_cache.group_for(label=label)
                if group_cache is not None
                else _mesh_axis_group(
                    output_layout.mesh_labels,
                    output_layout.mesh_shape,
                    label=label,
                    process_group=process_group,
                )
            )
            dist.all_reduce(tensor, group=group)
            collective_count += 1
            collective_bytes += (
                int(tensor.numel())
                * int(tensor.element_size())
                * (output_layout.mesh_shape[output_layout.mesh_labels.index(label)] - 1)
            )
    return DistributedTNPartialMeshReverseResult(
        left_cotangent=left_cotangent,
        right_cotangent=right_cotangent,
        reduced_mesh_labels=left_reduce + right_reduce,
        subgroup_collective_count=collective_count,
        subgroup_collective_bytes=collective_bytes,
    )


def execute_partial_mesh_forward_pair(
    left: torch.Tensor,
    left_layout: DistributedTNPartialMeshLayout,
    right: torch.Tensor,
    right_layout: DistributedTNPartialMeshLayout,
    output_layout: DistributedTNPartialMeshLayout,
    *,
    group_cache: DistributedTNMeshGroupCache,
) -> DistributedTNPartialMeshForwardResult:
    """Contract partial-mesh operands and reduce only eliminated mesh axes."""

    layouts = (left_layout, right_layout, output_layout)
    if any(
        layout.mesh_labels != output_layout.mesh_labels
        or layout.mesh_shape != output_layout.mesh_shape
        for layout in layouts[:2]
    ):
        raise ValueError("partial mesh forward layouts use different meshes")
    if (
        group_cache.mesh_labels != output_layout.mesh_labels
        or group_cache.mesh_shape != output_layout.mesh_shape
        or group_cache.closed
    ):
        raise ValueError("partial mesh forward group cache mismatch")
    if (
        tuple(left.shape) != left_layout.local_shape
        or tuple(right.shape) != right_layout.local_shape
    ):
        raise ValueError("partial mesh forward operand shape mismatch")
    value = einsum_pair_by_labels(
        left,
        left_layout.labels,
        right,
        right_layout.labels,
        output_layout.labels,
    ).contiguous()
    if tuple(value.shape) != output_layout.local_shape:
        raise RuntimeError("partial mesh forward produced invalid local shape")
    reduced = tuple(
        label
        for label in output_layout.mesh_labels
        if label not in output_layout.labels
        and (label in left_layout.labels or label in right_layout.labels)
    )
    collective_bytes = 0
    for label in reduced:
        dist.all_reduce(value, group=group_cache.group_for(label=label))
        collective_bytes += (
            int(value.numel())
            * int(value.element_size())
            * (output_layout.mesh_shape[output_layout.mesh_labels.index(label)] - 1)
        )
    return DistributedTNPartialMeshForwardResult(
        value=value,
        reduced_mesh_labels=reduced,
        subgroup_collective_count=len(reduced),
        subgroup_collective_bytes=collective_bytes,
    )


def _rank_coordinates(rank: int, mesh_shape: Sequence[int]) -> tuple[int, ...]:
    if not 0 <= int(rank) < prod(mesh_shape):
        raise ValueError("partial TN mesh rank is outside the mesh")
    remainder = int(rank)
    reversed_coordinates = []
    for extent in reversed(mesh_shape):
        reversed_coordinates.append(remainder % extent)
        remainder //= extent
    return tuple(reversed(reversed_coordinates))


def _partial_mesh_global_slices(
    layout: DistributedTNPartialMeshLayout,
    rank: int,
) -> tuple[tuple[int, int], ...]:
    coordinates = _rank_coordinates(rank, layout.mesh_shape)
    slices = [(0, extent) for extent in layout.logical_shape]
    for label, coordinate in zip(layout.mesh_labels, coordinates):
        if label in layout.labels:
            axis = layout.labels.index(label)
            slices[axis] = (coordinate, coordinate + 1)
    return tuple(slices)


def _mesh_axis_group(
    mesh_labels: tuple[int, ...],
    mesh_shape: tuple[int, ...],
    *,
    label: int,
    process_group: Any | None,
) -> Any:
    axis = mesh_labels.index(label)
    rank = dist.get_rank(group=process_group)
    selected_group = None
    fixed_axes = tuple(index for index in range(len(mesh_shape)) if index != axis)
    fixed_ranges = tuple(range(mesh_shape[index]) for index in fixed_axes)
    for fixed in product(*fixed_ranges):
        members = []
        for coordinate in range(mesh_shape[axis]):
            candidate = [0] * len(mesh_shape)
            candidate[axis] = coordinate
            for index, value in zip(fixed_axes, fixed):
                candidate[index] = value
            candidate_rank = 0
            for value, extent in zip(candidate, mesh_shape):
                candidate_rank = candidate_rank * extent + value
            members.append(candidate_rank)
        group = dist.new_group(ranks=members)
        if rank in members:
            selected_group = group
    if selected_group is None:
        raise RuntimeError("partial TN mesh failed to select a subgroup")
    return selected_group


__all__ = (
    "DistributedTNMeshGroupCache",
    "DistributedTNPartialMeshLayout",
    "DistributedTNPartialMeshForwardResult",
    "DistributedTNPartialMeshRedistributionPlan",
    "DistributedTNPartialMeshRedistributionResult",
    "DistributedTNPartialMeshReverseResult",
    "DistributedTNPartialMeshTransferBlock",
    "execute_partial_mesh_tn_redistribution",
    "execute_partial_mesh_forward_pair",
    "execute_partial_mesh_reverse_pair",
    "partition_tn_tensor_for_partial_mesh",
    "plan_partial_mesh_tn_layout",
    "plan_partial_mesh_tn_redistribution",
)

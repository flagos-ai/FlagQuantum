"""Multi-label rank coordinates for binary tensor-network bonds."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from itertools import combinations
from math import log2, prod
from typing import Any, Mapping, Sequence

import torch
import torch.distributed as dist

from ....simulation.tensor_stages import einsum_pair_by_labels_with_fallback
from .distributed_dag import (
    DistributedTNContractionDAG,
    DistributedTNValueLayout,
)

TN_MULTI_AXIS_VERSION = "flagquantum.distributed_tn_multi_axis.v1"


@dataclass(frozen=True)
class DistributedTNMultiAxisShard:
    """One Cartesian tensor block assigned to a rank."""

    rank: int
    coordinates: tuple[int, ...]
    global_slices: tuple[tuple[int, int], ...]
    local_shape: tuple[int, ...]
    nbytes: int


@dataclass(frozen=True)
class DistributedTNMultiAxisLayout:
    """Cartesian product sharding over multiple logical TN labels."""

    version: str
    identity: str
    value_id: str
    labels: tuple[int, ...]
    shape: tuple[int, ...]
    dtype: str
    nbytes: int
    shard_labels: tuple[int, ...]
    shard_axes: tuple[int, ...]
    mesh_shape: tuple[int, ...]
    shards: tuple[DistributedTNMultiAxisShard, ...]

    @property
    def world_size(self) -> int:
        return prod(self.mesh_shape)

    def summary(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "identity": self.identity,
            "value_id": self.value_id,
            "labels": self.labels,
            "shape": self.shape,
            "dtype": self.dtype,
            "nbytes": self.nbytes,
            "shard_labels": self.shard_labels,
            "shard_axes": self.shard_axes,
            "mesh_shape": self.mesh_shape,
            "world_size": self.world_size,
            "shards": tuple(asdict(shard) for shard in self.shards),
            "distribution_semantics": "multi_label_cartesian_shard",
            "planning_only": True,
            "scalability_claim_allowed": False,
        }


@dataclass(frozen=True)
class DistributedTNMultiAxisContractionResult:
    """Rank-local multi-label partial and its reduced result."""

    value: torch.Tensor
    rank: int
    world_size: int
    shard_labels: tuple[int, ...]
    local_input_bytes: int
    collective_count: int
    collective_bytes: int
    high_rank_einsum_fallback_count: int
    full_input_materialized: bool = False
    full_state_materialization: bool = False


@dataclass(frozen=True)
class DistributedTNMultiAxisConeResult:
    """Evidence from a shard-preserving forward cone into one contraction."""

    value: torch.Tensor
    rank: int
    world_size: int
    target_operation_id: str
    shard_labels: tuple[int, ...]
    partitioned_input_value_ids: tuple[str, ...]
    partitioned_intermediate_value_ids: tuple[str, ...]
    target_local_input_bytes: int
    target_logical_input_bytes: int
    peak_live_local_tensor_bytes: int
    peak_live_logical_tensor_bytes: int
    released_value_count: int
    collective_count: int
    collective_bytes: int
    high_rank_einsum_fallback_count: int
    forward_tape: Mapping[str, torch.Tensor] | None = None
    full_target_inputs_materialized: bool = False
    full_state_materialization: bool = False
    silent_statevector_fallback: bool = False

    def summary(self) -> dict[str, Any]:
        return {
            "executor": "distributed_tn_multi_axis_cone_v1",
            "rank": self.rank,
            "world_size": self.world_size,
            "target_operation_id": self.target_operation_id,
            "shard_labels": self.shard_labels,
            "partitioned_input_value_ids": self.partitioned_input_value_ids,
            "partitioned_intermediate_value_ids": (
                self.partitioned_intermediate_value_ids
            ),
            "target_local_input_bytes": self.target_local_input_bytes,
            "target_logical_input_bytes": self.target_logical_input_bytes,
            "peak_live_local_tensor_bytes": self.peak_live_local_tensor_bytes,
            "peak_live_logical_tensor_bytes": self.peak_live_logical_tensor_bytes,
            "released_value_count": self.released_value_count,
            "collective_count": self.collective_count,
            "collective_bytes": self.collective_bytes,
            "high_rank_einsum_fallback_count": (self.high_rank_einsum_fallback_count),
            "forward_tape_retained": self.forward_tape is not None,
            "full_target_inputs_materialized": (self.full_target_inputs_materialized),
            "full_state_materialization": self.full_state_materialization,
            "silent_statevector_fallback": self.silent_statevector_fallback,
            "scalability_claim_allowed": False,
            "scalability_blockers": (
                "whole_dag_peak_memory_capacity_evidence_pending",
                "reverse_mode_pending",
            ),
        }


@dataclass(frozen=True)
class DistributedTNMultiAxisPeakPlan:
    """Shard labels selected to minimize predicted rank-local live bytes."""

    version: str
    identity: str
    dag_identity: str
    target_operation_id: str
    shard_labels: tuple[int, ...]
    mesh_shape: tuple[int, ...]
    predicted_peak_local_bytes: int
    predicted_peak_logical_bytes: int
    peak_reduction_bytes: int
    candidate_count: int
    peak_value_ids: tuple[str, ...]

    def summary(self) -> dict[str, Any]:
        return asdict(self)


def plan_multi_axis_tn_peak_sharding(
    dag: DistributedTNContractionDAG,
    *,
    target_operation_id: str | None = None,
    world_size: int | None = None,
) -> DistributedTNMultiAxisPeakPlan:
    """Choose contracted labels minimizing the predicted local lifecycle peak."""

    dag.validate()
    actual_world_size = int(dag.world_size if world_size is None else world_size)
    if actual_world_size <= 1:
        raise ValueError("multi-axis TN peak planning requires multiple ranks")
    target = (
        dag.operations[-1]
        if target_operation_id is None
        else next(
            (
                operation
                for operation in dag.operations
                if operation.operation_id == target_operation_id
            ),
            None,
        )
    )
    if target is None:
        raise ValueError("multi-axis TN peak target is absent from the DAG")
    values = {value.value_id: value for value in dag.values}
    left, right = (values[value_id] for value_id in target.input_value_ids)
    contracted = tuple(
        label
        for label in left.labels
        if label in right.labels and label not in target.output_labels
    )
    label_extents = {
        label: left.shape[left.labels.index(label)]
        for label in contracted
        if left.shape[left.labels.index(label)] > 1
    }
    contracted = tuple(label for label in contracted if label in label_extents)
    max_candidate_labels = min(
        len(contracted),
        int(log2(actual_world_size)),
    )
    candidates = tuple(
        candidate
        for count in range(1, max_candidate_labels + 1)
        for candidate in combinations(contracted, count)
        if prod(label_extents[label] for label in candidate) == actual_world_size
    )
    if not candidates:
        raise ValueError("multi-axis TN target has no mesh-compatible label set")

    ranked = []
    for candidate in candidates:
        simulation = _simulate_multi_axis_live_bytes(
            dag,
            target_operation_id=target.operation_id,
            shard_labels=candidate,
            label_extents=label_extents,
        )
        if simulation is None:
            continue
        local_peak, logical_peak, peak_ids = simulation
        ranked.append(
            (
                local_peak,
                -int(logical_peak - local_peak),
                candidate,
                logical_peak,
                peak_ids,
            )
        )
    if not ranked:
        raise ValueError("multi-axis TN target has no lifecycle-valid label set")
    local_peak, _, selected, logical_peak, peak_ids = min(ranked)
    mesh_shape = tuple(label_extents[label] for label in selected)
    payload = {
        "version": TN_MULTI_AXIS_VERSION,
        "dag_identity": dag.identity,
        "target_operation_id": target.operation_id,
        "shard_labels": tuple(selected),
        "mesh_shape": mesh_shape,
        "predicted_peak_local_bytes": int(local_peak),
        "predicted_peak_logical_bytes": int(logical_peak),
        "peak_reduction_bytes": int(logical_peak - local_peak),
        "candidate_count": len(ranked),
        "peak_value_ids": tuple(peak_ids),
    }
    identity = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return DistributedTNMultiAxisPeakPlan(identity=identity, **payload)


def _simulate_multi_axis_live_bytes(
    dag: DistributedTNContractionDAG,
    *,
    target_operation_id: str,
    shard_labels: Sequence[int],
    label_extents: Mapping[int, int],
) -> tuple[int, int, tuple[str, ...]] | None:
    """Return local/logical peaks, or None when a label contracts too early."""

    values = {value.value_id: value for value in dag.values}
    remaining = {value.value_id: 0 for value in dag.values}
    for operation in dag.operations:
        for value_id in operation.input_value_ids:
            remaining[value_id] += 1
    live = {value.value_id for value in dag.values if value.producer_id is None}

    local_bytes = {}
    for value_id, layout in values.items():
        divisor = prod(
            label_extents[label] for label in shard_labels if label in layout.labels
        )
        local_bytes[value_id] = layout.nbytes // divisor
    current_local = sum(local_bytes[value_id] for value_id in live)
    current_logical = sum(values[value_id].nbytes for value_id in live)
    peak_local = current_local
    peak_logical = current_logical
    peak_ids = tuple(sorted(live))
    for operation in dag.operations:
        if operation.operation_id == target_operation_id:
            break
        input_layouts = tuple(
            values[value_id] for value_id in operation.input_value_ids
        )
        active = {
            label
            for label in shard_labels
            if any(label in layout.labels for layout in input_layouts)
        }
        retained = {label for label in shard_labels if label in operation.output_labels}
        if not active.issubset(retained):
            return None
        live.add(operation.output_value_id)
        current_local += local_bytes[operation.output_value_id]
        current_logical += values[operation.output_value_id].nbytes
        if current_local > peak_local:
            peak_local = current_local
            peak_logical = current_logical
            peak_ids = tuple(sorted(live))
        for value_id in operation.input_value_ids:
            remaining[value_id] -= 1
            if remaining[value_id] == 0:
                live.remove(value_id)
                current_local -= local_bytes[value_id]
                current_logical -= values[value_id].nbytes
    return peak_local, peak_logical, peak_ids


def plan_multi_axis_tn_layout(
    layout: DistributedTNValueLayout,
    *,
    shard_labels: Sequence[int],
    world_size: int,
) -> DistributedTNMultiAxisLayout:
    """Assign Cartesian label coordinates to ranks without flattening data."""

    labels = tuple(int(label) for label in shard_labels)
    if not labels or len(set(labels)) != len(labels):
        raise ValueError("multi-axis TN shard labels must be unique and non-empty")
    if any(label not in layout.labels for label in labels):
        raise ValueError("multi-axis TN shard label is absent from the value")
    axes = tuple(layout.labels.index(label) for label in labels)
    mesh_shape = tuple(layout.shape[axis] for axis in axes)
    if prod(mesh_shape) != int(world_size):
        raise ValueError(
            "multi-axis TN mesh product must equal the distributed world size"
        )
    element_size = layout.nbytes // max(1, prod(layout.shape))
    shards = []
    for rank in range(world_size):
        remainder = rank
        coordinates_reversed = []
        for extent in reversed(mesh_shape):
            coordinates_reversed.append(remainder % extent)
            remainder //= extent
        coordinates = tuple(reversed(coordinates_reversed))
        slices = [(0, extent) for extent in layout.shape]
        for axis, coordinate in zip(axes, coordinates):
            slices[axis] = (coordinate, coordinate + 1)
        local_shape = tuple(stop - start for start, stop in slices)
        shards.append(
            DistributedTNMultiAxisShard(
                rank=rank,
                coordinates=coordinates,
                global_slices=tuple(slices),
                local_shape=local_shape,
                nbytes=prod(local_shape) * element_size,
            )
        )
    payload = {
        "version": TN_MULTI_AXIS_VERSION,
        "value_id": layout.value_id,
        "labels": layout.labels,
        "shape": layout.shape,
        "dtype": layout.dtype,
        "nbytes": layout.nbytes,
        "shard_labels": labels,
        "shard_axes": axes,
        "mesh_shape": mesh_shape,
        "shards": tuple(asdict(shard) for shard in shards),
    }
    identity = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return DistributedTNMultiAxisLayout(
        identity=identity,
        shards=tuple(shards),
        **{key: value for key, value in payload.items() if key != "shards"},
    )


def partition_tn_tensor_for_multi_axis_shard(
    tensor: torch.Tensor,
    layout: DistributedTNMultiAxisLayout,
    *,
    rank: int,
) -> torch.Tensor:
    """Return the Cartesian block assigned to one rank."""

    if tuple(tensor.shape) != layout.shape:
        raise ValueError("multi-axis TN logical tensor shape mismatch")
    if not 0 <= int(rank) < layout.world_size:
        raise ValueError("multi-axis TN rank is outside the mesh")
    shard = layout.shards[int(rank)]
    index = tuple(slice(start, stop) for start, stop in shard.global_slices)
    local = tensor[index]
    if tuple(local.shape) != shard.local_shape:
        raise RuntimeError("multi-axis TN partition produced an invalid shape")
    return local


def partition_tn_tensor_by_rank_coordinates(
    tensor: torch.Tensor,
    tensor_labels: Sequence[int],
    *,
    shard_labels: Sequence[int],
    mesh_shape: Sequence[int],
    rank: int,
) -> torch.Tensor:
    """Slice every mesh label present in a tensor using one rank coordinate."""

    labels = tuple(int(label) for label in tensor_labels)
    selected = tuple(int(label) for label in shard_labels)
    mesh = tuple(int(extent) for extent in mesh_shape)
    if len(selected) != len(mesh) or prod(mesh) <= 0:
        raise ValueError("multi-axis TN mesh metadata is inconsistent")
    if not 0 <= int(rank) < prod(mesh):
        raise ValueError("multi-axis TN rank is outside the mesh")
    remainder = int(rank)
    reversed_coordinates = []
    for extent in reversed(mesh):
        reversed_coordinates.append(remainder % extent)
        remainder //= extent
    coordinates = tuple(reversed(reversed_coordinates))
    index: list[slice] = [slice(None)] * tensor.ndim
    for label, extent, coordinate in zip(selected, mesh, coordinates):
        if label not in labels:
            continue
        axis = labels.index(label)
        if int(tensor.shape[axis]) != extent:
            raise ValueError("multi-axis TN tensor label extent mismatches mesh")
        index[axis] = slice(coordinate, coordinate + 1)
    return tensor[tuple(index)]


def execute_multi_axis_tn_target_cone(
    dag: DistributedTNContractionDAG,
    local_inputs: Mapping[str, torch.Tensor],
    *,
    target_operation_id: str,
    shard_labels: Sequence[int],
    process_group: Any | None = None,
    retain_tape: bool = False,
    retain_value_ids: Sequence[str] = (),
) -> DistributedTNMultiAxisConeResult:
    """Keep selected labels rank-local from raw tensors through a target pair."""

    dag.validate()
    if not dist.is_initialized():
        raise RuntimeError("multi-axis TN cone execution requires torch.distributed")
    rank = int(dist.get_rank(group=process_group))
    world_size = int(dist.get_world_size(group=process_group))
    if world_size != dag.world_size:
        raise RuntimeError("multi-axis TN cone world size mismatches the DAG")
    labels = tuple(int(label) for label in shard_labels)
    values = {value.value_id: value for value in dag.values}
    selected_retained = {str(value_id) for value_id in retain_value_ids}
    unknown_retained = selected_retained - set(values)
    if unknown_retained:
        raise ValueError(
            f"multi-axis retained tape has unknown values {tuple(sorted(unknown_retained))}"
        )
    target = next(
        (
            operation
            for operation in dag.operations
            if operation.operation_id == target_operation_id
        ),
        None,
    )
    if target is None:
        raise ValueError("multi-axis TN target operation is absent from the DAG")
    target_inputs = tuple(values[value_id] for value_id in target.input_value_ids)
    if any(label not in layout.labels for layout in target_inputs for label in labels):
        raise ValueError("all multi-axis labels must occur in both target inputs")
    mesh_shape = tuple(
        target_inputs[0].shape[target_inputs[0].labels.index(label)] for label in labels
    )
    if prod(mesh_shape) != world_size:
        raise ValueError("multi-axis target mesh product must equal world size")

    raw = tuple(value for value in dag.values if value.producer_id is None)
    expected = {value.value_id for value in raw}
    if {str(value_id) for value_id in local_inputs} != expected:
        raise ValueError("multi-axis TN cone requires exactly all raw DAG inputs")
    local_values: dict[str, torch.Tensor] = {}
    partitioned_inputs = []
    partitioned_intermediates = []
    high_rank_fallback_count = 0
    for layout in raw:
        tensor = local_inputs[layout.value_id]
        if tuple(tensor.shape) != layout.shape:
            raise ValueError("multi-axis TN raw input shape mismatch")
        local = partition_tn_tensor_by_rank_coordinates(
            tensor,
            layout.labels,
            shard_labels=labels,
            mesh_shape=mesh_shape,
            rank=rank,
        )
        if any(label in layout.labels for label in labels):
            partitioned_inputs.append(layout.value_id)
        local_values[layout.value_id] = local
    saved_tape = dict(local_values) if retain_tape or selected_retained else None
    remaining_uses = {value.value_id: 0 for value in dag.values}
    for operation in dag.operations:
        for value_id in operation.input_value_ids:
            remaining_uses[value_id] += 1
    peak_local = sum(
        tensor.numel() * tensor.element_size() for tensor in local_values.values()
    )
    peak_logical = sum(values[value_id].nbytes for value_id in local_values)
    released = 0

    for operation in dag.operations:
        if operation.operation_id == target_operation_id:
            break
        left_id, right_id = operation.input_value_ids
        left_layout, right_layout = values[left_id], values[right_id]
        active = {
            label
            for label in labels
            if label in left_layout.labels or label in right_layout.labels
        }
        retained = {label for label in labels if label in operation.output_labels}
        if not active.issubset(retained):
            raise RuntimeError(
                "multi-axis TN shard label contracts before the target operation"
            )
        output, used_fallback = einsum_pair_by_labels_with_fallback(
            local_values[left_id],
            left_layout.labels,
            local_values[right_id],
            right_layout.labels,
            operation.output_labels,
        )
        high_rank_fallback_count += int(used_fallback)
        local_values[operation.output_value_id] = output
        if saved_tape is not None and (
            retain_tape or operation.output_value_id in selected_retained
        ):
            saved_tape[operation.output_value_id] = output
        if retained:
            partitioned_intermediates.append(operation.output_value_id)
        peak_local = max(
            peak_local,
            sum(
                tensor.numel() * tensor.element_size()
                for tensor in local_values.values()
            ),
        )
        peak_logical = max(
            peak_logical,
            sum(values[value_id].nbytes for value_id in local_values),
        )
        for value_id in operation.input_value_ids:
            remaining_uses[value_id] -= 1
            if remaining_uses[value_id] == 0:
                del local_values[value_id]
                released += 1

    left_id, right_id = target.input_value_ids
    left_local = local_values[left_id]
    right_local = local_values[right_id]
    contracted = execute_multi_axis_contracted_pair(
        left_local,
        values[left_id].labels,
        right_local,
        values[right_id].labels,
        target.output_labels,
        shard_labels=labels,
        rank=rank,
        process_group=process_group,
    )
    local_values[target.output_value_id] = contracted.value
    if saved_tape is not None:
        saved_tape[target.output_value_id] = contracted.value
    return DistributedTNMultiAxisConeResult(
        value=contracted.value,
        rank=rank,
        world_size=world_size,
        target_operation_id=target.operation_id,
        shard_labels=labels,
        partitioned_input_value_ids=tuple(partitioned_inputs),
        partitioned_intermediate_value_ids=tuple(partitioned_intermediates),
        target_local_input_bytes=contracted.local_input_bytes,
        target_logical_input_bytes=sum(layout.nbytes for layout in target_inputs),
        peak_live_local_tensor_bytes=peak_local,
        peak_live_logical_tensor_bytes=peak_logical,
        released_value_count=released,
        collective_count=contracted.collective_count,
        collective_bytes=contracted.collective_bytes,
        high_rank_einsum_fallback_count=(
            high_rank_fallback_count + contracted.high_rank_einsum_fallback_count
        ),
        forward_tape=saved_tape,
    )


def execute_multi_axis_contracted_pair(
    left_local: torch.Tensor,
    left_labels: Sequence[int],
    right_local: torch.Tensor,
    right_labels: Sequence[int],
    output_labels: Sequence[int],
    *,
    shard_labels: Sequence[int],
    rank: int | None = None,
    process_group: Any | None = None,
) -> DistributedTNMultiAxisContractionResult:
    """Contract local Cartesian blocks and reduce fully contracted shard axes."""

    if not dist.is_initialized():
        raise RuntimeError("multi-axis TN contraction requires torch.distributed")
    actual_rank = int(dist.get_rank(group=process_group) if rank is None else rank)
    world_size = int(dist.get_world_size(group=process_group))
    labels = tuple(int(label) for label in shard_labels)
    left_set = {int(label) for label in left_labels}
    right_set = {int(label) for label in right_labels}
    output_set = {int(label) for label in output_labels}
    if any(
        label not in left_set or label not in right_set or label in output_set
        for label in labels
    ):
        raise ValueError(
            "multi-axis contracted labels must be shared inputs absent from output"
        )
    value, used_fallback = einsum_pair_by_labels_with_fallback(
        left_local,
        tuple(int(label) for label in left_labels),
        right_local,
        tuple(int(label) for label in right_labels),
        tuple(int(label) for label in output_labels),
    )
    dist.all_reduce(value, op=dist.ReduceOp.SUM, group=process_group)
    return DistributedTNMultiAxisContractionResult(
        value=value,
        rank=actual_rank,
        world_size=world_size,
        shard_labels=labels,
        local_input_bytes=(
            left_local.numel() * left_local.element_size()
            + right_local.numel() * right_local.element_size()
        ),
        collective_count=1,
        collective_bytes=value.numel() * value.element_size(),
        high_rank_einsum_fallback_count=int(used_fallback),
    )


__all__ = (
    "DistributedTNMultiAxisContractionResult",
    "DistributedTNMultiAxisConeResult",
    "DistributedTNMultiAxisLayout",
    "DistributedTNMultiAxisPeakPlan",
    "DistributedTNMultiAxisShard",
    "execute_multi_axis_contracted_pair",
    "execute_multi_axis_tn_target_cone",
    "partition_tn_tensor_by_rank_coordinates",
    "partition_tn_tensor_for_multi_axis_shard",
    "plan_multi_axis_tn_layout",
    "plan_multi_axis_tn_peak_sharding",
)

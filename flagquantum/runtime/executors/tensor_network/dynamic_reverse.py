"""Versioned execution of dependent reverse records on one partial mesh."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import asdict, dataclass
from math import prod
from typing import Any, Callable, Mapping, Sequence

import torch

from .distributed_dag import DistributedTNContractionDAG
from .partial_mesh import (
    DistributedTNMeshGroupCache,
    execute_partial_mesh_reverse_pair,
    execute_partial_mesh_tn_redistribution,
    plan_partial_mesh_tn_layout,
    plan_partial_mesh_tn_redistribution,
)
from .reverse_dag import (
    DistributedTNReverseDAG,
    DistributedTNReverseRecord,
)

TN_DYNAMIC_REVERSE_SEGMENT_VERSION = (
    "flagquantum.distributed_tn_dynamic_reverse_segment.v3"
)


@dataclass(frozen=True)
class DistributedTNDynamicReverseSegment:
    """A deterministic dependency subgraph executed under one destination mesh."""

    version: str
    identity: str
    forward_dag_identity: str
    reverse_dag_identity: str
    source_mesh_labels: tuple[int, ...]
    destination_mesh_labels: tuple[int, ...]
    mesh_shape: tuple[int, ...]
    selection_mode: str
    records: tuple[DistributedTNReverseRecord, ...]

    def validate(
        self,
        forward: DistributedTNContractionDAG,
        reverse: DistributedTNReverseDAG,
    ) -> None:
        reverse.validate(forward)
        if self.version != TN_DYNAMIC_REVERSE_SEGMENT_VERSION:
            raise ValueError("unsupported dynamic TN reverse segment version")
        if (
            self.forward_dag_identity != forward.identity
            or self.reverse_dag_identity != reverse.identity
        ):
            raise ValueError("dynamic reverse segment DAG identity mismatch")
        if (
            not self.records
            or len(self.source_mesh_labels) != len(self.mesh_shape)
            or len(self.destination_mesh_labels) != len(self.mesh_shape)
        ):
            raise ValueError("dynamic reverse segment mesh metadata is invalid")
        reverse_records = {record.reverse_id: record for record in reverse.records}
        expected = _reachable_contribution_counts(
            reverse, self.records[0].output_value_id
        )
        received: Counter[str] = Counter()
        ready = {self.records[0].output_value_id}
        for record in self.records:
            if reverse_records.get(record.reverse_id) != record:
                raise ValueError("dynamic reverse segment contains an unknown record")
            if record.output_value_id not in ready:
                raise ValueError(
                    "dynamic reverse segment record is outside the available frontier"
                )
            ready.remove(record.output_value_id)
            for value_id in record.input_value_ids:
                received[value_id] += 1
                if received[value_id] > expected[value_id]:
                    raise ValueError(
                        "dynamic reverse segment has excess cotangent contributions"
                    )
                if received[value_id] == expected[value_id]:
                    ready.add(value_id)
        if self.selection_mode != "largest_ready_frontier":
            raise ValueError("unsupported dynamic reverse segment selection mode")
        if self.identity != _segment_identity(self._payload()):
            raise ValueError("dynamic reverse segment identity mismatch")

    def _payload(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "forward_dag_identity": self.forward_dag_identity,
            "reverse_dag_identity": self.reverse_dag_identity,
            "source_mesh_labels": self.source_mesh_labels,
            "destination_mesh_labels": self.destination_mesh_labels,
            "mesh_shape": self.mesh_shape,
            "selection_mode": self.selection_mode,
            "records": tuple(asdict(record) for record in self.records),
        }

    def summary(self) -> dict[str, Any]:
        return self._payload() | {
            "identity": self.identity,
            "record_count": len(self.records),
            "execution_semantics": "frontier_partial_mesh_reverse_subgraph",
        }


@dataclass(frozen=True)
class DistributedTNDynamicReverseSegmentResult:
    """Destination-mesh frontier and measured segment lifecycle evidence."""

    cotangents: Mapping[str, torch.Tensor]
    segment_identity: str
    executed_reverse_ids: tuple[str, ...]
    subgroup_collective_count: int
    subgroup_collective_bytes: int
    redistribution_count: int
    redistribution_sent_bytes: int
    redistribution_received_bytes: int
    accumulated_cotangent_count: int
    released_forward_value_count: int
    peak_cached_forward_bytes: int
    ready_cotangent_ids: tuple[str, ...]
    pending_cotangent_contributions: Mapping[str, int]
    rank_consensus_validated: bool
    rank_tensor_preflight_validated: bool
    completed: bool
    next_record_index: int
    rematerialized_forward_value_count: int
    rematerialized_forward_bytes: int


@dataclass(frozen=True)
class DistributedTNParameterPullbackResult:
    """Replica-correct parameter gradients produced from a dynamic frontier."""

    gradients: tuple[torch.Tensor, ...]
    input_cotangent_count: int
    cotangent_redistribution_count: int
    cotangent_redistribution_sent_bytes: int
    cotangent_redistribution_received_bytes: int
    replica_normalized_input_count: int
    parameter_allreduce_count: int
    nonfinite_gradient_count: int


def plan_dynamic_tn_reverse_segment(
    forward: DistributedTNContractionDAG,
    reverse: DistributedTNReverseDAG,
    *,
    start_output_value_id: str,
    source_mesh_labels: Sequence[int],
    destination_mesh_labels: Sequence[int],
    mesh_shape: Sequence[int],
    max_records: int,
    max_tensor_rank: int = 8,
) -> DistributedTNDynamicReverseSegment:
    """Select a deterministic largest-byte traversal of the ready frontier."""

    reverse.validate(forward)
    if max_records <= 0 or max_tensor_rank <= 0:
        raise ValueError("dynamic reverse segment limits must be positive")
    source_labels = tuple(int(label) for label in source_mesh_labels)
    destination_labels = tuple(int(label) for label in destination_mesh_labels)
    shape = tuple(int(extent) for extent in mesh_shape)
    if (
        len(source_labels) != len(destination_labels)
        or len(source_labels) != len(shape)
        or len(set(source_labels)) != len(source_labels)
        or len(set(destination_labels)) != len(destination_labels)
    ):
        raise ValueError("dynamic reverse segment mesh metadata is inconsistent")
    by_output = {record.output_value_id: record for record in reverse.records}
    values = {value.value_id: value for value in forward.values}
    if str(start_output_value_id) not in by_output:
        raise ValueError("dynamic reverse segment start value has no reverse record")
    records = []
    start_value_id = str(start_output_value_id)
    expected = _reachable_contribution_counts(reverse, start_value_id)
    received: Counter[str] = Counter()
    frontier = {start_value_id}
    executed_outputs = set()
    while len(records) < max_records:
        candidates = [
            by_output[value_id]
            for value_id in frontier
            if value_id in by_output
            and value_id not in executed_outputs
            and max(
                len(by_output[value_id].output_labels),
                len(by_output[value_id].left_labels),
                len(by_output[value_id].right_labels),
            )
            <= max_tensor_rank
        ]
        if not candidates:
            break
        current = max(
            candidates,
            key=lambda record: (
                values[record.output_value_id].nbytes,
                record.output_value_id,
            ),
        )
        records.append(current)
        executed_outputs.add(current.output_value_id)
        frontier.remove(current.output_value_id)
        for value_id in current.input_value_ids:
            received[value_id] += 1
            if received[value_id] == expected[value_id]:
                frontier.add(value_id)
    if not records:
        raise ValueError("dynamic reverse segment has no executable records")
    payload = {
        "version": TN_DYNAMIC_REVERSE_SEGMENT_VERSION,
        "forward_dag_identity": forward.identity,
        "reverse_dag_identity": reverse.identity,
        "source_mesh_labels": source_labels,
        "destination_mesh_labels": destination_labels,
        "mesh_shape": shape,
        "selection_mode": "largest_ready_frontier",
        "records": tuple(asdict(record) for record in records),
    }
    segment = DistributedTNDynamicReverseSegment(
        version=TN_DYNAMIC_REVERSE_SEGMENT_VERSION,
        identity=_segment_identity(payload),
        forward_dag_identity=forward.identity,
        reverse_dag_identity=reverse.identity,
        source_mesh_labels=source_labels,
        destination_mesh_labels=destination_labels,
        mesh_shape=shape,
        selection_mode="largest_ready_frontier",
        records=tuple(records),
    )
    segment.validate(forward, reverse)
    return segment


def execute_dynamic_tn_reverse_segment(
    forward: DistributedTNContractionDAG,
    reverse: DistributedTNReverseDAG,
    segment: DistributedTNDynamicReverseSegment,
    source_forward_tape: Mapping[str, torch.Tensor],
    source_output_cotangent: torch.Tensor | None,
    *,
    group_cache: DistributedTNMeshGroupCache,
    resume_from: DistributedTNDynamicReverseSegmentResult | None = None,
    max_records: int | None = None,
    source_forward_provider: Callable[[str], torch.Tensor] | None = None,
) -> DistributedTNDynamicReverseSegmentResult:
    """Execute or resume a bounded dependency chunk on one destination mesh."""

    segment.validate(forward, reverse)
    if (
        group_cache.mesh_labels != segment.destination_mesh_labels
        or group_cache.mesh_shape != segment.mesh_shape
    ):
        raise ValueError("dynamic reverse segment group cache mismatch")
    if max_records is not None and max_records <= 0:
        raise ValueError("dynamic reverse chunk record limit must be positive")
    if resume_from is not None and (
        resume_from.segment_identity != segment.identity
        or resume_from.completed
        or resume_from.next_record_index != len(resume_from.executed_reverse_ids)
    ):
        raise ValueError("dynamic reverse resume checkpoint is invalid")
    if source_output_cotangent is None:
        if resume_from is None:
            raise ValueError("initial dynamic reverse execution requires a seed")
        device = next(iter(resume_from.cotangents.values())).device
    else:
        device = source_output_cotangent.device
    _validate_segment_rank_consensus(segment, device)
    _validate_resume_rank_consensus(resume_from, device)
    start_index = 0 if resume_from is None else resume_from.next_record_index
    stop_index = len(segment.records)
    if max_records is not None:
        stop_index = min(stop_index, start_index + max_records)
    active_records = segment.records[start_index:stop_index]
    values = {value.value_id: value for value in forward.values}
    required_forward_ids = {
        value_id for record in active_records for value_id in record.input_value_ids
    }
    _validate_rank_tensor_preflight(
        values,
        segment,
        required_forward_ids,
        source_forward_tape,
        source_output_cotangent,
        resume_from,
        allow_missing=source_forward_provider is not None,
    )
    redistribution_count = (
        0 if resume_from is None else resume_from.redistribution_count
    )
    sent = 0 if resume_from is None else resume_from.redistribution_sent_bytes
    received = 0 if resume_from is None else resume_from.redistribution_received_bytes
    rematerialized_count = (
        0 if resume_from is None else resume_from.rematerialized_forward_value_count
    )
    rematerialized_bytes = (
        0 if resume_from is None else resume_from.rematerialized_forward_bytes
    )
    forward_cache: dict[str, torch.Tensor] = {}
    forward_layouts = {}
    remaining = {value_id: 0 for value_id in required_forward_ids}
    for record in active_records:
        for value_id in record.input_value_ids:
            remaining[value_id] += 1

    def remesh(value_id: str, tensor: torch.Tensor):
        nonlocal redistribution_count, sent, received
        logical = values[value_id]
        source = plan_partial_mesh_tn_layout(
            logical,
            mesh_labels=segment.source_mesh_labels,
            mesh_shape=segment.mesh_shape,
        )
        destination = plan_partial_mesh_tn_layout(
            logical,
            mesh_labels=segment.destination_mesh_labels,
            mesh_shape=segment.mesh_shape,
        )
        if source.identity == destination.identity:
            return destination, tensor
        plan = plan_partial_mesh_tn_redistribution(source, destination)
        result = execute_partial_mesh_tn_redistribution(
            tensor, source, destination, plan
        )
        redistribution_count += 1
        sent += result.sent_bytes
        received += result.received_bytes
        return destination, result.local_tensor

    start = segment.records[0].output_value_id
    expected = _reachable_contribution_counts(reverse, start)
    if resume_from is None:
        assert source_output_cotangent is not None
        output_layout, output_cotangent = remesh(start, source_output_cotangent)
        cotangents = {start: output_cotangent}
        cotangent_layouts = {start: output_layout}
        executed = []
        collective_count = collective_bytes = accumulated = released = 0
        peak_cache = 0
        contribution_counts: Counter[str] = Counter()
        ready = {start}
    else:
        cotangents = dict(resume_from.cotangents)
        cotangent_layouts = {
            value_id: plan_partial_mesh_tn_layout(
                values[value_id],
                mesh_labels=segment.destination_mesh_labels,
                mesh_shape=segment.mesh_shape,
            )
            for value_id in cotangents
        }
        executed = list(resume_from.executed_reverse_ids)
        collective_count = resume_from.subgroup_collective_count
        collective_bytes = resume_from.subgroup_collective_bytes
        accumulated = resume_from.accumulated_cotangent_count
        released = resume_from.released_forward_value_count
        peak_cache = resume_from.peak_cached_forward_bytes
        ready = set(resume_from.ready_cotangent_ids)
        contribution_counts = Counter(
            {
                value_id: expected[value_id]
                - resume_from.pending_cotangent_contributions.get(value_id, 0)
                for value_id in cotangents
            }
        )
    for record in active_records:
        if record.output_value_id not in ready:
            raise RuntimeError(
                "dynamic reverse segment consumed an incomplete cotangent"
            )
        ready.remove(record.output_value_id)
        output_cot = cotangents.pop(record.output_value_id, None)
        output_layout = cotangent_layouts.pop(record.output_value_id, None)
        if output_cot is None or output_layout is None:
            raise RuntimeError("dynamic reverse segment dependency is unavailable")
        operand_tensors = []
        operand_layouts = []
        for value_id in record.input_value_ids:
            if value_id not in forward_cache:
                source_tensor = source_forward_tape.get(value_id)
                if source_tensor is None:
                    if source_forward_provider is None:
                        raise RuntimeError(
                            "dynamic reverse forward value is unavailable"
                        )
                    source_tensor = _collectively_provide_forward_value(
                        value_id,
                        values[value_id],
                        segment,
                        source_forward_provider,
                        device,
                    )
                    rematerialized_count += 1
                    rematerialized_bytes += values[value_id].nbytes
                layout, tensor = remesh(value_id, source_tensor)
                forward_cache[value_id] = tensor
                forward_layouts[value_id] = layout
            operand_tensors.append(forward_cache[value_id])
            operand_layouts.append(forward_layouts[value_id])
        peak_cache = max(
            peak_cache,
            sum(
                int(tensor.numel()) * int(tensor.element_size())
                for tensor in forward_cache.values()
            ),
        )
        pair = execute_partial_mesh_reverse_pair(
            output_cot,
            output_layout,
            operand_tensors[0],
            operand_layouts[0],
            operand_tensors[1],
            operand_layouts[1],
            group_cache=group_cache,
        )
        collective_count += pair.subgroup_collective_count
        collective_bytes += pair.subgroup_collective_bytes
        for value_id, contribution, layout in zip(
            record.input_value_ids,
            (pair.left_cotangent, pair.right_cotangent),
            operand_layouts,
        ):
            contribution_counts[value_id] += 1
            if contribution_counts[value_id] > expected[value_id]:
                raise RuntimeError(
                    "dynamic reverse segment received excess cotangent contributions"
                )
            previous = cotangents.get(value_id)
            cotangents[value_id] = (
                contribution if previous is None else previous + contribution
            )
            cotangent_layouts[value_id] = layout
            accumulated += int(previous is not None)
            if contribution_counts[value_id] == expected[value_id]:
                ready.add(value_id)
            remaining[value_id] -= 1
            if remaining[value_id] == 0:
                del forward_cache[value_id]
                del forward_layouts[value_id]
                released += 1
        executed.append(record.reverse_id)
    return DistributedTNDynamicReverseSegmentResult(
        cotangents=cotangents,
        segment_identity=segment.identity,
        executed_reverse_ids=tuple(executed),
        subgroup_collective_count=collective_count,
        subgroup_collective_bytes=collective_bytes,
        redistribution_count=redistribution_count,
        redistribution_sent_bytes=sent,
        redistribution_received_bytes=received,
        accumulated_cotangent_count=accumulated,
        released_forward_value_count=released,
        peak_cached_forward_bytes=peak_cache,
        ready_cotangent_ids=tuple(sorted(ready)),
        pending_cotangent_contributions={
            value_id: expected[value_id] - contribution_counts[value_id]
            for value_id in sorted(cotangents)
            if contribution_counts[value_id] < expected[value_id]
        },
        rank_consensus_validated=True,
        rank_tensor_preflight_validated=True,
        completed=stop_index == len(segment.records),
        next_record_index=stop_index,
        rematerialized_forward_value_count=rematerialized_count,
        rematerialized_forward_bytes=rematerialized_bytes,
    )


def execute_dynamic_tn_parameter_pullback(
    forward: DistributedTNContractionDAG,
    segment: DistributedTNDynamicReverseSegment,
    result: DistributedTNDynamicReverseSegmentResult,
    source_forward_tape: Mapping[str, torch.Tensor],
    parameters: Sequence[torch.Tensor],
    *,
    retain_graph: bool = False,
) -> DistributedTNParameterPullbackResult:
    """Pull a destination-mesh input frontier back to replicated parameters."""

    if not torch.distributed.is_initialized():
        raise RuntimeError("dynamic TN parameter pullback requires torch.distributed")
    if result.segment_identity != segment.identity:
        raise ValueError("dynamic TN parameter pullback segment identity mismatch")
    if not result.completed:
        raise ValueError("dynamic TN parameter pullback requires a complete segment")
    parameter_tuple = tuple(parameters)
    if not parameter_tuple:
        raise ValueError("dynamic TN parameter pullback requires parameters")
    values = {value.value_id: value for value in forward.values}
    input_ids = tuple(
        value.value_id for value in forward.values if value.producer_id is None
    )
    ready_ids = set(result.ready_cotangent_ids)
    missing = tuple(
        value_id
        for value_id in input_ids
        if value_id not in source_forward_tape
        or value_id not in result.cotangents
        or value_id not in ready_ids
    )
    if missing:
        raise ValueError(
            "dynamic TN parameter pullback frontier is missing inputs " f"{missing}"
        )

    nodes = []
    cotangents = []
    redistribution_count = sent = received = normalized = 0
    for value_id in input_ids:
        node = source_forward_tape[value_id]
        if not node.requires_grad:
            continue
        logical = values[value_id]
        destination = plan_partial_mesh_tn_layout(
            logical,
            mesh_labels=segment.destination_mesh_labels,
            mesh_shape=segment.mesh_shape,
        )
        source = plan_partial_mesh_tn_layout(
            logical,
            mesh_labels=segment.source_mesh_labels,
            mesh_shape=segment.mesh_shape,
        )
        plan = plan_partial_mesh_tn_redistribution(destination, source)
        remeshed = execute_partial_mesh_tn_redistribution(
            result.cotangents[value_id],
            destination,
            source,
            plan,
        )
        redistribution_count += 1
        sent += remeshed.sent_bytes
        received += remeshed.received_bytes
        cotangent = remeshed.local_tensor
        if source.replication_factor > 1:
            cotangent = cotangent / source.replication_factor
            normalized += 1
        nodes.append(node)
        cotangents.append(cotangent)
    if not nodes:
        raise ValueError("dynamic TN parameter pullback has no differentiable inputs")

    gradients = torch.autograd.grad(
        tuple(nodes),
        parameter_tuple,
        grad_outputs=tuple(cotangents),
        allow_unused=True,
        retain_graph=retain_graph,
    )
    materialized = tuple(
        torch.zeros_like(parameter) if gradient is None else gradient
        for parameter, gradient in zip(parameter_tuple, gradients)
    )
    for gradient in materialized:
        torch.distributed.all_reduce(gradient, op=torch.distributed.ReduceOp.SUM)
    return DistributedTNParameterPullbackResult(
        gradients=materialized,
        input_cotangent_count=len(nodes),
        cotangent_redistribution_count=redistribution_count,
        cotangent_redistribution_sent_bytes=sent,
        cotangent_redistribution_received_bytes=received,
        replica_normalized_input_count=normalized,
        parameter_allreduce_count=len(materialized),
        nonfinite_gradient_count=sum(
            int(not bool(torch.isfinite(gradient).all())) for gradient in materialized
        ),
    )


def _segment_identity(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _reachable_contribution_counts(
    reverse: DistributedTNReverseDAG,
    start_output_value_id: str,
) -> Counter[str]:
    """Count all parent contributions required before a cotangent is ready."""

    by_output = {record.output_value_id: record for record in reverse.records}
    pending = [str(start_output_value_id)]
    visited = set()
    counts: Counter[str] = Counter()
    while pending:
        output_value_id = pending.pop()
        if output_value_id in visited:
            continue
        visited.add(output_value_id)
        record = by_output.get(output_value_id)
        if record is None:
            continue
        for value_id in record.input_value_ids:
            counts[value_id] += 1
            if value_id in by_output:
                pending.append(value_id)
    return counts


def _validate_segment_rank_consensus(
    segment: DistributedTNDynamicReverseSegment,
    device: torch.device,
) -> None:
    """Fail collectively before data movement when ranks planned differently."""

    if not torch.distributed.is_initialized():
        raise RuntimeError("dynamic TN reverse requires torch.distributed")
    if torch.distributed.get_world_size() != prod(segment.mesh_shape):
        raise RuntimeError("dynamic TN reverse mesh does not match world size")
    identity = bytes.fromhex(segment.identity)
    local = torch.tensor(tuple(identity), dtype=torch.uint8, device=device)
    gathered = [
        torch.empty_like(local) for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather(gathered, local)
    if any(not torch.equal(candidate, local) for candidate in gathered):
        raise RuntimeError("dynamic TN reverse segment differs across ranks")


def _validate_rank_tensor_preflight(
    values: Mapping[str, Any],
    segment: DistributedTNDynamicReverseSegment,
    required_forward_ids: set[str],
    source_forward_tape: Mapping[str, torch.Tensor],
    source_output_cotangent: torch.Tensor | None,
    resume_from: DistributedTNDynamicReverseSegmentResult | None,
    *,
    allow_missing: bool,
) -> None:
    """Collectively reject missing or incompatible local tensors."""

    invalid = False
    for value_id in required_forward_ids:
        tensor = source_forward_tape.get(value_id)
        if tensor is None:
            invalid = invalid or not allow_missing
            continue
        layout = plan_partial_mesh_tn_layout(
            values[value_id],
            mesh_labels=segment.source_mesh_labels,
            mesh_shape=segment.mesh_shape,
        )
        invalid = invalid or (
            tuple(tensor.shape) != layout.local_shape
            or str(tensor.dtype) != values[value_id].dtype
            or tensor.device
            != (
                source_output_cotangent.device
                if source_output_cotangent is not None
                else next(iter(resume_from.cotangents.values())).device
            )
        )
    if resume_from is None:
        assert source_output_cotangent is not None
        start = segment.records[0].output_value_id
        start_layout = plan_partial_mesh_tn_layout(
            values[start],
            mesh_labels=segment.source_mesh_labels,
            mesh_shape=segment.mesh_shape,
        )
        invalid = invalid or (
            tuple(source_output_cotangent.shape) != start_layout.local_shape
            or str(source_output_cotangent.dtype) != values[start].dtype
        )
    else:
        for value_id, tensor in resume_from.cotangents.items():
            layout = plan_partial_mesh_tn_layout(
                values[value_id],
                mesh_labels=segment.destination_mesh_labels,
                mesh_shape=segment.mesh_shape,
            )
            invalid = invalid or tuple(tensor.shape) != layout.local_shape
    device = (
        source_output_cotangent.device
        if source_output_cotangent is not None
        else next(iter(resume_from.cotangents.values())).device
    )
    status = torch.tensor(
        int(invalid),
        dtype=torch.uint8,
        device=device,
    )
    torch.distributed.all_reduce(status, op=torch.distributed.ReduceOp.MAX)
    if int(status.item()) != 0:
        raise RuntimeError(
            "dynamic TN reverse tensor preflight failed on at least one rank"
        )


def _validate_resume_rank_consensus(
    resume_from: DistributedTNDynamicReverseSegmentResult | None,
    device: torch.device,
) -> None:
    """Require every rank to resume from the same logical frontier."""

    payload = (
        {"initial": True}
        if resume_from is None
        else {
            "segment_identity": resume_from.segment_identity,
            "next_record_index": resume_from.next_record_index,
            "executed_reverse_ids": resume_from.executed_reverse_ids,
            "ready_cotangent_ids": resume_from.ready_cotangent_ids,
            "pending_cotangent_contributions": dict(
                resume_from.pending_cotangent_contributions
            ),
            "cotangent_ids": tuple(sorted(resume_from.cotangents)),
        }
    )
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).digest()
    local = torch.tensor(tuple(digest), dtype=torch.uint8, device=device)
    gathered = [
        torch.empty_like(local) for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather(gathered, local)
    if any(not torch.equal(candidate, local) for candidate in gathered):
        raise RuntimeError("dynamic TN reverse checkpoint differs across ranks")


def _collectively_provide_forward_value(
    value_id: str,
    value: Any,
    segment: DistributedTNDynamicReverseSegment,
    provider: Callable[[str], torch.Tensor],
    device: torch.device,
) -> torch.Tensor:
    """Materialize one missing source value and fail consistently on all ranks."""

    tensor = None
    invalid = False
    try:
        candidate = provider(value_id)
        tensor = candidate if isinstance(candidate, torch.Tensor) else None
    except Exception:
        invalid = True
    layout = plan_partial_mesh_tn_layout(
        value,
        mesh_labels=segment.source_mesh_labels,
        mesh_shape=segment.mesh_shape,
    )
    invalid = invalid or tensor is None
    if tensor is not None:
        invalid = invalid or (
            tuple(tensor.shape) != layout.local_shape
            or str(tensor.dtype) != value.dtype
            or tensor.device != device
        )
    status = torch.tensor(int(invalid), dtype=torch.uint8, device=device)
    torch.distributed.all_reduce(status, op=torch.distributed.ReduceOp.MAX)
    if bool(status.item()):
        raise RuntimeError(f"dynamic TN forward provider failed for value {value_id!r}")
    assert tensor is not None
    return tensor


__all__ = (
    "DistributedTNDynamicReverseSegment",
    "DistributedTNDynamicReverseSegmentResult",
    "DistributedTNParameterPullbackResult",
    "execute_dynamic_tn_reverse_segment",
    "execute_dynamic_tn_parameter_pullback",
    "plan_dynamic_tn_reverse_segment",
)

"""Explicit reverse-mode records and execution for a contraction DAG."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from math import prod
from typing import Any, Mapping, Sequence

import torch

from ....simulation.real_imag_kernels import complex_einsum_pair
from ....simulation.tensor_network.stages import (
    batched_pair_equation,
    einsum_pair_by_labels,
    einsum_pair_pullback,
    einsum_pair_pullback_by_equations,
    pair_equation,
)
from .distributed_dag import DistributedTNContractionDAG

TN_REVERSE_DAG_VERSION = "flagquantum.distributed_tn_reverse_dag.v1"
TN_CHECKPOINT_PLAN_VERSION = "flagquantum.distributed_tn_checkpoint_plan.v1"
TN_ADJOINT_LAYOUT_VERSION = "flagquantum.distributed_tn_adjoint_layout.v1"


@dataclass(frozen=True)
class CompiledTNForwardBucket:
    """Same-shape independent contractions executable as one batched kernel."""

    equation: str
    batched_equation: str
    operation_ids: tuple[str, ...]


@dataclass(frozen=True)
class CompiledTNForwardSchedule:
    """Dependency-level schedule for a fixed contraction DAG."""

    forward_dag_identity: str
    stages: tuple[tuple[CompiledTNForwardBucket, ...], ...]
    operation_count: int
    kernel_launch_count: int


@dataclass(frozen=True)
class CompiledTNReverseBucket:
    left_equation: str
    left_batched_equation: str
    right_equation: str
    right_batched_equation: str
    reverse_ids: tuple[str, ...]


@dataclass(frozen=True)
class CompiledTNReverseSchedule:
    forward_dag_identity: str
    reverse_dag_identity: str
    stages: tuple[tuple[CompiledTNReverseBucket, ...], ...]
    operation_count: int
    kernel_launch_count: int


_COMPILED_FORWARD_SCHEDULE_CACHE: dict[str, CompiledTNForwardSchedule] = {}
_COMPILED_REVERSE_SCHEDULE_CACHE: dict[tuple[str, str], CompiledTNReverseSchedule] = {}


@dataclass(frozen=True)
class DistributedTNAdjointValueLayout:
    """Rank-local cotangent ownership inherited from a forward mesh."""

    value_id: str
    logical_labels: tuple[int, ...]
    logical_shape: tuple[int, ...]
    shard_labels: tuple[int, ...]
    replicated_mesh_labels: tuple[int, ...]
    local_shape: tuple[int, ...]
    logical_nbytes: int
    local_nbytes: int


@dataclass(frozen=True)
class DistributedTNAdjointLayoutPlan:
    """Auditable cotangent layouts for a fixed multi-axis rank mesh."""

    version: str
    identity: str
    forward_dag_identity: str
    reverse_dag_identity: str
    mesh_labels: tuple[int, ...]
    mesh_shape: tuple[int, ...]
    world_size: int
    values: tuple[DistributedTNAdjointValueLayout, ...]

    def validate(
        self,
        forward: DistributedTNContractionDAG,
        reverse: DistributedTNReverseDAG,
    ) -> None:
        reverse.validate(forward)
        if self.version != TN_ADJOINT_LAYOUT_VERSION:
            raise ValueError("unsupported distributed TN adjoint layout version")
        if self.forward_dag_identity != forward.identity:
            raise ValueError("adjoint layout does not match the forward DAG")
        if self.reverse_dag_identity != reverse.identity:
            raise ValueError("adjoint layout does not match the reverse DAG")
        if prod(self.mesh_shape) != self.world_size:
            raise ValueError("adjoint mesh shape does not match world size")
        if len(self.mesh_labels) != len(self.mesh_shape):
            raise ValueError("adjoint mesh labels and shape are inconsistent")
        forward_values = {value.value_id: value for value in forward.values}
        if {value.value_id for value in self.values} != set(forward_values):
            raise ValueError("adjoint layouts do not cover the forward values")
        for value in self.values:
            logical = forward_values[value.value_id]
            if (
                value.logical_labels != logical.labels
                or value.logical_shape != logical.shape
                or value.logical_nbytes != logical.nbytes
            ):
                raise ValueError("adjoint logical layout mismatches forward value")
            expected_shards = tuple(
                label for label in self.mesh_labels if label in logical.labels
            )
            expected_replicated = tuple(
                label for label in self.mesh_labels if label not in logical.labels
            )
            if (
                value.shard_labels != expected_shards
                or value.replicated_mesh_labels != expected_replicated
            ):
                raise ValueError("adjoint mesh inheritance is inconsistent")
            expected_shape = tuple(
                1 if label in expected_shards else extent
                for label, extent in zip(logical.labels, logical.shape)
            )
            divisor = prod(
                self.mesh_shape[self.mesh_labels.index(label)]
                for label in expected_shards
            )
            if (
                value.local_shape != expected_shape
                or value.local_nbytes != logical.nbytes // divisor
            ):
                raise ValueError("adjoint rank-local shape or bytes are inconsistent")
        if self.identity != _adjoint_layout_identity(self._payload()):
            raise ValueError("adjoint layout identity does not match its contents")

    def _payload(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "forward_dag_identity": self.forward_dag_identity,
            "reverse_dag_identity": self.reverse_dag_identity,
            "mesh_labels": self.mesh_labels,
            "mesh_shape": self.mesh_shape,
            "world_size": self.world_size,
            "values": tuple(asdict(value) for value in self.values),
        }

    def summary(self) -> dict[str, Any]:
        return {
            **self._payload(),
            "identity": self.identity,
            "partitioned_value_count": sum(
                bool(value.shard_labels) for value in self.values
            ),
            "fully_replicated_value_count": sum(
                not value.shard_labels for value in self.values
            ),
            "distribution_semantics": "forward_mesh_inherited_adjoint",
        }


@dataclass(frozen=True)
class DistributedTNCheckpointPlan:
    """Deterministic saved-value selection under a tape byte budget."""

    version: str
    identity: str
    forward_dag_identity: str
    budget_bytes: int
    checkpoint_value_ids: tuple[str, ...]
    saved_intermediate_bytes: int
    estimated_rematerialization_cost: int

    def validate(self, forward: DistributedTNContractionDAG) -> None:
        forward.validate()
        if self.version != TN_CHECKPOINT_PLAN_VERSION:
            raise ValueError("unsupported distributed TN checkpoint plan version")
        if self.forward_dag_identity != forward.identity:
            raise ValueError("checkpoint plan does not match the forward DAG identity")
        if self.budget_bytes < 0:
            raise ValueError("checkpoint budget must be non-negative")
        values = {value.value_id: value for value in forward.values}
        if len(set(self.checkpoint_value_ids)) != len(self.checkpoint_value_ids):
            raise ValueError("checkpoint plan contains duplicate value ids")
        if any(value_id not in values for value_id in self.checkpoint_value_ids):
            raise ValueError("checkpoint plan contains an unknown value id")
        if any(
            values[value_id].producer_id is None or value_id == forward.output_value_id
            for value_id in self.checkpoint_value_ids
        ):
            raise ValueError("checkpoint plan may contain only intermediates")
        saved = sum(values[value_id].nbytes for value_id in self.checkpoint_value_ids)
        if saved != self.saved_intermediate_bytes or saved > self.budget_bytes:
            raise ValueError("checkpoint plan saved bytes are inconsistent")
        if self.identity != _checkpoint_identity(self._payload()):
            raise ValueError("checkpoint plan identity does not match its contents")

    def _payload(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "forward_dag_identity": self.forward_dag_identity,
            "budget_bytes": self.budget_bytes,
            "checkpoint_value_ids": self.checkpoint_value_ids,
            "saved_intermediate_bytes": self.saved_intermediate_bytes,
            "estimated_rematerialization_cost": (self.estimated_rematerialization_cost),
        }

    def summary(self) -> dict[str, Any]:
        return {
            **self._payload(),
            "identity": self.identity,
            "checkpoint_count": len(self.checkpoint_value_ids),
            "strategy": "cost_per_saved_byte_greedy",
        }


@dataclass(frozen=True)
class DistributedTNReverseRecord:
    """One reverse pair tied to a forward contraction operation."""

    reverse_id: str
    sequence: int
    forward_operation_id: str
    output_value_id: str
    input_value_ids: tuple[str, str]
    output_labels: tuple[int, ...]
    left_labels: tuple[int, ...]
    right_labels: tuple[int, ...]


@dataclass(frozen=True)
class DistributedTNReverseDAG:
    """Versioned reverse schedule sharing a forward DAG identity."""

    version: str
    identity: str
    forward_dag_identity: str
    records: tuple[DistributedTNReverseRecord, ...]
    checkpoint_value_ids: tuple[str, ...]

    def validate(self, forward: DistributedTNContractionDAG) -> None:
        forward.validate()
        if self.version != TN_REVERSE_DAG_VERSION:
            raise ValueError("unsupported distributed TN reverse DAG version")
        if self.forward_dag_identity != forward.identity:
            raise ValueError("reverse TN DAG does not match the forward DAG identity")
        expected = tuple(reversed(forward.operations))
        if len(self.records) != len(expected):
            raise ValueError("reverse TN DAG operation count mismatch")
        for sequence, (record, operation) in enumerate(zip(self.records, expected)):
            if record.sequence != sequence:
                raise ValueError("reverse TN DAG sequence is not contiguous")
            if record.forward_operation_id != operation.operation_id:
                raise ValueError("reverse TN DAG order does not invert the forward DAG")
        if self.identity != _reverse_identity(self._payload()):
            raise ValueError("reverse TN DAG identity does not match its contents")

    def _payload(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "forward_dag_identity": self.forward_dag_identity,
            "records": tuple(asdict(record) for record in self.records),
            "checkpoint_value_ids": self.checkpoint_value_ids,
        }

    def summary(self) -> dict[str, Any]:
        return {
            **self._payload(),
            "identity": self.identity,
            "record_count": len(self.records),
            "explicit_reverse": True,
            "uses_generic_autograd_for_contractions": False,
            "scalability_claim_allowed": False,
            "scalability_blockers": (
                "distributed_adjoint_ownership_pending",
                "parameter_jacobian_chain_pending",
            ),
        }


@dataclass(frozen=True)
class DistributedTNReverseResult:
    """Explicit input cotangents and reverse execution evidence."""

    input_cotangents: Mapping[str, torch.Tensor]
    reverse_dag_identity: str
    executed_reverse_ids: tuple[str, ...]
    accumulated_cotangent_count: int
    nonfinite_cotangent_count: int
    rematerialized_operation_count: int = 0
    saved_tape_bytes: int = 0
    cotangent_tape: Mapping[str, torch.Tensor] | None = None


def plan_explicit_tn_reverse_dag(
    forward: DistributedTNContractionDAG,
) -> DistributedTNReverseDAG:
    """Build the deterministic inverse schedule for a forward contraction DAG."""

    forward.validate()
    values = {value.value_id: value for value in forward.values}
    records = tuple(
        DistributedTNReverseRecord(
            reverse_id=f"reverse:{operation.sequence}",
            sequence=sequence,
            forward_operation_id=operation.operation_id,
            output_value_id=operation.output_value_id,
            input_value_ids=operation.input_value_ids,
            output_labels=operation.output_labels,
            left_labels=values[operation.input_value_ids[0]].labels,
            right_labels=values[operation.input_value_ids[1]].labels,
        )
        for sequence, operation in enumerate(reversed(forward.operations))
    )
    checkpoints = tuple(
        value.value_id
        for value in forward.values
        if value.value_id != forward.output_value_id
    )
    payload = {
        "version": TN_REVERSE_DAG_VERSION,
        "forward_dag_identity": forward.identity,
        "records": tuple(asdict(record) for record in records),
        "checkpoint_value_ids": checkpoints,
    }
    reverse = DistributedTNReverseDAG(
        identity=_reverse_identity(payload),
        records=records,
        checkpoint_value_ids=checkpoints,
        version=TN_REVERSE_DAG_VERSION,
        forward_dag_identity=forward.identity,
    )
    reverse.validate(forward)
    return reverse


def plan_tn_adjoint_layouts(
    forward: DistributedTNContractionDAG,
    reverse: DistributedTNReverseDAG,
    *,
    mesh_labels: Sequence[int],
) -> DistributedTNAdjointLayoutPlan:
    """Inherit a fixed forward Cartesian mesh for every reverse cotangent."""

    reverse.validate(forward)
    labels = tuple(int(label) for label in mesh_labels)
    if not labels or len(set(labels)) != len(labels):
        raise ValueError("adjoint mesh labels must be unique and non-empty")
    mesh_shape = []
    for label in labels:
        extents = {
            value.shape[value.labels.index(label)]
            for value in forward.values
            if label in value.labels
        }
        if not extents:
            raise ValueError("adjoint mesh label is absent from the forward DAG")
        if len(extents) != 1:
            raise ValueError("adjoint mesh label has inconsistent extents")
        mesh_shape.append(next(iter(extents)))
    mesh = tuple(mesh_shape)
    if prod(mesh) != forward.world_size:
        raise ValueError("adjoint mesh product must equal the DAG world size")
    layouts = []
    for value in forward.values:
        active = tuple(label for label in labels if label in value.labels)
        replicated = tuple(label for label in labels if label not in value.labels)
        local_shape = tuple(
            1 if label in active else extent
            for label, extent in zip(value.labels, value.shape)
        )
        divisor = prod(mesh[labels.index(label)] for label in active)
        layouts.append(
            DistributedTNAdjointValueLayout(
                value_id=value.value_id,
                logical_labels=value.labels,
                logical_shape=value.shape,
                shard_labels=active,
                replicated_mesh_labels=replicated,
                local_shape=local_shape,
                logical_nbytes=value.nbytes,
                local_nbytes=value.nbytes // divisor,
            )
        )
    payload = {
        "version": TN_ADJOINT_LAYOUT_VERSION,
        "forward_dag_identity": forward.identity,
        "reverse_dag_identity": reverse.identity,
        "mesh_labels": labels,
        "mesh_shape": mesh,
        "world_size": forward.world_size,
        "values": tuple(asdict(value) for value in layouts),
    }
    plan = DistributedTNAdjointLayoutPlan(
        version=TN_ADJOINT_LAYOUT_VERSION,
        identity=_adjoint_layout_identity(payload),
        forward_dag_identity=forward.identity,
        reverse_dag_identity=reverse.identity,
        mesh_labels=labels,
        mesh_shape=mesh,
        world_size=forward.world_size,
        values=tuple(layouts),
    )
    plan.validate(forward, reverse)
    return plan


def validate_tn_adjoint_tensors(
    plan: DistributedTNAdjointLayoutPlan,
    tensors: Mapping[str, torch.Tensor],
    *,
    require_all: bool = False,
) -> None:
    """Fail closed when rank-local cotangents violate their planned layouts."""

    layouts = {value.value_id: value for value in plan.values}
    unknown = tuple(sorted(set(tensors) - set(layouts)))
    if unknown:
        raise ValueError(f"adjoint tensors contain unknown values {unknown}")
    if require_all:
        missing = tuple(sorted(set(layouts) - set(tensors)))
        if missing:
            raise ValueError(f"adjoint tensors are missing planned values {missing}")
    for value_id, tensor in tensors.items():
        if (
            tuple(int(extent) for extent in tensor.shape)
            != layouts[value_id].local_shape
        ):
            raise ValueError(
                f"adjoint tensor {value_id!r} violates its rank-local shape"
            )


def plan_tn_checkpoints(
    forward: DistributedTNContractionDAG,
    *,
    budget_bytes: int,
) -> DistributedTNCheckpointPlan:
    """Select saved intermediates by avoided compute per stored byte."""

    forward.validate()
    if budget_bytes < 0:
        raise ValueError("checkpoint budget must be non-negative")
    operations = {
        operation.output_value_id: operation for operation in forward.operations
    }
    candidates = [
        value
        for value in forward.values
        if value.producer_id is not None
        and value.value_id != forward.output_value_id
        and value.nbytes <= budget_bytes
    ]
    candidates.sort(
        key=lambda value: (
            -(operations[value.value_id].estimated_cost / max(1, value.nbytes)),
            value.value_id,
        )
    )
    selected = []
    saved_bytes = 0
    for value in candidates:
        if saved_bytes + value.nbytes <= budget_bytes:
            selected.append(value.value_id)
            saved_bytes += value.nbytes
    selected_ids = tuple(
        value.value_id for value in forward.values if value.value_id in set(selected)
    )
    estimated_rematerialization_cost = sum(
        operation.estimated_cost
        for operation in forward.operations
        if operation.output_value_id != forward.output_value_id
        and operation.output_value_id not in selected_ids
    )
    payload = {
        "version": TN_CHECKPOINT_PLAN_VERSION,
        "forward_dag_identity": forward.identity,
        "budget_bytes": budget_bytes,
        "checkpoint_value_ids": selected_ids,
        "saved_intermediate_bytes": saved_bytes,
        "estimated_rematerialization_cost": estimated_rematerialization_cost,
    }
    plan = DistributedTNCheckpointPlan(
        **payload,
        identity=_checkpoint_identity(payload),
    )
    plan.validate(forward)
    return plan


def execute_tn_forward_with_tape(
    dag: DistributedTNContractionDAG,
    inputs: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Execute the exact forward schedule and retain its explicit value tape."""

    dag.validate()
    values = {value.value_id: value for value in dag.values}
    expected = {value.value_id for value in dag.values if value.producer_id is None}
    if {str(value_id) for value_id in inputs} != expected:
        raise ValueError("TN forward tape requires exactly all raw DAG inputs")
    tape = dict(inputs)
    for operation in dag.operations:
        left_id, right_id = operation.input_value_ids
        tape[operation.output_value_id] = einsum_pair_by_labels(
            tape[left_id],
            values[left_id].labels,
            tape[right_id],
            values[right_id].labels,
            operation.output_labels,
        )
    return tape


def compile_tn_forward_schedule(
    dag: DistributedTNContractionDAG,
) -> CompiledTNForwardSchedule:
    """Group independent same-shape DAG operations into batched stages."""

    dag.validate()
    cached = _COMPILED_FORWARD_SCHEDULE_CACHE.get(dag.identity)
    if cached is not None:
        return cached
    values = {value.value_id: value for value in dag.values}
    operations = {operation.operation_id: operation for operation in dag.operations}
    value_levels = {
        value.value_id: 0 for value in dag.values if value.producer_id is None
    }
    levels: dict[int, list[Any]] = {}
    for operation in dag.operations:
        level = (
            max(value_levels[value_id] for value_id in operation.input_value_ids) + 1
        )
        value_levels[operation.output_value_id] = level
        levels.setdefault(level, []).append(operation)
    stages: list[tuple[CompiledTNForwardBucket, ...]] = []
    for level in sorted(levels):
        grouped: dict[tuple[Any, ...], list[str]] = {}
        for operation in levels[level]:
            left_id, right_id = operation.input_value_ids
            equation = pair_equation(
                values[left_id].labels,
                values[right_id].labels,
                operation.output_labels,
            )
            key = (
                equation,
                values[left_id].shape,
                values[right_id].shape,
                operation.output_shape,
            )
            grouped.setdefault(key, []).append(operation.operation_id)
        stages.append(
            tuple(
                CompiledTNForwardBucket(
                    equation=key[0],
                    batched_equation=batched_pair_equation(key[0]),
                    operation_ids=tuple(operation_ids),
                )
                for key, operation_ids in grouped.items()
            )
        )
    schedule = CompiledTNForwardSchedule(
        forward_dag_identity=dag.identity,
        stages=tuple(stages),
        operation_count=len(dag.operations),
        kernel_launch_count=sum(len(stage) for stage in stages),
    )
    if {
        operation_id
        for stage in schedule.stages
        for bucket in stage
        for operation_id in bucket.operation_ids
    } != set(operations):
        raise RuntimeError("compiled TN forward schedule does not cover the DAG")
    _COMPILED_FORWARD_SCHEDULE_CACHE[dag.identity] = schedule
    return schedule


def execute_compiled_tn_forward_with_tape(
    dag: DistributedTNContractionDAG,
    inputs: Mapping[str, torch.Tensor],
    schedule: CompiledTNForwardSchedule | None = None,
) -> dict[str, torch.Tensor]:
    """Execute a dependency-staged forward while retaining the exact tape."""

    dag.validate()
    schedule = schedule or compile_tn_forward_schedule(dag)
    if schedule.forward_dag_identity != dag.identity:
        raise ValueError("compiled TN forward schedule does not match the DAG")
    expected = {value.value_id for value in dag.values if value.producer_id is None}
    if {str(value_id) for value_id in inputs} != expected:
        raise ValueError("compiled TN forward requires exactly all raw DAG inputs")
    operations = {operation.operation_id: operation for operation in dag.operations}
    tape = dict(inputs)
    for stage in schedule.stages:
        for bucket in stage:
            bucket_operations = tuple(
                operations[operation_id] for operation_id in bucket.operation_ids
            )
            if len(bucket_operations) == 1:
                operation = bucket_operations[0]
                left_id, right_id = operation.input_value_ids
                tape[operation.output_value_id] = complex_einsum_pair(
                    bucket.equation,
                    tape[left_id],
                    tape[right_id],
                    compile_cuda=False,
                )
                continue
            left = torch.stack(
                [tape[operation.input_value_ids[0]] for operation in bucket_operations]
            )
            right = torch.stack(
                [tape[operation.input_value_ids[1]] for operation in bucket_operations]
            )
            outputs = complex_einsum_pair(bucket.batched_equation, left, right)
            for position, operation in enumerate(bucket_operations):
                tape[operation.output_value_id] = outputs[position]
    return tape


def execute_tn_forward_with_checkpoint_tape(
    dag: DistributedTNContractionDAG,
    inputs: Mapping[str, torch.Tensor],
    checkpoints: DistributedTNCheckpointPlan,
) -> dict[str, torch.Tensor]:
    """Execute forward and retain only raw inputs, selected values, and output."""

    checkpoints.validate(dag)
    retained = {value.value_id for value in dag.values if value.producer_id is None}
    retained.update(checkpoints.checkpoint_value_ids)
    retained.add(dag.output_value_id)
    values = {value.value_id: value for value in dag.values}
    remaining_uses = {value.value_id: 0 for value in dag.values}
    for operation in dag.operations:
        for value_id in operation.input_value_ids:
            remaining_uses[value_id] += 1
    live = dict(inputs)
    tape = dict(inputs)
    for operation in dag.operations:
        left_id, right_id = operation.input_value_ids
        output = einsum_pair_by_labels(
            live[left_id],
            values[left_id].labels,
            live[right_id],
            values[right_id].labels,
            operation.output_labels,
        )
        live[operation.output_value_id] = output
        if operation.output_value_id in retained:
            tape[operation.output_value_id] = output
        for value_id in operation.input_value_ids:
            remaining_uses[value_id] -= 1
            if remaining_uses[value_id] == 0 and value_id not in retained:
                del live[value_id]
    return tape


def execute_checkpointed_tn_reverse_dag(
    forward: DistributedTNContractionDAG,
    reverse: DistributedTNReverseDAG,
    checkpoints: DistributedTNCheckpointPlan,
    tape: Mapping[str, torch.Tensor],
    *,
    output_cotangent: torch.Tensor | None = None,
) -> DistributedTNReverseResult:
    """Run explicit reverse, reconstructing unsaved operands from checkpoints."""

    reverse.validate(forward)
    checkpoints.validate(forward)
    raw_ids = {value.value_id for value in forward.values if value.producer_id is None}
    required = (
        raw_ids | set(checkpoints.checkpoint_value_ids) | {forward.output_value_id}
    )
    missing = tuple(sorted(required - set(tape)))
    if missing:
        raise ValueError(f"checkpointed TN reverse tape is missing values {missing}")
    unexpected = tuple(sorted(set(tape) - required))
    if unexpected:
        raise ValueError(
            f"checkpointed TN reverse tape contains unplanned values {unexpected}"
        )
    values = {value.value_id: value for value in forward.values}
    producers = {
        operation.output_value_id: operation for operation in forward.operations
    }
    output = tape[forward.output_value_id]
    seed = torch.ones_like(output) if output_cotangent is None else output_cotangent
    if tuple(seed.shape) != tuple(output.shape):
        raise ValueError("explicit TN reverse output cotangent shape mismatch")
    cotangents: dict[str, torch.Tensor] = {forward.output_value_id: seed}
    executed = []
    accumulated = 0
    rematerialized = 0

    def materialize_pair(
        left_id: str,
        right_id: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        nonlocal rematerialized
        targets = {left_id, right_id}
        selected = set()
        pending = list(targets)
        while pending:
            value_id = pending.pop()
            if value_id in tape:
                continue
            operation = producers.get(value_id)
            if operation is None:
                raise ValueError(f"cannot rematerialize TN value {value_id!r}")
            if operation.operation_id in selected:
                continue
            selected.add(operation.operation_id)
            pending.extend(operation.input_value_ids)
        operations = tuple(
            operation
            for operation in forward.operations
            if operation.operation_id in selected
        )
        remaining_uses = {
            operation.output_value_id: int(operation.output_value_id in targets)
            for operation in operations
        }
        for operation in operations:
            for input_id in operation.input_value_ids:
                if input_id in remaining_uses:
                    remaining_uses[input_id] += 1
        transient: dict[str, torch.Tensor] = {}
        for operation in operations:
            operand_ids = operation.input_value_ids
            operands = tuple(
                tape.get(value_id, transient.get(value_id)) for value_id in operand_ids
            )
            if any(operand is None for operand in operands):
                raise RuntimeError("TN rematerialization schedule is incomplete")
            transient[operation.output_value_id] = einsum_pair_by_labels(
                operands[0],
                values[operand_ids[0]].labels,
                operands[1],
                values[operand_ids[1]].labels,
                operation.output_labels,
            )
            rematerialized += 1
            for input_id in operand_ids:
                if input_id not in remaining_uses:
                    continue
                remaining_uses[input_id] -= 1
                if remaining_uses[input_id] == 0:
                    transient.pop(input_id, None)
        return (
            tape.get(left_id, transient.get(left_id)),
            tape.get(right_id, transient.get(right_id)),
        )

    for record in reverse.records:
        output_cot = cotangents.pop(record.output_value_id, None)
        if output_cot is None:
            continue
        left_id, right_id = record.input_value_ids
        left, right = materialize_pair(left_id, right_id)
        if left is None or right is None:
            raise RuntimeError("TN rematerialization did not produce both operands")
        left_cot, right_cot = einsum_pair_pullback(
            output_cot,
            record.output_labels,
            left,
            record.left_labels,
            right,
            record.right_labels,
        )
        accumulated += _accumulate_cotangent(cotangents, left_id, left_cot)
        accumulated += _accumulate_cotangent(cotangents, right_id, right_cot)
        executed.append(record.reverse_id)
    input_cotangents = {
        value_id: cotangents[value_id] for value_id in raw_ids if value_id in cotangents
    }
    return DistributedTNReverseResult(
        input_cotangents=input_cotangents,
        reverse_dag_identity=reverse.identity,
        executed_reverse_ids=tuple(executed),
        accumulated_cotangent_count=accumulated,
        nonfinite_cotangent_count=sum(
            int(not bool(torch.isfinite(value).all()))
            for value in input_cotangents.values()
        ),
        rematerialized_operation_count=rematerialized,
        saved_tape_bytes=sum(
            int(value.numel()) * int(value.element_size()) for value in tape.values()
        ),
    )


def compile_tn_reverse_schedule(
    forward: DistributedTNContractionDAG,
    reverse: DistributedTNReverseDAG,
) -> CompiledTNReverseSchedule:
    """Group independent reverse records into same-shape batched stages."""

    reverse.validate(forward)
    cache_key = (forward.identity, reverse.identity)
    cached = _COMPILED_REVERSE_SCHEDULE_CACHE.get(cache_key)
    if cached is not None:
        return cached
    values = {value.value_id: value for value in forward.values}
    reverse_levels = {forward.output_value_id: 0}
    levels: dict[int, list[DistributedTNReverseRecord]] = {}
    for record in reverse.records:
        level = reverse_levels[record.output_value_id]
        levels.setdefault(level, []).append(record)
        for value_id in record.input_value_ids:
            reverse_levels[value_id] = level + 1
    stages: list[tuple[CompiledTNReverseBucket, ...]] = []
    for level in sorted(levels):
        grouped: dict[tuple[Any, ...], list[str]] = {}
        for record in levels[level]:
            left_id, right_id = record.input_value_ids
            left_equation = pair_equation(
                record.output_labels,
                record.right_labels,
                record.left_labels,
            )
            right_equation = pair_equation(
                record.left_labels,
                record.output_labels,
                record.right_labels,
            )
            key = (
                left_equation,
                right_equation,
                values[record.output_value_id].shape,
                values[left_id].shape,
                values[right_id].shape,
            )
            grouped.setdefault(key, []).append(record.reverse_id)
        stages.append(
            tuple(
                CompiledTNReverseBucket(
                    left_equation=key[0],
                    left_batched_equation=batched_pair_equation(key[0]),
                    right_equation=key[1],
                    right_batched_equation=batched_pair_equation(key[1]),
                    reverse_ids=tuple(reverse_ids),
                )
                for key, reverse_ids in grouped.items()
            )
        )
    schedule = CompiledTNReverseSchedule(
        forward_dag_identity=forward.identity,
        reverse_dag_identity=reverse.identity,
        stages=tuple(stages),
        operation_count=len(reverse.records),
        kernel_launch_count=2 * sum(len(stage) for stage in stages),
    )
    _COMPILED_REVERSE_SCHEDULE_CACHE[cache_key] = schedule
    return schedule


def execute_compiled_tn_reverse_dag(
    forward: DistributedTNContractionDAG,
    reverse: DistributedTNReverseDAG,
    tape: Mapping[str, torch.Tensor],
    *,
    output_cotangent: torch.Tensor | None = None,
    schedule: CompiledTNReverseSchedule | None = None,
) -> DistributedTNReverseResult:
    """Execute exact explicit reverse using dependency-staged batched kernels."""

    reverse.validate(forward)
    schedule = schedule or compile_tn_reverse_schedule(forward, reverse)
    if (
        schedule.forward_dag_identity != forward.identity
        or schedule.reverse_dag_identity != reverse.identity
    ):
        raise ValueError("compiled TN reverse schedule does not match the DAG")
    records = {record.reverse_id: record for record in reverse.records}
    output = tape.get(forward.output_value_id)
    if output is None:
        raise ValueError("compiled TN reverse tape is missing the forward output")
    seed = torch.ones_like(output) if output_cotangent is None else output_cotangent
    if tuple(seed.shape) != tuple(output.shape):
        raise ValueError("compiled TN reverse output cotangent shape mismatch")
    cotangents: dict[str, torch.Tensor] = {forward.output_value_id: seed}
    executed: list[str] = []
    accumulated = 0
    for stage in schedule.stages:
        for bucket in stage:
            active = tuple(
                records[reverse_id]
                for reverse_id in bucket.reverse_ids
                if records[reverse_id].output_value_id in cotangents
            )
            if not active:
                continue
            if len(active) == 1:
                record = active[0]
                output_cot = cotangents.pop(record.output_value_id)
                left_id, right_id = record.input_value_ids
                left_cot, right_cot = einsum_pair_pullback_by_equations(
                    bucket.left_equation,
                    bucket.right_equation,
                    output_cot,
                    tape[left_id],
                    tape[right_id],
                    compile_cuda=False,
                )
                accumulated += _accumulate_cotangent(cotangents, left_id, left_cot)
                accumulated += _accumulate_cotangent(cotangents, right_id, right_cot)
                executed.append(record.reverse_id)
                continue
            output_cots = torch.stack(
                [cotangents.pop(record.output_value_id) for record in active]
            )
            right_values = torch.stack(
                [tape[record.input_value_ids[1]] for record in active]
            )
            left_values = torch.stack(
                [tape[record.input_value_ids[0]] for record in active]
            )
            left_cots, right_cots = einsum_pair_pullback_by_equations(
                bucket.left_batched_equation,
                bucket.right_batched_equation,
                output_cots,
                left_values,
                right_values,
            )
            for position, record in enumerate(active):
                left_id, right_id = record.input_value_ids
                accumulated += _accumulate_cotangent(
                    cotangents, left_id, left_cots[position]
                )
                accumulated += _accumulate_cotangent(
                    cotangents, right_id, right_cots[position]
                )
                executed.append(record.reverse_id)
    input_ids = tuple(
        value.value_id for value in forward.values if value.producer_id is None
    )
    input_cotangents = {
        value_id: cotangents[value_id]
        for value_id in input_ids
        if value_id in cotangents
    }
    return DistributedTNReverseResult(
        input_cotangents=input_cotangents,
        reverse_dag_identity=reverse.identity,
        executed_reverse_ids=tuple(executed),
        accumulated_cotangent_count=accumulated,
        nonfinite_cotangent_count=sum(
            int(not bool(torch.isfinite(value).all()))
            for value in input_cotangents.values()
        ),
    )


def execute_explicit_tn_reverse_dag(
    forward: DistributedTNContractionDAG,
    reverse: DistributedTNReverseDAG,
    tape: Mapping[str, torch.Tensor],
    *,
    output_cotangent: torch.Tensor | None = None,
    retain_cotangent_tape: bool = False,
) -> DistributedTNReverseResult:
    """Propagate cotangents using explicit conjugate pair-contraction rules."""

    reverse.validate(forward)
    missing = tuple(
        value_id for value_id in reverse.checkpoint_value_ids if value_id not in tape
    )
    if missing:
        raise ValueError(f"explicit TN reverse tape is missing checkpoints {missing}")
    output = tape.get(forward.output_value_id)
    if output is None:
        raise ValueError("explicit TN reverse tape is missing the forward output")
    seed = torch.ones_like(output) if output_cotangent is None else output_cotangent
    if tuple(seed.shape) != tuple(output.shape):
        raise ValueError("explicit TN reverse output cotangent shape mismatch")
    cotangents: dict[str, torch.Tensor] = {forward.output_value_id: seed}
    executed = []
    accumulated = 0
    retained_cotangents: dict[str, torch.Tensor] | None = (
        {} if retain_cotangent_tape else None
    )
    for record in reverse.records:
        output_cot = cotangents.pop(record.output_value_id, None)
        if output_cot is None:
            continue
        if retained_cotangents is not None:
            retained_cotangents[record.output_value_id] = output_cot
        left_id, right_id = record.input_value_ids
        left_cot, right_cot = einsum_pair_pullback(
            output_cot,
            record.output_labels,
            tape[left_id],
            record.left_labels,
            tape[right_id],
            record.right_labels,
        )
        accumulated += _accumulate_cotangent(cotangents, left_id, left_cot)
        accumulated += _accumulate_cotangent(cotangents, right_id, right_cot)
        executed.append(record.reverse_id)
    input_ids = tuple(
        value.value_id for value in forward.values if value.producer_id is None
    )
    input_cotangents = {
        value_id: cotangents[value_id]
        for value_id in input_ids
        if value_id in cotangents
    }
    nonfinite = sum(
        int(not bool(torch.isfinite(value).all()))
        for value in input_cotangents.values()
    )
    if retained_cotangents is not None:
        retained_cotangents.update(input_cotangents)
    return DistributedTNReverseResult(
        input_cotangents=input_cotangents,
        reverse_dag_identity=reverse.identity,
        executed_reverse_ids=tuple(executed),
        accumulated_cotangent_count=accumulated,
        nonfinite_cotangent_count=nonfinite,
        cotangent_tape=retained_cotangents,
    )


def _accumulate_cotangent(
    cotangents: dict[str, torch.Tensor],
    value_id: str,
    contribution: torch.Tensor,
) -> int:
    previous = cotangents.get(value_id)
    cotangents[value_id] = contribution if previous is None else previous + contribution
    return int(previous is not None)


def _reverse_identity(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _checkpoint_identity(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _adjoint_layout_identity(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


__all__ = (
    "CompiledTNForwardBucket",
    "CompiledTNForwardSchedule",
    "CompiledTNReverseBucket",
    "CompiledTNReverseSchedule",
    "DistributedTNAdjointLayoutPlan",
    "DistributedTNAdjointValueLayout",
    "DistributedTNCheckpointPlan",
    "DistributedTNReverseDAG",
    "DistributedTNReverseRecord",
    "DistributedTNReverseResult",
    "execute_checkpointed_tn_reverse_dag",
    "execute_compiled_tn_forward_with_tape",
    "execute_compiled_tn_reverse_dag",
    "execute_explicit_tn_reverse_dag",
    "execute_tn_forward_with_checkpoint_tape",
    "execute_tn_forward_with_tape",
    "compile_tn_forward_schedule",
    "compile_tn_reverse_schedule",
    "plan_explicit_tn_reverse_dag",
    "plan_tn_adjoint_layouts",
    "plan_tn_checkpoints",
    "validate_tn_adjoint_tensors",
)

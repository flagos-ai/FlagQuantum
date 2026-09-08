"""Dependency-staged tensor-network forward and reverse execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch

from ....simulation.real_imag_kernels import complex_einsum_pair
from ....simulation.tensor_network.stages import (
    batched_pair_equation,
    einsum_pair_pullback_by_equations,
    pair_equation,
)
from .distributed_dag import DistributedTNContractionDAG
from .reverse_dag import (
    DistributedTNReverseDAG,
    DistributedTNReverseRecord,
    DistributedTNReverseResult,
    _accumulate_cotangent,
)


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


_FORWARD_SCHEDULE_CACHE: dict[str, CompiledTNForwardSchedule] = {}
_REVERSE_SCHEDULE_CACHE: dict[tuple[str, str], CompiledTNReverseSchedule] = {}


def compile_tn_forward_schedule(
    dag: DistributedTNContractionDAG,
) -> CompiledTNForwardSchedule:
    """Group independent same-shape DAG operations into batched stages."""

    dag.validate()
    cached = _FORWARD_SCHEDULE_CACHE.get(dag.identity)
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
    _FORWARD_SCHEDULE_CACHE[dag.identity] = schedule
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


def compile_tn_reverse_schedule(
    forward: DistributedTNContractionDAG,
    reverse: DistributedTNReverseDAG,
) -> CompiledTNReverseSchedule:
    """Group independent reverse records into same-shape batched stages."""

    reverse.validate(forward)
    cache_key = (forward.identity, reverse.identity)
    cached = _REVERSE_SCHEDULE_CACHE.get(cache_key)
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
    _REVERSE_SCHEDULE_CACHE[cache_key] = schedule
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


__all__ = (
    "CompiledTNForwardBucket",
    "CompiledTNForwardSchedule",
    "CompiledTNReverseBucket",
    "CompiledTNReverseSchedule",
    "compile_tn_forward_schedule",
    "compile_tn_reverse_schedule",
    "execute_compiled_tn_forward_with_tape",
    "execute_compiled_tn_reverse_dag",
)

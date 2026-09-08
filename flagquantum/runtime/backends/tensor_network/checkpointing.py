"""Checkpoint planning and rematerialized tensor-network reverse execution."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping

import torch

from ....simulation.tensor_network.stages import (
    einsum_pair_by_labels,
    einsum_pair_pullback,
)
from .distributed_dag import DistributedTNContractionDAG
from .reverse_dag import (
    DistributedTNReverseDAG,
    DistributedTNReverseResult,
    _accumulate_cotangent,
)

TN_CHECKPOINT_PLAN_VERSION = "flagquantum.distributed_tn_checkpoint_plan.v1"


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
            "estimated_rematerialization_cost": self.estimated_rematerialization_cost,
        }

    def summary(self) -> dict[str, Any]:
        return {
            **self._payload(),
            "identity": self.identity,
            "checkpoint_count": len(self.checkpoint_value_ids),
            "strategy": "cost_per_saved_byte_greedy",
        }


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


def _checkpoint_identity(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


__all__ = (
    "DistributedTNCheckpointPlan",
    "execute_checkpointed_tn_reverse_dag",
    "execute_tn_forward_with_checkpoint_tape",
    "plan_tn_checkpoints",
)

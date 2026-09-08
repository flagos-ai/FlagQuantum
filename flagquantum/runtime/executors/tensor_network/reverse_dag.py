"""Explicit reverse-mode records and execution for a contraction DAG."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Mapping

import torch

from ....simulation.tensor_network.stages import (
    einsum_pair_by_labels,
    einsum_pair_pullback,
)
from .distributed_dag import DistributedTNContractionDAG

TN_REVERSE_DAG_VERSION = "flagquantum.distributed_tn_reverse_dag.v1"


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


__all__ = (
    "DistributedTNReverseDAG",
    "DistributedTNReverseRecord",
    "DistributedTNReverseResult",
    "execute_explicit_tn_reverse_dag",
    "execute_tn_forward_with_tape",
    "plan_explicit_tn_reverse_dag",
)

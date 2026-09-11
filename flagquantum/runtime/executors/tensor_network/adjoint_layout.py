"""Rank-local cotangent layouts for distributed tensor-network reverse passes."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from math import prod
from typing import Any, Mapping, Sequence

import torch

from .distributed_dag import DistributedTNContractionDAG
from .reverse_dag import DistributedTNReverseDAG

TN_ADJOINT_LAYOUT_VERSION = "flagquantum.distributed_tn_adjoint_layout.v1"


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


def _adjoint_layout_identity(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


__all__ = (
    "DistributedTNAdjointLayoutPlan",
    "DistributedTNAdjointValueLayout",
    "plan_tn_adjoint_layouts",
    "validate_tn_adjoint_tensors",
)

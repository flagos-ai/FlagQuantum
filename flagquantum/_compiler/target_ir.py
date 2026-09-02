"""Immutable provider-free TargetIR for the private compiler."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any, Mapping

from flagquantum.core.operator_schema import OPERATOR_SCHEMAS, canonical_opcode

from .ir.operations import FrozenAttributes, SourceLocation, canonical_value
from .target_capabilities import MeasurementResult

_SHA256 = re.compile(r"[0-9a-f]{64}")
TARGET_IR_IDENTITY_SCHEMA = "flagquantum.target_ir.program.v1alpha1"


@dataclass(frozen=True)
class TargetOperation:
    operation: str
    physical_qubits: tuple[int, ...]
    attributes: FrozenAttributes | Mapping[str, Any] = field(
        default_factory=FrozenAttributes
    )
    location: SourceLocation | None = field(default=None, compare=False, hash=False)

    def __post_init__(self) -> None:
        operation = canonical_opcode(self.operation)
        schema = OPERATOR_SCHEMAS.get(operation)
        if schema is None:
            raise ValueError(f"unknown target operation {self.operation!r}")
        qubits = tuple(self.physical_qubits)
        if any(isinstance(item, bool) or not isinstance(item, int) for item in qubits):
            raise ValueError("physical qubits must be integers")
        if len(qubits) != schema.arity:
            raise ValueError("physical-qubit arity does not match the operation")
        if len(qubits) != len(set(qubits)) or any(item < 0 for item in qubits):
            raise ValueError("physical qubits must be unique and non-negative")
        attributes = FrozenAttributes(self.attributes)
        unknown = set(attributes) - set(schema.parameters)
        missing = set(schema.parameters) - set(attributes)
        if unknown or missing:
            raise ValueError(
                "target operation attributes must exactly match the schema"
            )
        if self.location is not None and not isinstance(self.location, SourceLocation):
            raise ValueError("target operation location must use SourceLocation")
        object.__setattr__(self, "operation", operation)
        object.__setattr__(self, "physical_qubits", qubits)
        object.__setattr__(self, "attributes", attributes)

    def canonical(self) -> dict[str, object]:
        return {
            "operation": self.operation,
            "physical_qubits": list(self.physical_qubits),
            "attributes": canonical_value(self.attributes),
        }


@dataclass(frozen=True)
class TargetIR:
    source_program_identity: str
    target_capability_fingerprint: str
    logical_to_physical: tuple[int, ...]
    operations: tuple[TargetOperation, ...]
    required_results: tuple[MeasurementResult, ...] = ()
    requested_shots: int | None = None
    identity_schema_version: str = TARGET_IR_IDENTITY_SCHEMA

    def __post_init__(self) -> None:
        for name in ("source_program_identity", "target_capability_fingerprint"):
            if _SHA256.fullmatch(getattr(self, name)) is None:
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")
        if self.identity_schema_version != TARGET_IR_IDENTITY_SCHEMA:
            raise ValueError("unsupported TargetIR identity schema version")
        layout = tuple(self.logical_to_physical)
        if any(isinstance(item, bool) or not isinstance(item, int) for item in layout):
            raise ValueError("TargetIR layout entries must be integers")
        if len(layout) != len(set(layout)) or any(item < 0 for item in layout):
            raise ValueError("TargetIR layout must contain unique non-negative qubits")
        operations = tuple(self.operations)
        if any(not isinstance(item, TargetOperation) for item in operations):
            raise ValueError("TargetIR operations must use TargetOperation")
        results = tuple(self.required_results)
        if any(not isinstance(item, MeasurementResult) for item in results):
            raise ValueError("TargetIR results must use MeasurementResult")
        if len(results) != len(set(results)):
            raise ValueError("TargetIR required results must be unique")
        if self.requested_shots is not None and (
            isinstance(self.requested_shots, bool)
            or not isinstance(self.requested_shots, int)
            or self.requested_shots <= 0
        ):
            raise ValueError("TargetIR requested shots must be a positive integer")
        object.__setattr__(self, "logical_to_physical", layout)
        object.__setattr__(self, "operations", operations)
        object.__setattr__(
            self,
            "required_results",
            tuple(sorted(results, key=lambda item: item.value)),
        )

    def canonical(self) -> dict[str, object]:
        return {
            "identity_schema_version": self.identity_schema_version,
            "source_program_identity": self.source_program_identity,
            "target_capability_fingerprint": self.target_capability_fingerprint,
            "logical_to_physical": list(self.logical_to_physical),
            "operations": [item.canonical() for item in self.operations],
            "required_results": [item.value for item in self.required_results],
            "requested_shots": self.requested_shots,
        }

    @cached_property
    def target_program_identity(self) -> str:
        payload = json.dumps(
            self.canonical(), ensure_ascii=True, separators=(",", ":"), sort_keys=True
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


__all__ = ["TARGET_IR_IDENTITY_SCHEMA", "TargetIR", "TargetOperation"]

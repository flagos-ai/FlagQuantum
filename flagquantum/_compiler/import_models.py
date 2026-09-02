"""Typed, immutable models at the CircuitIR import boundary."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import Enum

from .bindings import BindingTable
from .diagnostics import Diagnostic
from .ir.modules import QuantumModule
from .ir.operations import FrozenAttributes, canonical_value


class ImportStatus(str, Enum):
    SUPPORTED_EXACT = "supported_exact"
    UNSUPPORTED_WITH_DIAGNOSTICS = "unsupported_with_diagnostics"
    INVALID_INPUT = "invalid_input"


@dataclass(frozen=True)
class InternalObservableRequest:
    name: str
    wires: tuple[int, ...]
    coefficient: object
    metadata: FrozenAttributes = FrozenAttributes()

    def canonical(self) -> dict[str, object]:
        return {
            "name": self.name,
            "wires": list(self.wires),
            "coefficient": canonical_value(self.coefficient),
            "metadata": canonical_value(self.metadata),
        }


@dataclass(frozen=True)
class InternalMeasurementRequest:
    kind: str
    wires: tuple[int, ...]
    shots: int | None
    seed: int | None = None
    name: str | None = None
    format: str | None = None
    postselection: FrozenAttributes | None = None
    max_marginal_wires: int | None = None
    max_postselection_draw_multiplier: int | None = None

    def canonical(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "wires": list(self.wires),
            "shots": self.shots,
            "seed": self.seed,
            "name": self.name,
            "format": self.format,
            "postselection": (
                canonical_value(self.postselection) if self.postselection else None
            ),
            "max_marginal_wires": self.max_marginal_wires,
            "max_postselection_draw_multiplier": (
                self.max_postselection_draw_multiplier
            ),
        }


@dataclass(frozen=True)
class InternalExecutionRequest:
    observables: tuple[InternalObservableRequest, ...] = ()
    measurements: tuple[InternalMeasurementRequest, ...] = ()
    classical_width: int | None = None

    def canonical(self) -> dict[str, object]:
        return {
            "observables": [item.canonical() for item in self.observables],
            "measurements": [item.canonical() for item in self.measurements],
            "classical_width": self.classical_width,
        }


@dataclass(frozen=True)
class ImportConstraints:
    dtype: str
    shape: tuple[int, ...]
    batch_size: int | None = None
    logical_state_shape: tuple[int, ...] | None = None
    runtime_config: FrozenAttributes | None = None

    def __post_init__(self) -> None:
        dtype = str(self.dtype).removeprefix("torch.")
        shape = tuple(int(size) for size in self.shape)
        if dtype not in {"complex64", "complex128"}:
            raise ValueError(f"unsupported import dtype {dtype!r}")
        if not shape or any(size <= 0 for size in shape):
            raise ValueError("import shape must contain positive dimensions")
        if self.batch_size is not None and int(self.batch_size) <= 0:
            raise ValueError("import batch_size must be positive")
        object.__setattr__(self, "dtype", dtype)
        object.__setattr__(self, "shape", shape)
        if self.batch_size is not None:
            object.__setattr__(self, "batch_size", int(self.batch_size))

    def semantic_canonical(self) -> dict[str, object]:
        return {
            "dtype": self.dtype,
            "shape": list(self.shape),
            "batch_size": self.batch_size,
            "logical_state_shape": (
                list(self.logical_state_shape) if self.logical_state_shape else None
            ),
            "runtime_config": (
                canonical_value(self.runtime_config) if self.runtime_config else None
            ),
        }


@dataclass(frozen=True)
class SourceIdentity:
    schema_version: str
    circuit_ir_content_hash: str


@dataclass(frozen=True)
class SourceProvenance:
    values: FrozenAttributes = FrozenAttributes()


@dataclass(frozen=True)
class InstructionSemantics:
    instruction_index: int
    values: FrozenAttributes

    def canonical(self) -> dict[str, object]:
        return {
            "instruction_index": self.instruction_index,
            "values": canonical_value(self.values),
        }


@dataclass(frozen=True)
class ImportedCircuitProgram:
    module: QuantumModule
    request: InternalExecutionRequest
    constraints: ImportConstraints
    source: SourceIdentity
    provenance: SourceProvenance
    instruction_semantics: tuple[InstructionSemantics, ...]
    bindings: BindingTable

    @property
    def internal_program_identity(self) -> str:
        payload = {
            "module": self.module.canonical(),
            "constraints": self.constraints.semantic_canonical(),
            "instruction_semantics": [
                item.canonical() for item in self.instruction_semantics
            ],
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CircuitImportResult:
    status: ImportStatus
    imported: ImportedCircuitProgram | None = None
    diagnostics: tuple[Diagnostic, ...] = ()

    @property
    def ok(self) -> bool:
        return self.status is ImportStatus.SUPPORTED_EXACT


__all__ = [
    "CircuitImportResult",
    "ImportConstraints",
    "ImportedCircuitProgram",
    "ImportStatus",
    "InstructionSemantics",
    "InternalExecutionRequest",
    "InternalMeasurementRequest",
    "InternalObservableRequest",
    "SourceIdentity",
    "SourceProvenance",
]

"""Typed, versioned cross-layer runtime contracts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, fields
from typing import Any, ClassVar, Literal, Mapping, TypeVar

CONTRACT_VERSION = "1.0"
DistributionSemantics = Literal[
    "single_device_fast_path",
    "sharded_across_ranks",
    "rank_local_replicated_kernel",
    "manual_sliced_tensor_contraction",
    "observable_term_parallel",
    "data_parallel_replicated",
    "replicated_per_rank",
]


class ContractError(ValueError):
    """Base error for malformed or unsupported runtime contracts."""


class UnknownContractFieldError(ContractError):
    """Production input contains a field absent from the typed schema."""


class ContractVersionError(ContractError):
    """Contract version requires an explicit migration or is unsupported."""


T = TypeVar("T", bound="VersionedContract")


def _encode(value: Any) -> Any:
    if isinstance(value, VersionedContract):
        return {
            "kind": value.KIND,
            **{item.name: _encode(getattr(value, item.name)) for item in fields(value)},
        }
    if isinstance(value, tuple):
        return [_encode(item) for item in value]
    return value


def _strict_values(cls: type[T], payload: Mapping[str, Any]) -> dict[str, Any]:
    allowed = {item.name for item in fields(cls)} | {"kind"}
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise UnknownContractFieldError(
            f"unknown {cls.KIND} field(s): {', '.join(unknown)}"
        )
    if payload.get("kind") != cls.KIND:
        raise ContractError(f"expected contract kind {cls.KIND!r}")
    version = payload.get("version")
    if version != CONTRACT_VERSION:
        raise ContractVersionError(
            f"unsupported {cls.KIND} version {version!r}; migrate explicitly to "
            f"{CONTRACT_VERSION!r}"
        )
    return {key: value for key, value in payload.items() if key != "kind"}


@dataclass(frozen=True)
class VersionedContract:
    version: str = CONTRACT_VERSION
    KIND: ClassVar[str] = "contract"

    def __post_init__(self) -> None:
        if self.version != CONTRACT_VERSION:
            raise ContractVersionError(
                f"unsupported {self.KIND} version {self.version!r}"
            )

    def to_dict(self) -> dict[str, Any]:
        encoded = _encode(self)
        assert isinstance(encoded, dict)
        return encoded

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
        )

    @property
    def content_hash(self) -> str:
        return hashlib.sha256(self.to_json().encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CapabilityContract(VersionedContract):
    backend: str = "pytorch"
    devices: tuple[str, ...] = ("cpu",)
    dtypes: tuple[str, ...] = ("complex64",)
    modes: tuple[str, ...] = ("statevector",)
    supports_autograd: bool = True
    supports_distributed: bool = False
    KIND: ClassVar[str] = "capability"


@dataclass(frozen=True)
class RequestedExecution(VersionedContract):
    mode: str = "statevector"
    backend: str = "pytorch"
    device: str = "cpu"
    dtype: str = "complex64"
    world_size: int = 1
    KIND: ClassVar[str] = "requested_execution"


@dataclass(frozen=True)
class EstimatedResources(VersionedContract):
    memory_bytes: int = 0
    communication_bytes: int = 0
    depth: int = 0
    KIND: ClassVar[str] = "estimated_resources"


@dataclass(frozen=True)
class OwnershipContract(VersionedContract):
    rank_owners: tuple[tuple[str, int], ...] = ()
    distribution_semantics: DistributionSemantics = "single_device_fast_path"
    KIND: ClassVar[str] = "ownership"


@dataclass(frozen=True)
class MeasurementContract(VersionedContract):
    kind_name: str = "expectation"
    wires: tuple[int, ...] = ()
    shots: int | None = None
    values: tuple[float, ...] = ()
    KIND: ClassVar[str] = "measurement"


@dataclass(frozen=True)
class AccuracyContract(VersionedContract):
    metric: str = "not_measured"
    value: float | None = None
    tolerance: float | None = None
    passed: bool | None = None
    KIND: ClassVar[str] = "accuracy"


@dataclass(frozen=True)
class FailureContract(VersionedContract):
    code: str = ""
    message: str = ""
    retryable: bool = False
    blockers: tuple[str, ...] = ()
    KIND: ClassVar[str] = "failure"


@dataclass(frozen=True)
class ProvenanceContract(VersionedContract):
    commit: str = ""
    workload_sha256: str = ""
    command: tuple[str, ...] = ()
    devices: tuple[str, ...] = ()
    rank_mapping: tuple[str, ...] = ()
    raw_log_sha256: str = ""
    KIND: ClassVar[str] = "provenance"


@dataclass(frozen=True)
class ExecutionObservation(VersionedContract):
    executor: str = ""
    elapsed_seconds: float = 0.0
    peak_memory_bytes: int = 0
    communication_bytes: int = 0
    completed: bool = False
    KIND: ClassVar[str] = "execution_observation"


@dataclass(frozen=True)
class AuditValidation(VersionedContract):
    validator: str = ""
    passed: bool = False
    release_claim_approved: bool = False
    errors: tuple[str, ...] = ()
    KIND: ClassVar[str] = "audit_validation"


@dataclass(frozen=True)
class RuntimePlanContract(VersionedContract):
    requested: RequestedExecution = RequestedExecution()
    estimated: EstimatedResources = EstimatedResources()
    capability: CapabilityContract = CapabilityContract()
    plan_id: str = ""
    KIND: ClassVar[str] = "runtime_plan"

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RuntimePlanContract":
        values = _strict_values(cls, payload)
        values["requested"] = RequestedExecution(
            **_strict_values(RequestedExecution, values["requested"])
        )
        values["estimated"] = EstimatedResources(
            **_strict_values(EstimatedResources, values["estimated"])
        )
        capability = _strict_values(CapabilityContract, values["capability"])
        for name in ("devices", "dtypes", "modes"):
            capability[name] = tuple(capability[name])
        values["capability"] = CapabilityContract(**capability)
        return cls(**values)


@dataclass(frozen=True)
class ExecutionRecordContract(VersionedContract):
    plan_id: str = ""
    observed: ExecutionObservation = ExecutionObservation()
    ownership: OwnershipContract = OwnershipContract()
    measurements: tuple[MeasurementContract, ...] = ()
    accuracy: AccuracyContract = AccuracyContract()
    failures: tuple[FailureContract, ...] = ()
    provenance: ProvenanceContract = ProvenanceContract()
    KIND: ClassVar[str] = "execution_record"

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ExecutionRecordContract":
        values = _strict_values(cls, payload)
        values["observed"] = ExecutionObservation(
            **_strict_values(ExecutionObservation, values["observed"])
        )
        ownership = _strict_values(OwnershipContract, values["ownership"])
        ownership["rank_owners"] = tuple(
            tuple(item) for item in ownership["rank_owners"]
        )
        values["ownership"] = OwnershipContract(**ownership)
        measurements = []
        for item in values["measurements"]:
            measurement = _strict_values(MeasurementContract, item)
            measurement["wires"] = tuple(measurement["wires"])
            measurement["values"] = tuple(measurement["values"])
            measurements.append(MeasurementContract(**measurement))
        values["measurements"] = tuple(measurements)
        values["accuracy"] = AccuracyContract(
            **_strict_values(AccuracyContract, values["accuracy"])
        )
        failures = []
        for item in values["failures"]:
            failure = _strict_values(FailureContract, item)
            failure["blockers"] = tuple(failure["blockers"])
            failures.append(FailureContract(**failure))
        values["failures"] = tuple(failures)
        provenance = _strict_values(ProvenanceContract, values["provenance"])
        for name in ("command", "devices", "rank_mapping"):
            provenance[name] = tuple(provenance[name])
        values["provenance"] = ProvenanceContract(**provenance)
        return cls(**values)


@dataclass(frozen=True)
class AuditRecordContract(VersionedContract):
    execution_hash: str = ""
    validated: AuditValidation = AuditValidation()
    KIND: ClassVar[str] = "audit_record"

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "AuditRecordContract":
        values = _strict_values(cls, payload)
        validated = _strict_values(AuditValidation, values["validated"])
        validated["errors"] = tuple(validated["errors"])
        values["validated"] = AuditValidation(**validated)
        return cls(**values)


def migrate_contract(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Explicitly migrate the documented 0.9 plan envelope to version 1.0."""

    source = dict(payload)
    if source.get("version") != "0.9" or source.get("kind") != "runtime_plan":
        raise ContractVersionError(
            "no migration is registered; supported legacy input is runtime_plan 0.9"
        )
    required = {"kind", "version", "plan_id", "request", "estimate", "capability"}
    unknown = sorted(set(source) - required)
    if unknown:
        raise UnknownContractFieldError(
            f"unknown legacy runtime_plan field(s): {', '.join(unknown)}"
        )
    return {
        "kind": "runtime_plan",
        "version": CONTRACT_VERSION,
        "plan_id": source["plan_id"],
        "requested": {
            "kind": "requested_execution",
            **source["request"],
            "version": CONTRACT_VERSION,
        },
        "estimated": {
            "kind": "estimated_resources",
            **source["estimate"],
            "version": CONTRACT_VERSION,
        },
        "capability": {
            "kind": "capability",
            **source["capability"],
            "version": CONTRACT_VERSION,
        },
    }


CONTRACT_TYPES = (
    CapabilityContract,
    RequestedExecution,
    EstimatedResources,
    OwnershipContract,
    MeasurementContract,
    AccuracyContract,
    FailureContract,
    ProvenanceContract,
    ExecutionObservation,
    AuditValidation,
    RuntimePlanContract,
    ExecutionRecordContract,
    AuditRecordContract,
)


__all__ = [item.__name__ for item in CONTRACT_TYPES] + [
    "CONTRACT_TYPES",
    "CONTRACT_VERSION",
    "ContractError",
    "ContractVersionError",
    "DistributionSemantics",
    "UnknownContractFieldError",
    "migrate_contract",
]

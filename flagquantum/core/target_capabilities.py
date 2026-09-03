"""Internal, backend-neutral Target Capabilities v1 contracts.

This module deliberately is not re-exported from :mod:`flagquantum.core` or the
package root.  It is the Core-owned contract seam used by future adapters; it
does not change the existing compiler/runtime capability types or selection
behaviour.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Iterable, Iterator, Mapping, TypeVar

TARGET_CAPABILITIES_SCHEMA_VERSION = "1.0"

CAPABILITY_NAMES = frozenset(
    {
        "target.class",
        "device.kind",
        "device.count",
        "memory.available_bytes",
        "qubits.logical_capacity",
        "qubits.physical_capacity",
        "gates.native",
        "measurements.results",
        "artifacts.profiles",
        "limits.maximum_shots",
        "limits.maximum_program_operations",
        "ancillas.policy",
        "ancillas.maximum_compiler",
        "precision.native_dtype",
        "precision.effective_dtype",
        "precision.storage_dtype",
        "precision.parameter_dtype",
        "precision.accumulator_dtype",
        "precision.software_mechanism",
    }
)

AUTHORITATIVE_STATIC_DECLARATION_ALLOWED = frozenset(
    {
        "target.class",
        "qubits.logical_capacity",
        "qubits.physical_capacity",
        "gates.native",
        "measurements.results",
        "artifacts.profiles",
        "limits.maximum_shots",
        "limits.maximum_program_operations",
        "ancillas.policy",
        "ancillas.maximum_compiler",
    }
)

OBSERVED_REQUIRED = frozenset(
    {
        "device.kind",
        "device.count",
        "memory.available_bytes",
        "precision.native_dtype",
        "precision.effective_dtype",
        "precision.storage_dtype",
        "precision.parameter_dtype",
        "precision.accumulator_dtype",
        "precision.software_mechanism",
    }
)


class CapabilityContractError(ValueError):
    """Base error for an invalid Target Capabilities v1 value object."""


class UnknownCapabilityFieldError(CapabilityContractError):
    """A JSON object contains a field outside its closed schema."""


class CapabilityVersionError(CapabilityContractError):
    """A value object uses a schema version requiring explicit migration."""


class CapabilityIdentityError(CapabilityContractError):
    """A supplied canonical identity does not match its semantic payload."""


class ComparisonOperator(str, Enum):
    EQUALS = "equals"
    AT_LEAST = "at_least"
    AT_MOST = "at_most"
    CONTAINS_ALL = "contains_all"
    COVERS = "covers"


class RequirementStrength(str, Enum):
    MANDATORY = "mandatory"
    PREFERENCE = "preference"


class RequirementSource(str, Enum):
    USER = "user"
    COMPILER = "compiler"
    RUNTIME_PROTOCOL = "runtime_protocol"


class EvidenceLevel(str, Enum):
    BASIC = "basic"
    OBSERVABLE = "observable"
    CERTIFICATION = "certification"


class SupportStatus(str, Enum):
    UNKNOWN = "unknown"
    UNMEASURED = "unmeasured"
    UNSUPPORTED = "unsupported"
    VERIFIED = "verified"


class FactExposure(str, Enum):
    OBSERVED = "observed"
    DECLARED = "declared"
    NOT_EXPOSED = "not_exposed"
    UNKNOWN = "unknown"
    NOT_APPLICABLE = "not_applicable"


class MatchBlockerCode(str, Enum):
    INVALID_REQUIREMENT_SET_ID = "invalid_requirement_set_id"
    INVALID_SNAPSHOT_ID = "invalid_snapshot_id"
    TARGET_IDENTITY_MISMATCH = "target_identity_mismatch"
    SCOPE_MISMATCH = "scope_mismatch"
    SNAPSHOT_NOT_YET_VALID = "snapshot_not_yet_valid"
    SNAPSHOT_STALE = "snapshot_stale"
    UNRESOLVED_EVIDENCE_REFERENCE = "unresolved_evidence_reference"
    UNKNOWN_EXTENSION_HANDLER = "unknown_extension_handler"
    MISSING_FACT = "missing_fact"
    FACT_UNKNOWN = "fact_unknown"
    FACT_UNMEASURED = "fact_unmeasured"
    FACT_UNSUPPORTED = "fact_unsupported"
    EXPOSURE_NOT_ACCEPTED = "exposure_not_accepted"
    EXPOSURE_REQUIRES_OBSERVATION = "exposure_requires_observation"
    DECLARATION_NOT_AUTHORITATIVE = "declaration_not_authoritative"
    NOT_APPLICABLE = "not_applicable"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    VALUE_MISMATCH = "value_mismatch"


_EVIDENCE_RANK = {
    EvidenceLevel.BASIC: 0,
    EvidenceLevel.OBSERVABLE: 1,
    EvidenceLevel.CERTIFICATION: 2,
}
_REVERSE_DNS = re.compile(
    r"^[a-z0-9](?:[a-z0-9-]*[a-z0-9])?(?:\.[a-z0-9](?:[a-z0-9-]*[a-z0-9])?)+$"
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class _FrozenJSONObject(Mapping[str, Any]):
    items_tuple: tuple[tuple[str, Any], ...]

    def __iter__(self) -> Iterator[str]:
        return (key for key, _ in self.items_tuple)

    def __len__(self) -> int:
        return len(self.items_tuple)

    def __getitem__(self, key: str) -> Any:
        for item_key, value in self.items_tuple:
            if item_key == key:
                return value
        raise KeyError(key)


def _freeze_json(value: Any, *, path: str) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise CapabilityContractError(f"{path} must not contain NaN or infinity")
        return value
    if isinstance(value, Mapping):
        frozen: list[tuple[str, Any]] = []
        if any(not isinstance(key, str) for key in value):
            raise CapabilityContractError(f"{path} object keys must be strings")
        for key in sorted(value):
            frozen.append((key, _freeze_json(value[key], path=f"{path}.{key}")))
        return _FrozenJSONObject(tuple(frozen))
    if isinstance(value, (list, tuple)):
        return tuple(
            _freeze_json(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        )
    raise CapabilityContractError(
        f"{path} must contain only JSON-safe immutable values, got {type(value).__name__}"
    )


def _thaw_json(value: Any) -> Any:
    if isinstance(value, _FrozenJSONObject):
        return {key: _thaw_json(item) for key, item in value.items_tuple}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _strict_fields(
    payload: Mapping[str, Any], *, allowed: set[str], kind: str
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise CapabilityContractError(f"{kind} must be a JSON object")
    unknown = sorted(set(payload) - allowed)
    missing = sorted(allowed - set(payload))
    if unknown:
        raise UnknownCapabilityFieldError(
            f"unknown {kind} field(s): {', '.join(unknown)}"
        )
    if missing:
        raise CapabilityContractError(f"missing {kind} field(s): {', '.join(missing)}")
    return dict(payload)


E = TypeVar("E", bound=Enum)


def _enum(enum_type: type[E], value: Any, *, field_name: str) -> E:
    try:
        return enum_type(value)
    except (TypeError, ValueError) as error:
        allowed = ", ".join(item.value for item in enum_type)
        raise CapabilityContractError(
            f"{field_name} must be one of: {allowed}"
        ) from error


def _require_non_empty_string(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise CapabilityContractError(f"{field_name} must be a non-empty string")
    return value


def _parse_timestamp(value: str, *, field_name: str) -> datetime:
    if not isinstance(value, str):
        raise CapabilityContractError(f"{field_name} must be an ISO-8601 string")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (TypeError, ValueError) as error:
        raise CapabilityContractError(
            f"{field_name} must be an ISO-8601 timestamp"
        ) from error
    if parsed.tzinfo is None:
        raise CapabilityContractError(f"{field_name} must include a timezone")
    return parsed.astimezone(timezone.utc)


def _validate_capability_value(
    name: str, value: Any, operator: ComparisonOperator
) -> Any:
    value = _freeze_json(value, path=f"capability[{name}]")
    numeric_names = {
        "device.count",
        "memory.available_bytes",
        "qubits.logical_capacity",
        "qubits.physical_capacity",
        "limits.maximum_shots",
        "limits.maximum_program_operations",
        "ancillas.maximum_compiler",
    }
    collection_names = {"gates.native", "measurements.results", "artifacts.profiles"}
    if name in numeric_names and (
        not isinstance(value, int) or isinstance(value, bool) or value < 0
    ):
        raise CapabilityContractError(f"{name} must be a non-negative integer")
    if name in collection_names and not isinstance(value, tuple):
        raise CapabilityContractError(f"{name} must be a JSON array")
    if name not in numeric_names | collection_names and not isinstance(value, str):
        raise CapabilityContractError(f"{name} must be a string")
    if (
        operator in {ComparisonOperator.AT_LEAST, ComparisonOperator.AT_MOST}
        and name not in numeric_names
    ):
        raise CapabilityContractError(f"{operator.value} requires a numeric capability")
    if operator is ComparisonOperator.CONTAINS_ALL and not isinstance(value, tuple):
        raise CapabilityContractError("contains_all requires an array value")
    return value


@dataclass(frozen=True)
class CapabilityBlocker:
    code: str
    message: str
    capability_name: str | None = None

    def __post_init__(self) -> None:
        _require_non_empty_string(self.code, field_name="blocker.code")
        _require_non_empty_string(self.message, field_name="blocker.message")
        if (
            self.capability_name is not None
            and self.capability_name not in CAPABILITY_NAMES
        ):
            raise CapabilityContractError(
                "blocker.capability_name is not in the v1 closed set"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "capability_name": self.capability_name,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CapabilityBlocker":
        values = _strict_fields(
            payload,
            allowed={"code", "message", "capability_name"},
            kind="capability blocker",
        )
        return cls(**values)


@dataclass(frozen=True)
class CapabilityScope:
    device_ids: tuple[str, ...] | None = None
    dtype: str | None = None
    kernel: str | None = None
    workload_id: str | None = None
    world_size: int | None = None
    node_count: int | None = None

    def __post_init__(self) -> None:
        if self.device_ids is not None:
            normalized = tuple(self.device_ids)
            if any(not isinstance(item, str) or not item for item in normalized):
                raise CapabilityContractError(
                    "scope.device_ids must contain non-empty strings"
                )
            if len(set(normalized)) != len(normalized):
                raise CapabilityContractError(
                    "scope.device_ids must not contain duplicates"
                )
            object.__setattr__(self, "device_ids", tuple(sorted(normalized)))
        for name in ("dtype", "kernel", "workload_id"):
            value = getattr(self, name)
            if value is not None:
                _require_non_empty_string(value, field_name=f"scope.{name}")
        for name in ("world_size", "node_count"):
            value = getattr(self, name)
            if value is not None and (
                not isinstance(value, int) or isinstance(value, bool) or value < 1
            ):
                raise CapabilityContractError(
                    f"scope.{name} must be a positive integer"
                )

    def to_dict(self) -> dict[str, Any]:
        return {
            "device_ids": None if self.device_ids is None else list(self.device_ids),
            "dtype": self.dtype,
            "kernel": self.kernel,
            "workload_id": self.workload_id,
            "world_size": self.world_size,
            "node_count": self.node_count,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CapabilityScope":
        values = _strict_fields(
            payload,
            allowed={
                "device_ids",
                "dtype",
                "kernel",
                "workload_id",
                "world_size",
                "node_count",
            },
            kind="capability scope",
        )
        if values["device_ids"] is not None:
            if not isinstance(values["device_ids"], list):
                raise CapabilityContractError(
                    "scope.device_ids must be an array or null"
                )
            values["device_ids"] = tuple(values["device_ids"])
        return cls(**values)


@dataclass(frozen=True)
class TargetIdentity:
    target_id: str
    target_class: str
    provider: str
    provider_version: str
    target_revision: str
    environment_id: str

    def __post_init__(self) -> None:
        for name in (
            "target_id",
            "target_class",
            "provider",
            "provider_version",
            "target_revision",
            "environment_id",
        ):
            _require_non_empty_string(
                getattr(self, name), field_name=f"target_identity.{name}"
            )

    def to_dict(self) -> dict[str, str]:
        return {
            "target_id": self.target_id,
            "target_class": self.target_class,
            "provider": self.provider,
            "provider_version": self.provider_version,
            "target_revision": self.target_revision,
            "environment_id": self.environment_id,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TargetIdentity":
        return cls(
            **_strict_fields(
                payload,
                allowed={
                    "target_id",
                    "target_class",
                    "provider",
                    "provider_version",
                    "target_revision",
                    "environment_id",
                },
                kind="target identity",
            )
        )


@dataclass(frozen=True)
class FactSource:
    kind: str
    ref: str

    def __post_init__(self) -> None:
        _require_non_empty_string(self.kind, field_name="fact_source.kind")
        _require_non_empty_string(self.ref, field_name="fact_source.ref")

    def to_dict(self) -> dict[str, str]:
        return {"kind": self.kind, "ref": self.ref}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FactSource":
        return cls(
            **_strict_fields(payload, allowed={"kind", "ref"}, kind="fact source")
        )


@dataclass(frozen=True)
class EvidenceReference:
    evidence_id: str
    sha256: str
    level: EvidenceLevel
    scope: CapabilityScope

    def __post_init__(self) -> None:
        _require_non_empty_string(
            self.evidence_id, field_name="evidence_ref.evidence_id"
        )
        if not _SHA256.fullmatch(self.sha256):
            raise CapabilityContractError(
                "evidence_ref.sha256 must be 64 lowercase hex characters"
            )
        object.__setattr__(
            self,
            "level",
            _enum(EvidenceLevel, self.level, field_name="evidence_ref.level"),
        )
        if not isinstance(self.scope, CapabilityScope):
            raise CapabilityContractError(
                "evidence_ref.scope must be a CapabilityScope"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "sha256": self.sha256,
            "level": self.level.value,
            "scope": self.scope.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "EvidenceReference":
        values = _strict_fields(
            payload,
            allowed={"evidence_id", "sha256", "level", "scope"},
            kind="evidence reference",
        )
        values["level"] = _enum(
            EvidenceLevel, values["level"], field_name="evidence_ref.level"
        )
        values["scope"] = CapabilityScope.from_dict(values["scope"])
        return cls(**values)


@dataclass(frozen=True)
class CapabilityFact:
    name: str
    value: Any
    support_status: SupportStatus
    fact_exposure: FactExposure
    source: FactSource
    blockers: tuple[CapabilityBlocker, ...] = ()

    def __post_init__(self) -> None:
        if self.name not in CAPABILITY_NAMES:
            raise CapabilityContractError(f"unknown v1 capability name {self.name!r}")
        object.__setattr__(
            self,
            "value",
            _validate_capability_value(
                self.name, self.value, ComparisonOperator.EQUALS
            ),
        )
        object.__setattr__(
            self,
            "support_status",
            _enum(SupportStatus, self.support_status, field_name="fact.support_status"),
        )
        object.__setattr__(
            self,
            "fact_exposure",
            _enum(FactExposure, self.fact_exposure, field_name="fact.fact_exposure"),
        )
        if not isinstance(self.source, FactSource):
            raise CapabilityContractError("fact.source must be a FactSource")
        blockers = tuple(self.blockers)
        if any(not isinstance(item, CapabilityBlocker) for item in blockers):
            raise CapabilityContractError(
                "fact.blockers must contain CapabilityBlocker values"
            )
        if self.support_status is not SupportStatus.VERIFIED and not blockers:
            raise CapabilityContractError(
                "non-verified facts require at least one blocker"
            )
        if self.fact_exposure is FactExposure.NOT_APPLICABLE and not blockers:
            raise CapabilityContractError("not_applicable facts require a blocker")
        object.__setattr__(self, "blockers", tuple(sorted(blockers, key=_blocker_key)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "value": _thaw_json(self.value),
            "support_status": self.support_status.value,
            "fact_exposure": self.fact_exposure.value,
            "source": self.source.to_dict(),
            "blockers": [item.to_dict() for item in self.blockers],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CapabilityFact":
        values = _strict_fields(
            payload,
            allowed={
                "name",
                "value",
                "support_status",
                "fact_exposure",
                "source",
                "blockers",
            },
            kind="capability fact",
        )
        if not isinstance(values["blockers"], list):
            raise CapabilityContractError("fact.blockers must be an array")
        values["support_status"] = _enum(
            SupportStatus, values["support_status"], field_name="fact.support_status"
        )
        values["fact_exposure"] = _enum(
            FactExposure, values["fact_exposure"], field_name="fact.fact_exposure"
        )
        values["source"] = FactSource.from_dict(values["source"])
        values["blockers"] = tuple(
            CapabilityBlocker.from_dict(item) for item in values["blockers"]
        )
        return cls(**values)


@dataclass(frozen=True)
class CapabilityRequirement:
    name: str
    operator: ComparisonOperator
    value: Any
    strength: RequirementStrength
    source: RequirementSource
    minimum_evidence_level: EvidenceLevel
    accepted_exposures: tuple[FactExposure, ...]

    def __post_init__(self) -> None:
        if self.name not in CAPABILITY_NAMES:
            raise CapabilityContractError(f"unknown v1 capability name {self.name!r}")
        operator = _enum(
            ComparisonOperator, self.operator, field_name="requirement.operator"
        )
        object.__setattr__(self, "operator", operator)
        object.__setattr__(
            self, "value", _validate_capability_value(self.name, self.value, operator)
        )
        object.__setattr__(
            self,
            "strength",
            _enum(
                RequirementStrength, self.strength, field_name="requirement.strength"
            ),
        )
        object.__setattr__(
            self,
            "source",
            _enum(RequirementSource, self.source, field_name="requirement.source"),
        )
        object.__setattr__(
            self,
            "minimum_evidence_level",
            _enum(
                EvidenceLevel,
                self.minimum_evidence_level,
                field_name="requirement.minimum_evidence_level",
            ),
        )
        exposures = tuple(
            _enum(FactExposure, item, field_name="requirement.accepted_exposures")
            for item in self.accepted_exposures
        )
        if not exposures:
            raise CapabilityContractError("accepted_exposures must not be empty")
        object.__setattr__(
            self,
            "accepted_exposures",
            tuple(sorted(set(exposures), key=lambda item: item.value)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "operator": self.operator.value,
            "value": _thaw_json(self.value),
            "strength": self.strength.value,
            "source": self.source.value,
            "minimum_evidence_level": self.minimum_evidence_level.value,
            "accepted_exposures": [item.value for item in self.accepted_exposures],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CapabilityRequirement":
        values = _strict_fields(
            payload,
            allowed={
                "name",
                "operator",
                "value",
                "strength",
                "source",
                "minimum_evidence_level",
                "accepted_exposures",
            },
            kind="capability requirement",
        )
        if not isinstance(values["accepted_exposures"], list):
            raise CapabilityContractError("accepted_exposures must be an array")
        values["operator"] = _enum(
            ComparisonOperator, values["operator"], field_name="requirement.operator"
        )
        values["strength"] = _enum(
            RequirementStrength, values["strength"], field_name="requirement.strength"
        )
        values["source"] = _enum(
            RequirementSource, values["source"], field_name="requirement.source"
        )
        values["minimum_evidence_level"] = _enum(
            EvidenceLevel,
            values["minimum_evidence_level"],
            field_name="requirement.minimum_evidence_level",
        )
        values["accepted_exposures"] = tuple(values["accepted_exposures"])
        return cls(**values)

    @property
    def canonical_key(self) -> str:
        return _canonical_json(self.to_dict())


@dataclass(frozen=True)
class FallbackAuthorizations:
    backend: bool = False
    device: bool = False
    cpu: bool = False
    precision: bool = False
    algorithm: bool = False
    approximation: bool = False

    def __post_init__(self) -> None:
        for name in (
            "backend",
            "device",
            "cpu",
            "precision",
            "algorithm",
            "approximation",
        ):
            if type(getattr(self, name)) is not bool:
                raise CapabilityContractError(
                    f"fallback_authorizations.{name} must be boolean"
                )

    def to_dict(self) -> dict[str, bool]:
        return {
            "backend": self.backend,
            "device": self.device,
            "cpu": self.cpu,
            "precision": self.precision,
            "algorithm": self.algorithm,
            "approximation": self.approximation,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FallbackAuthorizations":
        return cls(
            **_strict_fields(
                payload,
                allowed={
                    "backend",
                    "device",
                    "cpu",
                    "precision",
                    "algorithm",
                    "approximation",
                },
                kind="fallback authorizations",
            )
        )


@dataclass(frozen=True, kw_only=True)
class RequirementSet:
    schema_version: str = TARGET_CAPABILITIES_SCHEMA_VERSION
    requirement_set_id: str = ""
    requirements: tuple[CapabilityRequirement, ...]
    fallback_authorizations: FallbackAuthorizations = FallbackAuthorizations()
    extensions: Any = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.schema_version != TARGET_CAPABILITIES_SCHEMA_VERSION:
            raise CapabilityVersionError(
                f"unsupported requirement set version {self.schema_version!r}"
            )
        requirements = tuple(self.requirements)
        if any(not isinstance(item, CapabilityRequirement) for item in requirements):
            raise CapabilityContractError(
                "requirements must contain CapabilityRequirement values"
            )
        unique = {item.canonical_key: item for item in requirements}
        requirements = tuple(unique[key] for key in sorted(unique))
        _reject_conflicting_mandatory_requirements(requirements)
        object.__setattr__(self, "requirements", requirements)
        if not isinstance(self.fallback_authorizations, FallbackAuthorizations):
            raise CapabilityContractError(
                "fallback_authorizations must be FallbackAuthorizations"
            )
        extensions = _freeze_extensions(self.extensions)
        object.__setattr__(self, "extensions", extensions)
        expected = _canonical_sha256(self._identity_payload())
        if self.requirement_set_id and self.requirement_set_id != expected:
            raise CapabilityIdentityError(
                "requirement_set_id does not match canonical semantic payload"
            )
        object.__setattr__(self, "requirement_set_id", expected)

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "requirements": [item.to_dict() for item in self.requirements],
            "fallback_authorizations": self.fallback_authorizations.to_dict(),
            "extensions": _thaw_json(self.extensions),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "requirement_set_id": self.requirement_set_id,
            **{
                key: value
                for key, value in self._identity_payload().items()
                if key != "schema_version"
            },
        }

    def to_json(self) -> str:
        return _canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RequirementSet":
        values = _strict_fields(
            payload,
            allowed={
                "schema_version",
                "requirement_set_id",
                "requirements",
                "fallback_authorizations",
                "extensions",
            },
            kind="requirement set",
        )
        if not isinstance(values["requirements"], list):
            raise CapabilityContractError("requirements must be an array")
        values["requirements"] = tuple(
            CapabilityRequirement.from_dict(item) for item in values["requirements"]
        )
        values["fallback_authorizations"] = FallbackAuthorizations.from_dict(
            values["fallback_authorizations"]
        )
        return cls(**values)

    @classmethod
    def from_json(cls, payload: str) -> "RequirementSet":
        try:
            decoded = json.loads(payload)
        except (TypeError, json.JSONDecodeError) as error:
            raise CapabilityContractError(
                "requirement set must be valid JSON"
            ) from error
        return cls.from_dict(decoded)


@dataclass(frozen=True, kw_only=True)
class TargetCapabilitySnapshot:
    schema_version: str = TARGET_CAPABILITIES_SCHEMA_VERSION
    snapshot_id: str = ""
    target_identity: TargetIdentity
    scope: CapabilityScope
    captured_at: str
    valid_until: str
    facts: tuple[CapabilityFact, ...]
    evidence_refs: tuple[EvidenceReference, ...]
    blockers: tuple[CapabilityBlocker, ...] = ()
    extensions: Any = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.schema_version != TARGET_CAPABILITIES_SCHEMA_VERSION:
            raise CapabilityVersionError(
                f"unsupported target capability snapshot version {self.schema_version!r}"
            )
        if not isinstance(self.target_identity, TargetIdentity):
            raise CapabilityContractError("target_identity must be a TargetIdentity")
        if not isinstance(self.scope, CapabilityScope):
            raise CapabilityContractError("scope must be a CapabilityScope")
        captured = _parse_timestamp(self.captured_at, field_name="captured_at")
        valid_until = _parse_timestamp(self.valid_until, field_name="valid_until")
        if valid_until <= captured:
            raise CapabilityContractError("valid_until must be later than captured_at")
        facts = tuple(self.facts)
        if any(not isinstance(item, CapabilityFact) for item in facts):
            raise CapabilityContractError("facts must contain CapabilityFact values")
        if len({item.name for item in facts}) != len(facts):
            raise CapabilityContractError(
                "facts must contain at most one fact per capability name"
            )
        object.__setattr__(
            self, "facts", tuple(sorted(facts, key=lambda item: item.name))
        )
        evidence_refs = tuple(self.evidence_refs)
        if any(not isinstance(item, EvidenceReference) for item in evidence_refs):
            raise CapabilityContractError(
                "evidence_refs must contain EvidenceReference values"
            )
        if len({item.evidence_id for item in evidence_refs}) != len(evidence_refs):
            raise CapabilityContractError(
                "evidence_refs must contain unique evidence_id values"
            )
        object.__setattr__(
            self,
            "evidence_refs",
            tuple(sorted(evidence_refs, key=lambda item: item.evidence_id)),
        )
        blockers = tuple(self.blockers)
        if any(not isinstance(item, CapabilityBlocker) for item in blockers):
            raise CapabilityContractError(
                "snapshot blockers must contain CapabilityBlocker values"
            )
        object.__setattr__(self, "blockers", tuple(sorted(blockers, key=_blocker_key)))
        object.__setattr__(self, "extensions", _freeze_extensions(self.extensions))
        expected = _canonical_sha256(self._identity_payload())
        if self.snapshot_id and self.snapshot_id != expected:
            raise CapabilityIdentityError(
                "snapshot_id does not match canonical semantic payload"
            )
        object.__setattr__(self, "snapshot_id", expected)

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "target_identity": self.target_identity.to_dict(),
            "scope": self.scope.to_dict(),
            "captured_at": self.captured_at,
            "valid_until": self.valid_until,
            "facts": [item.to_dict() for item in self.facts],
            "evidence_refs": [item.to_dict() for item in self.evidence_refs],
            "blockers": [item.to_dict() for item in self.blockers],
            "extensions": _thaw_json(self.extensions),
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self._identity_payload()
        return {
            "schema_version": payload.pop("schema_version"),
            "snapshot_id": self.snapshot_id,
            **payload,
        }

    def to_json(self) -> str:
        return _canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TargetCapabilitySnapshot":
        values = _strict_fields(
            payload,
            allowed={
                "schema_version",
                "snapshot_id",
                "target_identity",
                "scope",
                "captured_at",
                "valid_until",
                "facts",
                "evidence_refs",
                "blockers",
                "extensions",
            },
            kind="target capability snapshot",
        )
        for sequence_name in ("facts", "evidence_refs", "blockers"):
            if not isinstance(values[sequence_name], list):
                raise CapabilityContractError(f"{sequence_name} must be an array")
        values["target_identity"] = TargetIdentity.from_dict(values["target_identity"])
        values["scope"] = CapabilityScope.from_dict(values["scope"])
        values["facts"] = tuple(
            CapabilityFact.from_dict(item) for item in values["facts"]
        )
        values["evidence_refs"] = tuple(
            EvidenceReference.from_dict(item) for item in values["evidence_refs"]
        )
        values["blockers"] = tuple(
            CapabilityBlocker.from_dict(item) for item in values["blockers"]
        )
        return cls(**values)

    @classmethod
    def from_json(cls, payload: str) -> "TargetCapabilitySnapshot":
        try:
            decoded = json.loads(payload)
        except (TypeError, json.JSONDecodeError) as error:
            raise CapabilityContractError(
                "target capability snapshot must be valid JSON"
            ) from error
        return cls.from_dict(decoded)


@dataclass(frozen=True)
class CapabilityMatchResult:
    executable: bool
    blockers: tuple[CapabilityBlocker, ...]
    satisfied_preferences: int
    total_preferences: int


def _compare_values(
    operator: ComparisonOperator, available: Any, required: Any
) -> bool:
    if operator is ComparisonOperator.EQUALS:
        return available == required
    if operator is ComparisonOperator.AT_LEAST:
        return available >= required
    if operator is ComparisonOperator.AT_MOST:
        return available <= required
    if operator is ComparisonOperator.CONTAINS_ALL:
        try:
            return all(item in available for item in required)
        except TypeError:
            return False
    if operator is ComparisonOperator.COVERS:
        return _covers(available, required)
    raise AssertionError(f"unhandled comparison operator {operator}")


def _covers(available: Any, required: Any) -> bool:
    if isinstance(required, _FrozenJSONObject):
        if not isinstance(available, _FrozenJSONObject):
            return False
        return all(
            key in available and _covers(available[key], value)
            for key, value in required.items_tuple
        )
    if isinstance(required, tuple):
        if not isinstance(available, tuple):
            return False
        return all(item in available for item in required)
    return available == required


def _freeze_extensions(value: Any) -> _FrozenJSONObject:
    frozen = _freeze_json(value, path="extensions")
    if not isinstance(frozen, _FrozenJSONObject):
        raise CapabilityContractError("extensions must be a JSON object")
    for namespace in frozen:
        if not _REVERSE_DNS.fullmatch(namespace):
            raise CapabilityContractError(
                f"extension namespace {namespace!r} is not registered reverse-DNS form"
            )
    return frozen


def _reject_conflicting_mandatory_requirements(
    requirements: Iterable[CapabilityRequirement],
) -> None:
    grouped: dict[str, list[CapabilityRequirement]] = {}
    for requirement in requirements:
        if requirement.strength is RequirementStrength.MANDATORY:
            grouped.setdefault(requirement.name, []).append(requirement)
    for name, group in grouped.items():
        equals = [
            item.value for item in group if item.operator is ComparisonOperator.EQUALS
        ]
        if len({_canonical_json({"value": _thaw_json(value)}) for value in equals}) > 1:
            raise CapabilityContractError(
                f"conflicting mandatory equals requirements for {name}"
            )
        lower = [
            item.value for item in group if item.operator is ComparisonOperator.AT_LEAST
        ]
        upper = [
            item.value for item in group if item.operator is ComparisonOperator.AT_MOST
        ]
        if lower and upper and max(lower) > min(upper):
            raise CapabilityContractError(f"conflicting mandatory bounds for {name}")
        if equals:
            candidate = equals[0]
            if any(
                not _compare_values(item.operator, candidate, item.value)
                for item in group
            ):
                raise CapabilityContractError(
                    f"conflicting mandatory requirements for {name}"
                )


def _blocker_key(blocker: CapabilityBlocker) -> tuple[str, str, str]:
    return (blocker.capability_name or "", blocker.code, blocker.message)


def match_target_capabilities(
    requirements: RequirementSet,
    snapshot: TargetCapabilitySnapshot,
    *,
    evaluated_at: datetime | None = None,
    expected_target_identity: TargetIdentity | None = None,
    required_scope: CapabilityScope | None = None,
    claim_minimum_evidence_level: EvidenceLevel = EvidenceLevel.BASIC,
) -> CapabilityMatchResult:
    """Compare a requirement set and snapshot through the pure matcher."""

    from ._target_capability_matching import match_target_capabilities as match

    return match(
        requirements,
        snapshot,
        evaluated_at=evaluated_at,
        expected_target_identity=expected_target_identity,
        required_scope=required_scope,
        claim_minimum_evidence_level=claim_minimum_evidence_level,
    )


__all__ = [
    "AUTHORITATIVE_STATIC_DECLARATION_ALLOWED",
    "CAPABILITY_NAMES",
    "OBSERVED_REQUIRED",
    "TARGET_CAPABILITIES_SCHEMA_VERSION",
    "CapabilityBlocker",
    "CapabilityContractError",
    "CapabilityFact",
    "CapabilityIdentityError",
    "CapabilityMatchResult",
    "CapabilityRequirement",
    "CapabilityScope",
    "CapabilityVersionError",
    "ComparisonOperator",
    "EvidenceLevel",
    "EvidenceReference",
    "FactExposure",
    "FactSource",
    "FallbackAuthorizations",
    "MatchBlockerCode",
    "RequirementSet",
    "RequirementSource",
    "RequirementStrength",
    "SupportStatus",
    "TargetCapabilitySnapshot",
    "TargetIdentity",
    "UnknownCapabilityFieldError",
    "match_target_capabilities",
]

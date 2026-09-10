"""Experimental read-only views over target-capability contracts."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from types import MappingProxyType
from typing import Any, Mapping, overload

from ..core import target_capabilities as _targets

_MAX_JSON_BYTES = 16 * 1024 * 1024
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _decode_json_object(payload: object, owner: str) -> dict[str, Any]:
    if type(payload) is not str:
        raise TypeError(f"{owner} must be JSON text")
    if len(payload.encode("utf-8")) > _MAX_JSON_BYTES:
        raise ValueError(f"{owner} exceeds maximum UTF-8 bytes {_MAX_JSON_BYTES}")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        decoded: dict[str, Any] = {}
        for key, value in pairs:
            if key in decoded:
                raise ValueError(f"duplicate {owner} field {key!r}")
            decoded[key] = value
        return decoded

    try:
        decoded = json.loads(payload, object_pairs_hook=reject_duplicates)
    except json.JSONDecodeError as error:
        raise _targets.CapabilityContractError(f"{owner} must be valid JSON") from error
    if not isinstance(decoded, dict):
        raise _targets.CapabilityContractError(f"{owner} must be a JSON object")
    return decoded


def _bounded_output(payload: str, owner: str) -> str:
    if len(payload.encode("utf-8")) > _MAX_JSON_BYTES:
        raise ValueError(f"{owner} exceeds maximum UTF-8 bytes {_MAX_JSON_BYTES}")
    return payload


@overload
def _immutable_json(value: dict[str, Any]) -> Mapping[str, Any]: ...


@overload
def _immutable_json(value: Any) -> Any: ...


def _immutable_json(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType(
            {key: _immutable_json(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_immutable_json(item) for item in value)
    return value


def _canonical_evaluation_time(value: object) -> tuple[datetime, str]:
    if not isinstance(value, datetime):
        raise TypeError("evaluated_at must be a datetime")
    if value.tzinfo is None:
        raise ValueError("evaluated_at must include a timezone")
    normalized = value.astimezone(timezone.utc)
    timespec = "microseconds" if normalized.microsecond else "seconds"
    return normalized, normalized.isoformat(timespec=timespec).replace("+00:00", "Z")


@dataclass(frozen=True, slots=True)
class TargetSnapshot:
    """Frozen role view over an identity-validated capability snapshot."""

    _value: _targets.TargetCapabilitySnapshot = field(repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self._value, _targets.TargetCapabilitySnapshot):
            raise TypeError("TargetSnapshot views require a loaded Core snapshot")

    @property
    def version(self) -> str:
        return self._value.schema_version

    @property
    def identity(self) -> str:
        return self._value.snapshot_id

    @property
    def target_id(self) -> str:
        return self._value.target_identity.target_id

    @property
    def target_class(self) -> str:
        return self._value.target_identity.target_class

    @property
    def provider(self) -> str:
        return self._value.target_identity.provider

    @property
    def captured_at(self) -> str:
        return self._value.captured_at

    @property
    def valid_until(self) -> str:
        return self._value.valid_until

    @property
    def capability_names(self) -> tuple[str, ...]:
        return tuple(fact.name for fact in self._value.facts)

    def fact(self, name: str) -> Mapping[str, Any] | None:
        if name not in _targets.CAPABILITY_NAMES:
            raise ValueError(f"unknown v1 capability name {name!r}")
        for fact in self._value.facts:
            if fact.name == name:
                return _immutable_json(fact.to_dict())
        return None

    @property
    def blockers(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(_immutable_json(item.to_dict()) for item in self._value.blockers)

    def to_dict(self) -> dict[str, Any]:
        return self._value.to_dict()

    def to_json(self) -> str:
        return dump_target_snapshot(self)


@dataclass(frozen=True, slots=True)
class RequirementSet:
    """Frozen role view over an identity-validated requirement set."""

    _value: _targets.RequirementSet = field(repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self._value, _targets.RequirementSet):
            raise TypeError(
                "RequirementSet views require a loaded Core requirement set"
            )

    @property
    def version(self) -> str:
        return self._value.schema_version

    @property
    def identity(self) -> str:
        return self._value.requirement_set_id

    @property
    def requirement_count(self) -> int:
        return len(self._value.requirements)

    def to_dict(self) -> dict[str, Any]:
        return self._value.to_dict()

    def to_json(self) -> str:
        return dump_requirement_set(self)


@dataclass(frozen=True, slots=True)
class CapabilityMatch:
    """Frozen result of one explicit-time target capability evaluation."""

    _value: _targets.CapabilityMatchResult = field(repr=False)
    requirement_identity: str
    snapshot_identity: str
    evaluated_at: str

    def __post_init__(self) -> None:
        if not isinstance(self._value, _targets.CapabilityMatchResult):
            raise TypeError("CapabilityMatch views require a Core match result")
        for name in ("requirement_identity", "snapshot_identity"):
            if not _SHA256.fullmatch(getattr(self, name)):
                raise ValueError(f"{name} must be a canonical SHA-256 identity")
        try:
            parsed = datetime.fromisoformat(self.evaluated_at.replace("Z", "+00:00"))
        except (AttributeError, ValueError) as error:
            raise ValueError(
                "evaluated_at must be a canonical UTC timestamp"
            ) from error
        if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(
            parsed
        ):
            raise ValueError("evaluated_at must be a canonical UTC timestamp")
        timespec = "microseconds" if parsed.microsecond else "seconds"
        canonical = parsed.isoformat(timespec=timespec).replace("+00:00", "Z")
        if self.evaluated_at != canonical:
            raise ValueError("evaluated_at must be a canonical UTC timestamp")

    @property
    def executable(self) -> bool:
        return self._value.executable

    @property
    def blockers(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(_immutable_json(item.to_dict()) for item in self._value.blockers)

    @property
    def satisfied_preferences(self) -> int:
        return self._value.satisfied_preferences

    @property
    def total_preferences(self) -> int:
        return self._value.total_preferences


def load_target_snapshot(payload: str) -> TargetSnapshot:
    """Load bounded snapshot JSON into a version-neutral frozen role view."""
    view = TargetSnapshot(
        _targets.TargetCapabilitySnapshot.from_dict(
            _decode_json_object(payload, "target capability snapshot")
        )
    )
    dump_target_snapshot(view)
    return view


def dump_target_snapshot(snapshot: TargetSnapshot) -> str:
    """Return the snapshot's exact canonical Core JSON."""
    if type(snapshot) is not TargetSnapshot:
        raise TypeError("snapshot must be a TargetSnapshot view")
    return _bounded_output(snapshot._value.to_json(), "target capability snapshot")


def load_requirement_set(payload: str) -> RequirementSet:
    """Load bounded requirement JSON into a frozen role view."""
    view = RequirementSet(
        _targets.RequirementSet.from_dict(
            _decode_json_object(payload, "requirement set")
        )
    )
    dump_requirement_set(view)
    return view


def dump_requirement_set(requirements: RequirementSet) -> str:
    """Return the requirement set's exact canonical Core JSON."""
    if type(requirements) is not RequirementSet:
        raise TypeError("requirements must be a RequirementSet view")
    return _bounded_output(requirements._value.to_json(), "requirement set")


def match_capabilities(
    requirements: RequirementSet,
    snapshot: TargetSnapshot,
    *,
    evaluated_at: datetime,
) -> CapabilityMatch:
    """Match two public views at one explicit timezone-aware instant."""
    if type(requirements) is not RequirementSet:
        raise TypeError("requirements must be a RequirementSet view")
    if type(snapshot) is not TargetSnapshot:
        raise TypeError("snapshot must be a TargetSnapshot view")
    normalized, canonical = _canonical_evaluation_time(evaluated_at)
    return CapabilityMatch(
        _targets.match_target_capabilities(
            requirements._value, snapshot._value, evaluated_at=normalized
        ),
        requirement_identity=requirements.identity,
        snapshot_identity=snapshot.identity,
        evaluated_at=canonical,
    )


__all__ = (
    "CapabilityMatch",
    "RequirementSet",
    "TargetSnapshot",
    "dump_requirement_set",
    "dump_target_snapshot",
    "load_requirement_set",
    "load_target_snapshot",
    "match_capabilities",
)

"""Synthetic remote-style Target Capabilities v1 producer.

This adapter is deliberately a local, deterministic fixture producer.  It is
useful for proving that Runtime can replace the CPU Platform producer without
changing its consumer, but it does not submit work, contact a service, read
credentials, or make a claim about a real QPU or hardware device.

The fixture supplies JSON-safe facts explicitly.  Explicit observed values are
the only values promoted to ``verified/observed``.  Every omitted v1 fact is
still represented as ``null/unknown/not_exposed`` with a typed blocker so that
the Core matcher fails closed when that fact is required.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from math import isfinite
from types import MappingProxyType
from typing import Any, Mapping

from flagquantum.core.target_capabilities import (
    AUTHORITATIVE_STATIC_DECLARATION_ALLOWED,
    CAPABILITY_NAMES,
    CapabilityBlocker,
    CapabilityFact,
    CapabilityScope,
    EvidenceLevel,
    EvidenceReference,
    FactExposure,
    FactSource,
    SupportStatus,
    TargetCapabilitySnapshot,
    TargetIdentity,
)

SYNTHETIC_REMOTE_ADAPTER_VERSION = "1.0"
SYNTHETIC_REMOTE_PROVIDER = "flagquantum.synthetic_remote"
SYNTHETIC_REMOTE_TARGET_CLASS = "synthetic_remote_service"
_OBSERVED_SOURCE_KIND = "synthetic_remote_fixture_observation"
_DECLARED_SOURCE_KIND = "synthetic_remote_fixture_declaration"
_NUMERIC_FACT_NAMES = {
    "device.count",
    "memory.available_bytes",
    "qubits.logical_capacity",
    "qubits.physical_capacity",
    "limits.maximum_shots",
    "limits.maximum_program_operations",
    "ancillas.maximum_compiler",
}
_COLLECTION_FACT_NAMES = {
    "gates.native",
    "measurements.results",
    "artifacts.profiles",
}


def _freeze_fixture_json(value: Any, *, path: str) -> Any:
    """Deep-copy JSON-safe values into immutable containers at fixture build."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not isfinite(value):
            raise ValueError(f"{path} must contain only finite numbers")
        return value
    if isinstance(value, Mapping):
        frozen: dict[str, Any] = {}
        for key in value:
            if not isinstance(key, str):
                raise TypeError(f"{path} mapping keys must be strings")
            frozen[key] = _freeze_fixture_json(value[key], path=f"{path}.{key}")
        return MappingProxyType({key: frozen[key] for key in sorted(frozen)})
    if isinstance(value, (list, tuple)):
        return tuple(
            _freeze_fixture_json(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        )
    raise TypeError(
        f"{path} must contain only JSON-safe values, got {type(value).__name__}"
    )


def _validate_fixture_fact_value(name: str, value: Any) -> None:
    if value is None:
        return
    if name in _NUMERIC_FACT_NAMES:
        if type(value) is not int or value < 0:
            raise ValueError(f"synthetic fact {name} must be a non-negative integer")
    elif name in _COLLECTION_FACT_NAMES:
        if not isinstance(value, tuple):
            raise ValueError(f"synthetic fact {name} must be a JSON array")
    elif not isinstance(value, str):
        raise ValueError(f"synthetic fact {name} must be a string")


def _freeze_fact_mapping(
    values: Mapping[str, Any], *, field_name: str
) -> Mapping[str, Any]:
    if not isinstance(values, Mapping):
        raise TypeError(f"synthetic fixture {field_name} must be a mapping")
    frozen: dict[str, Any] = {}
    for name, value in values.items():
        if not isinstance(name, str) or not name:
            raise ValueError(f"synthetic fixture {field_name} keys must be strings")
        frozen_value = _freeze_fixture_json(value, path=f"{field_name}[{name!r}]")
        frozen[name] = frozen_value
    return MappingProxyType({name: frozen[name] for name in sorted(frozen)})


@dataclass(frozen=True)
class SyntheticRemoteCapabilityFixture:
    """Explicit, anonymous input for the synthetic producer.

    ``device_kind`` is required because Runtime must verify an observed
    device identity before treating a candidate as non-CPU.  Other facts are
    optional and are never inferred from the target identity, device IDs, or
    the presence of an evidence reference.
    """

    target_id: str
    target_revision: str
    environment_id: str
    device_kind: str
    device_ids: tuple[str, ...]
    evidence_refs: tuple[EvidenceReference, ...]
    observed_facts: Mapping[str, Any] = field(default_factory=dict)
    declared_facts: Mapping[str, Any] = field(default_factory=dict)
    provider_version: str = SYNTHETIC_REMOTE_ADAPTER_VERSION
    target_class: str = SYNTHETIC_REMOTE_TARGET_CLASS
    observed_source_ref: str = "synthetic-remote-observation"
    declared_source_ref: str = "synthetic-remote-declaration"

    def __post_init__(self) -> None:
        for name in (
            "target_id",
            "target_revision",
            "environment_id",
            "device_kind",
            "provider_version",
            "target_class",
            "observed_source_ref",
            "declared_source_ref",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"synthetic fixture {name} must be non-empty")
        if isinstance(self.device_ids, (str, bytes)) or not isinstance(
            self.device_ids, (tuple, list)
        ):
            raise TypeError("synthetic fixture device_ids must be a tuple or list")
        device_ids = tuple(self.device_ids)
        if not device_ids or any(
            not isinstance(item, str) or not item for item in device_ids
        ):
            raise ValueError("synthetic fixture device_ids must be non-empty strings")
        if len(set(device_ids)) != len(device_ids):
            raise ValueError("synthetic fixture device_ids must be unique")
        object.__setattr__(self, "device_ids", device_ids)
        if not isinstance(self.evidence_refs, (tuple, list)):
            raise TypeError("synthetic fixture evidence_refs must be a tuple or list")
        evidence_refs = tuple(self.evidence_refs)
        if not evidence_refs or any(
            not isinstance(item, EvidenceReference) for item in evidence_refs
        ):
            raise ValueError(
                "synthetic fixture evidence_refs must contain EvidenceReference values"
            )
        if len({item.evidence_id for item in evidence_refs}) != len(evidence_refs):
            raise ValueError("synthetic fixture evidence_refs must be unique")
        object.__setattr__(self, "evidence_refs", evidence_refs)
        for field_name in ("observed_facts", "declared_facts"):
            frozen_values = _freeze_fact_mapping(
                getattr(self, field_name), field_name=field_name
            )
            for name, value in frozen_values.items():
                _validate_fixture_fact_value(name, value)
            object.__setattr__(self, field_name, frozen_values)
        _validate_fact_names(self)


def _isoformat(value: datetime) -> str:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise ValueError("captured_at must be a timezone-aware datetime")
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _missing_blocker(name: str) -> CapabilityBlocker:
    return CapabilityBlocker(
        code=f"synthetic_{name.replace('.', '_')}_not_exposed",
        message=(
            "synthetic fixture did not explicitly provide this capability fact; "
            "the producer will not infer it"
        ),
        capability_name=name,
    )


def _validate_fact_names(fixture: SyntheticRemoteCapabilityFixture) -> None:
    observed_names = set(fixture.observed_facts)
    declared_names = set(fixture.declared_facts)
    unknown = (observed_names | declared_names) - CAPABILITY_NAMES
    if unknown:
        raise ValueError(
            "synthetic fixture contains unknown v1 capability names: "
            + ", ".join(sorted(unknown))
        )
    if "device.kind" in observed_names or "device.kind" in declared_names:
        raise ValueError("device.kind must be supplied by fixture.device_kind")
    if "target.class" in observed_names or "target.class" in declared_names:
        raise ValueError("target.class must be supplied by fixture.target_class")
    overlap = observed_names & declared_names
    if overlap:
        raise ValueError(
            "synthetic fixture cannot expose a fact as both observed and declared: "
            + ", ".join(sorted(overlap))
        )
    invalid_declared = declared_names - AUTHORITATIVE_STATIC_DECLARATION_ALLOWED
    if invalid_declared:
        raise ValueError(
            "synthetic fixture declarations are only allowed for static v1 facts: "
            + ", ".join(sorted(invalid_declared))
        )


def synthetic_remote_target_capability_snapshot(
    *,
    fixture: SyntheticRemoteCapabilityFixture,
    captured_at: datetime,
    ttl: timedelta,
) -> TargetCapabilitySnapshot:
    """Produce one Core v1 snapshot from explicit synthetic fixture facts.

    The adapter has no network or hardware path.  It requires evidence IDs for
    both the observed and static declaration sources, and requires observable
    evidence for the explicitly observed ``device.kind`` identity.  A missing
    fact remains nullable and non-verified instead of being guessed.
    """

    if not isinstance(fixture, SyntheticRemoteCapabilityFixture):
        raise TypeError("fixture must be a SyntheticRemoteCapabilityFixture")
    if not isinstance(ttl, timedelta) or ttl <= timedelta(0):
        raise ValueError("synthetic capability snapshot ttl must be positive")
    _validate_fact_names(fixture)
    evidence_by_id = {item.evidence_id: item for item in fixture.evidence_refs}
    for source_ref in (fixture.observed_source_ref, fixture.declared_source_ref):
        if source_ref not in evidence_by_id:
            raise ValueError(
                f"synthetic fixture source_ref {source_ref!r} has no evidence reference"
            )
    if evidence_by_id[fixture.observed_source_ref].level not in {
        EvidenceLevel.OBSERVABLE,
        EvidenceLevel.CERTIFICATION,
    }:
        raise ValueError(
            "observed synthetic facts require observable or certification evidence"
        )

    observed_values = {
        name: value
        for name, value in fixture.observed_facts.items()
        if value is not None
    }
    declared_values = {
        name: value
        for name, value in fixture.declared_facts.items()
        if value is not None
    }
    observed_values["device.kind"] = fixture.device_kind
    declared_values["target.class"] = fixture.target_class
    scope = CapabilityScope(device_ids=fixture.device_ids)
    observed_source = FactSource(
        kind=_OBSERVED_SOURCE_KIND,
        ref=fixture.observed_source_ref,
    )
    declared_source = FactSource(
        kind=_DECLARED_SOURCE_KIND,
        ref=fixture.declared_source_ref,
    )

    facts: list[CapabilityFact] = []
    for name in sorted(CAPABILITY_NAMES):
        if name in observed_values:
            facts.append(
                CapabilityFact(
                    name=name,
                    value=observed_values[name],
                    support_status=SupportStatus.VERIFIED,
                    fact_exposure=FactExposure.OBSERVED,
                    source=observed_source,
                )
            )
        elif name in declared_values:
            facts.append(
                CapabilityFact(
                    name=name,
                    value=declared_values[name],
                    support_status=SupportStatus.VERIFIED,
                    fact_exposure=FactExposure.DECLARED,
                    source=declared_source,
                )
            )
        else:
            facts.append(
                CapabilityFact(
                    name=name,
                    value=None,
                    support_status=SupportStatus.UNKNOWN,
                    fact_exposure=FactExposure.NOT_EXPOSED,
                    source=observed_source,
                    blockers=(_missing_blocker(name),),
                )
            )

    return TargetCapabilitySnapshot(
        target_identity=TargetIdentity(
            target_id=fixture.target_id,
            target_class=fixture.target_class,
            provider=SYNTHETIC_REMOTE_PROVIDER,
            provider_version=fixture.provider_version,
            target_revision=fixture.target_revision,
            environment_id=fixture.environment_id,
        ),
        scope=scope,
        captured_at=_isoformat(captured_at),
        valid_until=_isoformat(captured_at + ttl),
        facts=tuple(facts),
        evidence_refs=fixture.evidence_refs,
    )


__all__ = [
    "SYNTHETIC_REMOTE_ADAPTER_VERSION",
    "SYNTHETIC_REMOTE_PROVIDER",
    "SYNTHETIC_REMOTE_TARGET_CLASS",
    "SyntheticRemoteCapabilityFixture",
    "synthetic_remote_target_capability_snapshot",
]

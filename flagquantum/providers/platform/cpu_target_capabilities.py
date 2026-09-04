"""Narrow CPU Platform Provider-to-Core Target Capabilities v1 adapter.

The adapter consumes an injected, provider-owned CPU probe.  It does not call
the resolver, choose a backend, authorize fallback, or infer precision from
Python/PyTorch defaults.  The Core value objects are intentionally imported
directly here and are not re-exported from the public package.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Iterable, Protocol

from flagquantum.core.target_capabilities import (
    CapabilityBlocker,
    CapabilityFact,
    CapabilityScope,
    EvidenceReference,
    FactExposure,
    FactSource,
    SupportStatus,
    TargetCapabilitySnapshot,
    TargetIdentity,
)

CPU_PLATFORM_ADAPTER_VERSION = "1.0"
CPU_PLATFORM_PROVIDER = "pytorch_cpu"
CPU_TARGET_CLASS = "local_runtime"
_PRECISION_FACTS = {
    "precision.native_dtype",
    "precision.effective_dtype",
    "precision.storage_dtype",
    "precision.parameter_dtype",
    "precision.accumulator_dtype",
    "precision.software_mechanism",
}


@dataclass(frozen=True)
class CPUPrecisionObservation:
    """One explicitly probed CPU precision fact.

    A value is required for an explicit negative or declaration.  Missing
    values are represented by a separate nullable Core fact with a blocker;
    this avoids fabricating a numeric or string sentinel.
    """

    name: str
    value: str
    support_status: SupportStatus = SupportStatus.VERIFIED
    fact_exposure: FactExposure = FactExposure.OBSERVED
    blockers: tuple[CapabilityBlocker, ...] = ()

    def __post_init__(self) -> None:
        if self.name not in _PRECISION_FACTS:
            raise ValueError(f"unsupported CPU precision fact {self.name!r}")
        if not isinstance(self.value, str) or not self.value:
            raise ValueError("CPU precision observation value must be non-empty")
        if not isinstance(self.support_status, SupportStatus):
            object.__setattr__(
                self, "support_status", SupportStatus(self.support_status)
            )
        if not isinstance(self.fact_exposure, FactExposure):
            object.__setattr__(self, "fact_exposure", FactExposure(self.fact_exposure))
        blockers = tuple(self.blockers)
        if any(not isinstance(item, CapabilityBlocker) for item in blockers):
            raise TypeError("CPU precision blockers must be CapabilityBlocker values")
        object.__setattr__(self, "blockers", blockers)
        if self.support_status is SupportStatus.VERIFIED and blockers:
            raise ValueError("verified CPU precision facts cannot carry blockers")
        if self.support_status is SupportStatus.VERIFIED and (
            self.fact_exposure is not FactExposure.OBSERVED
        ):
            raise ValueError("verified CPU precision facts must be observed")
        if self.support_status is not SupportStatus.VERIFIED and not blockers:
            raise ValueError("non-verified CPU precision facts require a blocker")


@dataclass(frozen=True)
class CPUCapabilityObservation:
    """JSON-safe output of one injected CPU probe invocation."""

    available: bool
    device_count: int | None
    device_ids: tuple[str, ...] | None = None
    memory_available_bytes: int | None = None
    precision: tuple[CPUPrecisionObservation, ...] = ()

    def __post_init__(self) -> None:
        if type(self.available) is not bool:
            raise TypeError("CPU probe available must be a boolean")
        if self.device_count is not None and (
            type(self.device_count) is not int or self.device_count < 0
        ):
            raise ValueError("CPU probe device_count must be a non-negative integer")
        if self.memory_available_bytes is not None and (
            type(self.memory_available_bytes) is not int
            or self.memory_available_bytes < 0
        ):
            raise ValueError(
                "CPU probe memory_available_bytes must be a non-negative integer"
            )
        if self.device_ids is not None:
            if isinstance(self.device_ids, (str, bytes)) or not isinstance(
                self.device_ids, (list, tuple)
            ):
                raise ValueError(
                    "CPU probe device_ids must be an array or tuple of strings"
                )
            ids = tuple(self.device_ids)
            if any(not isinstance(item, str) or not item for item in ids):
                raise ValueError("CPU probe device_ids must contain non-empty strings")
            if len(set(ids)) != len(ids):
                raise ValueError("CPU probe device_ids must not contain duplicates")
            object.__setattr__(self, "device_ids", ids)
        precision = tuple(self.precision)
        if any(not isinstance(item, CPUPrecisionObservation) for item in precision):
            raise TypeError(
                "CPU probe precision must contain CPUPrecisionObservation values"
            )
        if len({item.name for item in precision}) != len(precision):
            raise ValueError("CPU probe precision facts must have unique names")
        object.__setattr__(self, "precision", precision)
        if self.available:
            if self.device_count is None or self.device_count < 1:
                raise ValueError(
                    "available CPU probe must observe at least one CPU device"
                )
            if self.device_ids is None or len(self.device_ids) != self.device_count:
                raise ValueError(
                    "available CPU probe must provide one id per observed device"
                )
        elif self.device_count not in (None, 0):
            raise ValueError(
                "unavailable CPU probe cannot report a positive device count"
            )


class CPUCapabilityProbe(Protocol):
    """Provider-owned probe metadata and one observation operation."""

    target_id: str
    provider_version: str
    target_revision: str
    environment_id: str
    source_ref: str
    target_class_source_ref: str

    def observe(self) -> CPUCapabilityObservation: ...


def _isoformat(value: datetime) -> str:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise ValueError("captured_at must be a timezone-aware datetime")
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _blocker(
    code: str, message: str, capability_name: str | None = None
) -> CapabilityBlocker:
    return CapabilityBlocker(
        code=code, message=message, capability_name=capability_name
    )


def _missing_fact(name: str, *, reason: str) -> CapabilityBlocker:
    return _blocker(
        f"cpu_{name.replace('.', '_')}_not_observed",
        reason,
        name,
    )


def cpu_platform_to_target_capability_snapshot(
    *,
    probe: CPUCapabilityProbe,
    captured_at: datetime,
    ttl: timedelta,
    evidence_refs: Iterable[EvidenceReference] = (),
) -> TargetCapabilitySnapshot:
    """Build a CPU target snapshot from one injected observation.

    ``probe`` is the only source of device count, memory, and precision facts.
    ``captured_at`` and ``ttl`` are caller-owned to make freshness deterministic
    in tests and runners.  The adapter emits no requirements and never selects
    or substitutes a target.
    """

    for name in (
        "target_id",
        "provider_version",
        "target_revision",
        "environment_id",
        "source_ref",
    ):
        value = getattr(probe, name, None)
        if not isinstance(value, str) or not value:
            raise ValueError(f"CPU probe {name} must be a non-empty string")
    target_class_source_ref = getattr(
        probe, "target_class_source_ref", probe.source_ref
    )
    if not isinstance(target_class_source_ref, str) or not target_class_source_ref:
        raise ValueError("CPU probe target_class_source_ref must be a non-empty string")
    if not isinstance(ttl, timedelta) or ttl <= timedelta(0):
        raise ValueError("CPU capability snapshot ttl must be positive")
    captured_timestamp = _isoformat(captured_at)
    valid_timestamp = _isoformat(captured_at + ttl)
    observation = probe.observe()
    if not isinstance(observation, CPUCapabilityObservation):
        raise TypeError("CPU probe observe() must return CPUCapabilityObservation")
    refs = tuple(evidence_refs)
    if not refs:
        raise ValueError(
            "CPU capability snapshot requires at least one evidence reference"
        )
    if any(not isinstance(item, EvidenceReference) for item in refs):
        raise TypeError("evidence_refs must contain EvidenceReference values")
    evidence_ids = {item.evidence_id for item in refs}
    if probe.source_ref not in evidence_ids:
        raise ValueError(
            "CPU probe source_ref must resolve to an evidence reference id"
        )
    if target_class_source_ref not in evidence_ids:
        raise ValueError(
            "CPU probe target_class_source_ref must resolve to an evidence reference id"
        )
    evidence_by_id = {item.evidence_id: item for item in refs}
    source = FactSource(kind="cpu_platform_probe", ref=probe.source_ref)
    target_class_source = FactSource(
        kind="cpu_platform_static_declaration", ref=target_class_source_ref
    )
    scope = CapabilityScope(device_ids=observation.device_ids)
    facts: list[CapabilityFact] = []
    snapshot_blockers: list[CapabilityBlocker] = []

    facts.append(
        CapabilityFact(
            name="target.class",
            value=CPU_TARGET_CLASS,
            support_status=SupportStatus.VERIFIED,
            fact_exposure=FactExposure.DECLARED,
            source=target_class_source,
        )
    )

    if observation.available:
        facts.extend(
            (
                CapabilityFact(
                    name="device.kind",
                    value="cpu",
                    support_status=SupportStatus.VERIFIED,
                    fact_exposure=FactExposure.OBSERVED,
                    source=source,
                ),
                CapabilityFact(
                    name="device.count",
                    value=observation.device_count,
                    support_status=SupportStatus.VERIFIED,
                    fact_exposure=FactExposure.OBSERVED,
                    source=source,
                ),
            )
        )
    else:
        facts.append(
            CapabilityFact(
                name="device.kind",
                value="cpu",
                support_status=SupportStatus.UNKNOWN,
                fact_exposure=FactExposure.NOT_EXPOSED,
                source=source,
                blockers=(
                    _blocker(
                        "cpu_device_unavailable",
                        "CPU probe did not expose an available CPU device",
                        "device.kind",
                    ),
                ),
            )
        )
        if observation.device_count == 0:
            unavailable = _blocker(
                "cpu_unavailable",
                "CPU probe observed no available CPU device",
            )
            facts.append(
                CapabilityFact(
                    name="device.count",
                    value=0,
                    support_status=SupportStatus.UNSUPPORTED,
                    fact_exposure=FactExposure.OBSERVED,
                    source=source,
                    blockers=(unavailable,),
                )
            )
            snapshot_blockers.append(unavailable)
        else:
            facts.append(
                CapabilityFact(
                    name="device.count",
                    value=None,
                    support_status=SupportStatus.UNKNOWN,
                    fact_exposure=FactExposure.NOT_EXPOSED,
                    source=source,
                    blockers=(
                        _blocker(
                            "cpu_device_count_not_observed",
                            "CPU probe did not expose an unavailable-device count",
                            "device.count",
                        ),
                    ),
                )
            )
            snapshot_blockers.append(
                _blocker(
                    "cpu_unavailable",
                    "CPU probe reported no available device count",
                )
            )

    if observation.memory_available_bytes is None:
        missing = _missing_fact(
            "memory.available_bytes",
            reason="CPU probe did not expose available memory bytes",
        )
        facts.append(
            CapabilityFact(
                name="memory.available_bytes",
                value=None,
                support_status=SupportStatus.UNKNOWN,
                fact_exposure=FactExposure.NOT_EXPOSED,
                source=source,
                blockers=(missing,),
            )
        )
    else:
        facts.append(
            CapabilityFact(
                name="memory.available_bytes",
                value=observation.memory_available_bytes,
                support_status=SupportStatus.VERIFIED,
                fact_exposure=FactExposure.OBSERVED,
                source=source,
            )
        )

    observed_precision = {item.name: item for item in observation.precision}
    for name in sorted(_PRECISION_FACTS):
        item = observed_precision.get(name)
        if item is None:
            missing = _missing_fact(
                name,
                reason=f"CPU probe did not expose {name}",
            )
            facts.append(
                CapabilityFact(
                    name=name,
                    value=None,
                    support_status=SupportStatus.UNKNOWN,
                    fact_exposure=FactExposure.NOT_EXPOSED,
                    source=source,
                    blockers=(missing,),
                )
            )
            continue
        facts.append(
            CapabilityFact(
                name=item.name,
                value=item.value,
                support_status=item.support_status,
                fact_exposure=item.fact_exposure,
                source=source,
                blockers=item.blockers,
            )
        )

    if any(
        item.source.ref == probe.source_ref
        and item.fact_exposure is FactExposure.OBSERVED
        for item in facts
    ):
        probe_evidence = evidence_by_id[probe.source_ref]
        if probe_evidence.level.value not in {"observable", "certification"}:
            raise ValueError(
                "observed CPU facts require observable or certification evidence"
            )

    return TargetCapabilitySnapshot(
        target_identity=TargetIdentity(
            target_id=probe.target_id,
            target_class=CPU_TARGET_CLASS,
            provider=CPU_PLATFORM_PROVIDER,
            provider_version=probe.provider_version,
            target_revision=probe.target_revision,
            environment_id=probe.environment_id,
        ),
        scope=scope,
        captured_at=captured_timestamp,
        valid_until=valid_timestamp,
        facts=tuple(facts),
        evidence_refs=refs,
        blockers=tuple(snapshot_blockers),
    )


__all__ = [
    "CPUCapabilityObservation",
    "CPUCapabilityProbe",
    "CPUPrecisionObservation",
    "CPU_PLATFORM_ADAPTER_VERSION",
    "CPU_TARGET_CLASS",
    "cpu_platform_to_target_capability_snapshot",
]

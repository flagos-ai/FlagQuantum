"""Project an observed CUDA statevector path into Core capabilities."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from flagquantum.core.target_capabilities import (
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

CUDA_PLATFORM_PROVIDER = "pytorch_cuda"
CUDA_STATEVECTOR_WORKLOAD = "statevector_local_p0"


def _timestamp(value: datetime) -> str:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise ValueError("captured_at must be a timezone-aware datetime")
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def cuda_statevector_capability_snapshot(
    *,
    target_id: str,
    provider_version: str,
    target_revision: str,
    environment_id: str,
    device_id: str,
    memory_available_bytes: int,
    evidence_id: str,
    evidence_sha256: str,
    captured_at: datetime,
    ttl: timedelta,
) -> TargetCapabilitySnapshot:
    """Build the narrow snapshot justified by a passed complex128 CUDA probe.

    The caller must first execute ``statevector_local_p0`` with exactly one
    visible CUDA device.  This projector deliberately makes no claim about
    other workloads, multiple devices, communication, or absence of hidden
    host work.
    """

    strings = {
        "target_id": target_id,
        "provider_version": provider_version,
        "target_revision": target_revision,
        "environment_id": environment_id,
        "device_id": device_id,
        "evidence_id": evidence_id,
    }
    for name, value in strings.items():
        if not isinstance(value, str) or not value:
            raise ValueError(f"{name} must be a non-empty string")
    if type(memory_available_bytes) is not int or memory_available_bytes < 0:
        raise ValueError("memory_available_bytes must be a non-negative integer")
    if not isinstance(ttl, timedelta) or ttl <= timedelta(0):
        raise ValueError("CUDA capability snapshot ttl must be positive")

    scope = CapabilityScope(
        device_ids=(device_id,),
        dtype="complex128",
        kernel="statevector",
        workload_id=CUDA_STATEVECTOR_WORKLOAD,
        world_size=1,
        node_count=1,
    )
    evidence = EvidenceReference(
        evidence_id=evidence_id,
        sha256=evidence_sha256,
        level=EvidenceLevel.OBSERVABLE,
        scope=scope,
    )
    source = FactSource(kind="cuda_statevector_probe", ref=evidence_id)

    def observed(name: str, value: str | int) -> CapabilityFact:
        return CapabilityFact(
            name=name,
            value=value,
            support_status=SupportStatus.VERIFIED,
            fact_exposure=FactExposure.OBSERVED,
            source=source,
        )

    facts = (
        CapabilityFact(
            name="target.class",
            value="local_runtime",
            support_status=SupportStatus.VERIFIED,
            fact_exposure=FactExposure.DECLARED,
            source=source,
        ),
        observed("device.kind", "cuda"),
        observed("device.count", 1),
        observed("memory.available_bytes", memory_available_bytes),
        observed("precision.native_dtype", "float64"),
        observed("precision.effective_dtype", "complex128"),
        observed("precision.storage_dtype", "float64"),
        observed("precision.parameter_dtype", "float64"),
        observed("precision.accumulator_dtype", "float64"),
        observed("precision.software_mechanism", "none"),
    )
    return TargetCapabilitySnapshot(
        target_identity=TargetIdentity(
            target_id=target_id,
            target_class="local_runtime",
            provider=CUDA_PLATFORM_PROVIDER,
            provider_version=provider_version,
            target_revision=target_revision,
            environment_id=environment_id,
        ),
        scope=scope,
        captured_at=_timestamp(captured_at),
        valid_until=_timestamp(captured_at + ttl),
        facts=facts,
        evidence_refs=(evidence,),
    )


__all__ = (
    "CUDA_PLATFORM_PROVIDER",
    "CUDA_STATEVECTOR_WORKLOAD",
    "cuda_statevector_capability_snapshot",
)

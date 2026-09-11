"""Characterize the proposed Platform-to-Core fact projection.

The value objects and projection helper in this file are deliberately test-only.
ARCH-003 and ARCH-006 are still proposals, so the Platform team must not create a
private production contract before Core approves the shared vocabulary.  These
tests pin the facts that a later Core-owned adapter must preserve.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any, Mapping

import pytest
import torch

from flagquantum.compute import (
    MemorySnapshot,
    PlatformDevice,
    PlatformIdentity,
)

pytestmark = pytest.mark.unit


@dataclass(frozen=True)
class _Fact:
    value: Any
    support_status: str
    fact_exposure: str
    evidence_level: str
    blockers: tuple[str, ...] = ()
    applicability_reason: str | None = None


_DECLARED_FIELDS = (
    "storage_dtypes",
    "compute_dtypes",
    "accumulation_dtypes",
    "native_fp64",
    "native_complex128",
    "software_precision_modes",
)

_NOT_EXPOSED_FIELDS = (
    "stream_semantics",
    "event_semantics",
    "topology",
    "p2p",
    "intra_node_collectives",
    "inter_node_communication",
    "kernel_residency",
    "cpu_fallback_used",
)


def _portable_declaration(value: Any) -> bool:
    if isinstance(value, (str, bool, int, float)):
        return True
    return isinstance(value, (list, tuple)) and all(
        isinstance(item, str) for item in value
    )


def _candidate_projection(runtime: Any, device: torch.device) -> dict[str, Any]:
    """Test fixture for the smallest proposed projection, not a public schema."""

    identity = runtime.identity()
    devices = runtime.discover()
    selected = next(
        (
            item
            for item in devices
            if item.device_type == device.type
            and (device.index is None or item.index == device.index)
        ),
        None,
    )

    def lifecycle_fact(value: bool) -> _Fact:
        return _Fact(
            value=value,
            support_status="verified" if value else "unsupported",
            fact_exposure="observed",
            evidence_level="basic",
        )

    facts: dict[str, _Fact] = {
        "environment_installed": lifecycle_fact(bool(runtime.installed())),
        "environment_activated": lifecycle_fact(bool(runtime.activated())),
        "environment_available": lifecycle_fact(bool(runtime.is_available())),
        "device_memory_total_bytes": _Fact(
            value=None,
            support_status="unknown",
            fact_exposure="not_exposed",
            evidence_level="basic",
            blockers=("platform_memory_total_not_exposed",),
        ),
    }
    if selected is not None:
        memory = runtime.memory_snapshot(device)
        total = memory.total_bytes
        if total is None:
            total = selected.memory_bytes
        if total is not None:
            facts["device_memory_total_bytes"] = _Fact(
                value=int(total),
                support_status="verified",
                fact_exposure="observed",
                evidence_level="basic",
            )

    declarations: Mapping[str, Any] = identity.metadata
    for name in _DECLARED_FIELDS:
        if name not in declarations:
            facts[name] = _Fact(
                value=None,
                support_status="unknown",
                fact_exposure="not_exposed",
                evidence_level="basic",
                blockers=(f"{name}_not_exposed",),
            )
        elif _portable_declaration(declarations[name]):
            value = declarations[name]
            facts[name] = _Fact(
                value=list(value) if isinstance(value, tuple) else value,
                support_status="unmeasured",
                fact_exposure="declared",
                evidence_level="basic",
                blockers=(f"{name}_declared_not_verified",),
            )
        else:
            facts[name] = _Fact(
                value=None,
                support_status="unknown",
                fact_exposure="unknown",
                evidence_level="basic",
                blockers=(f"{name}_non_serializable_declaration_discarded",),
            )

    for name in _NOT_EXPOSED_FIELDS:
        facts[name] = _Fact(
            value=None,
            support_status="unknown",
            fact_exposure="not_exposed",
            evidence_level="basic",
            blockers=(f"{name}_not_exposed_by_platform_runtime",),
        )

    return {
        "provider": identity.provider,
        "device_type": identity.device_type,
        "provider_version": identity.provider_version,
        "vendor": identity.vendor,
        "facts": {name: asdict(fact) for name, fact in facts.items()},
    }


_EVIDENCE_STRENGTH = {"basic": 0, "observable": 1, "certification": 2}


def _claim_allowed(
    facts: Mapping[str, Mapping[str, Any]],
    required: tuple[str, ...],
    *,
    minimum_evidence: str,
) -> bool:
    """Model the proposed fail-closed consumer rule."""

    return all(
        name in facts
        and facts[name]["support_status"] == "verified"
        and facts[name]["fact_exposure"] == "observed"
        and _EVIDENCE_STRENGTH[facts[name]["evidence_level"]]
        >= _EVIDENCE_STRENGTH[minimum_evidence]
        for name in required
    )


class _FakeRuntime:
    def __init__(
        self,
        *,
        device_type: str,
        available: bool,
        installed: bool = True,
        activated: bool = True,
        memory_bytes: int | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self.name = f"fake_{device_type}"
        self.device_type = device_type
        self._available = available
        self._installed = installed
        self._activated = activated
        self._memory_bytes = memory_bytes
        self._metadata = dict(metadata or {})

    def installed(self) -> bool:
        return self._installed

    def activated(self) -> bool:
        return self._activated

    def is_available(self) -> bool:
        return self._available

    def discover(self) -> tuple[PlatformDevice, ...]:
        if not self._available:
            return ()
        return (
            PlatformDevice(
                device_type=self.device_type,
                index=None if self.device_type == "cpu" else 0,
                name=f"Fake {self.device_type}",
                provider=self.name,
                memory_bytes=self._memory_bytes,
            ),
        )

    def memory_snapshot(self, device: torch.device) -> MemorySnapshot:
        return MemorySnapshot(total_bytes=self._memory_bytes)

    def identity(self) -> PlatformIdentity:
        return PlatformIdentity(
            provider=self.name,
            device_type=self.device_type,
            torch_version="test",
            provider_version="fake-sdk-1",
            vendor="nvidia" if self.device_type == "cuda" else None,
            metadata=self._metadata,
        )


def test_cpu_fake_keeps_unexposed_capabilities_unknown() -> None:
    snapshot = _candidate_projection(
        _FakeRuntime(device_type="cpu", available=True), torch.device("cpu")
    )
    facts = snapshot["facts"]

    assert facts["environment_available"] == {
        "value": True,
        "support_status": "verified",
        "fact_exposure": "observed",
        "evidence_level": "basic",
        "blockers": (),
        "applicability_reason": None,
    }
    for name in (
        "native_fp64",
        "software_precision_modes",
        "device_memory_total_bytes",
        "p2p",
        "inter_node_communication",
        "kernel_residency",
        "cpu_fallback_used",
    ):
        assert facts[name]["value"] is None
        assert facts[name]["support_status"] == "unknown"
        assert facts[name]["fact_exposure"] == "not_exposed"

    # CPU identity and Python/PyTorch API support do not certify every dtype,
    # kernel, communication path, or the absence of a fallback.
    json.dumps(snapshot, sort_keys=True)


def test_cuda_fake_separates_availability_declaration_and_verification() -> None:
    snapshot = _candidate_projection(
        _FakeRuntime(
            device_type="cuda",
            available=True,
            memory_bytes=80 * 1024**3,
            metadata={
                "storage_dtypes": ("float32", "float64"),
                "native_fp64": True,
                "native_complex128": True,
                "software_precision_modes": ("double_single_fp32",),
            },
        ),
        torch.device("cuda:0"),
    )
    facts = snapshot["facts"]

    assert facts["environment_available"]["support_status"] == "verified"
    assert facts["device_memory_total_bytes"]["value"] == 80 * 1024**3
    assert facts["device_memory_total_bytes"]["fact_exposure"] == "observed"
    for name in (
        "storage_dtypes",
        "native_fp64",
        "native_complex128",
        "software_precision_modes",
    ):
        assert facts[name]["support_status"] == "unmeasured"
        assert facts[name]["fact_exposure"] == "declared"
        assert facts[name]["evidence_level"] == "basic"
        assert facts[name]["blockers"] == (f"{name}_declared_not_verified",)

    # Device discovery and vendor declarations do not promote topology,
    # collectives, kernel residency, or no-CPU-fallback claims.
    for name in _NOT_EXPOSED_FIELDS:
        assert facts[name]["value"] is None
        assert facts[name]["support_status"] == "unknown"
        assert facts[name]["fact_exposure"] == "not_exposed"


def test_missing_or_unsafe_sdk_fields_are_not_fabricated_or_leaked() -> None:
    vendor_object = object()
    snapshot = _candidate_projection(
        _FakeRuntime(
            device_type="cuda",
            available=False,
            metadata={"native_fp64": vendor_object},
        ),
        torch.device("cuda:0"),
    )
    facts = snapshot["facts"]

    assert facts["environment_available"]["value"] is False
    assert facts["environment_available"]["support_status"] == "unsupported"
    assert facts["environment_available"]["fact_exposure"] == "observed"
    assert facts["native_fp64"] == {
        "value": None,
        "support_status": "unknown",
        "fact_exposure": "unknown",
        "evidence_level": "basic",
        "blockers": ("native_fp64_non_serializable_declaration_discarded",),
        "applicability_reason": None,
    }
    assert facts["compute_dtypes"]["value"] is None
    assert facts["compute_dtypes"]["fact_exposure"] == "not_exposed"
    assert facts["cpu_fallback_used"]["value"] is None
    assert "object at" not in json.dumps(snapshot, sort_keys=True)


def test_support_exposure_and_evidence_axes_remain_orthogonal() -> None:
    facts = {
        "negative_probe": asdict(_Fact(False, "unsupported", "observed", "observable")),
        "vendor_declaration": asdict(_Fact(True, "unmeasured", "declared", "basic")),
        "missing_sdk_field": asdict(
            _Fact(None, "unknown", "not_exposed", "basic", ("sdk_field_missing",))
        ),
        "untrusted_value": asdict(
            _Fact(None, "unknown", "unknown", "basic", ("source_unknown",))
        ),
        "single_device_collective": asdict(
            _Fact(
                None,
                "unknown",
                "not_applicable",
                "basic",
                applicability_reason="world_size_is_one",
            )
        ),
    }

    assert {fact["fact_exposure"] for fact in facts.values()} == {
        "observed",
        "declared",
        "not_exposed",
        "unknown",
        "not_applicable",
    }
    assert facts["negative_probe"]["support_status"] == "unsupported"
    assert facts["negative_probe"]["evidence_level"] == "observable"
    assert facts["single_device_collective"]["applicability_reason"]
    assert not _claim_allowed(
        facts,
        tuple(facts),
        minimum_evidence="basic",
    )


def test_cuda_declarations_cannot_satisfy_observable_claims() -> None:
    snapshot = _candidate_projection(
        _FakeRuntime(
            device_type="cuda",
            available=True,
            memory_bytes=80 * 1024**3,
            metadata={
                "native_fp64": True,
                "native_complex128": True,
                "software_precision_modes": ("double_single_fp32",),
            },
        ),
        torch.device("cuda:0"),
    )

    assert not _claim_allowed(
        snapshot["facts"],
        ("native_fp64", "native_complex128", "kernel_residency"),
        minimum_evidence="observable",
    )
    assert not _claim_allowed(
        snapshot["facts"],
        ("intra_node_collectives", "inter_node_communication", "cpu_fallback_used"),
        minimum_evidence="basic",
    )

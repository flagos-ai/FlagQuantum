"""CUDA statevector capability projection tests."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from flagquantum.core.target_capabilities import (
    CapabilityContractError,
    EvidenceLevel,
    FactExposure,
    SupportStatus,
    TargetCapabilitySnapshot,
)
from flagquantum.providers.platform.cuda_target_capabilities import (
    cuda_statevector_capability_snapshot,
)

pytestmark = pytest.mark.unit


def _snapshot(**changes: object) -> TargetCapabilitySnapshot:
    values = {
        "target_id": "cuda-gpu-test",
        "provider_version": "torch-test-cuda-test",
        "target_revision": "sm_80",
        "environment_id": "environment-test",
        "device_id": "cuda:0:gpu-test",
        "memory_available_bytes": 80 * 1024**3,
        "evidence_id": "cuda-statevector-test",
        "evidence_sha256": "a" * 64,
        "captured_at": datetime(2026, 9, 5, tzinfo=timezone.utc),
        "ttl": timedelta(minutes=30),
    }
    values.update(changes)
    return cuda_statevector_capability_snapshot(**values)  # type: ignore[arg-type]


def test_cuda_statevector_snapshot_has_one_narrow_observed_scope() -> None:
    snapshot = _snapshot()
    facts = {item.name: item for item in snapshot.facts}

    assert snapshot.target_identity.provider == "pytorch_cuda"
    assert snapshot.scope.dtype == "complex128"
    assert snapshot.scope.kernel == "statevector"
    assert snapshot.scope.workload_id == "statevector_local_p0"
    assert snapshot.scope.world_size == snapshot.scope.node_count == 1
    assert facts["device.kind"].value == "cuda"
    assert facts["device.count"].value == 1
    assert facts["precision.native_dtype"].value == "float64"
    assert facts["precision.effective_dtype"].value == "complex128"
    assert facts["precision.software_mechanism"].value == "none"
    assert facts["target.class"].fact_exposure is FactExposure.DECLARED
    assert all(fact.support_status is SupportStatus.VERIFIED for fact in snapshot.facts)
    assert snapshot.evidence_refs[0].level is EvidenceLevel.OBSERVABLE
    assert snapshot.evidence_refs[0].scope == snapshot.scope


def test_cuda_statevector_snapshot_round_trips() -> None:
    snapshot = _snapshot()
    assert TargetCapabilitySnapshot.from_json(snapshot.to_json()) == snapshot


def test_cuda_statevector_snapshot_rejects_invalid_evidence_digest() -> None:
    with pytest.raises(CapabilityContractError, match="64 lowercase hex"):
        _snapshot(evidence_sha256="not-a-digest")


def test_cuda_statevector_snapshot_rejects_non_positive_ttl() -> None:
    with pytest.raises(ValueError, match="ttl must be positive"):
        _snapshot(ttl=timedelta(0))

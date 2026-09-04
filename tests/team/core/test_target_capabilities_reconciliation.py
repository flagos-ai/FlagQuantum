from __future__ import annotations

from dataclasses import fields, replace

import pytest

import flagquantum as fq
from flagquantum._compiler.capability_comparison import compare_target_capabilities
from flagquantum._compiler.target_capabilities import (
    ArtifactFormat,
    ArtifactProfile,
    GateCapability,
    MeasurementResult,
    TargetCapabilities,
    TargetClass,
)
from flagquantum.core.contracts import CapabilityContract
from flagquantum.deployment.cloud import CloudBackendProfile
from flagquantum.extensions.sdk import CapabilityRequest, CapabilityResponse
from flagquantum.providers.platform.contracts import MemorySnapshot, PlatformDevice
from flagquantum.runtime.backend_registry import BackendCapabilities
from flagquantum.runtime.capabilities import CapabilityEvidence

pytestmark = pytest.mark.unit


def _target() -> TargetCapabilities:
    return TargetCapabilities(
        target_class=TargetClass.LOCAL_RUNTIME,
        logical_qubit_capacity=2,
        physical_qubit_capacity=2,
        native_gates=(GateCapability("h"), GateCapability("cx")),
        measurement_results=(MeasurementResult.STATE,),
        artifact_profiles=(ArtifactProfile(ArtifactFormat.RUNTIME_PLAN, "1.0"),),
    )


def _field_names(contract: type[object]) -> tuple[str, ...]:
    return tuple(item.name for item in fields(contract))


def test_target_capabilities_v1_is_a_closed_compiler_semantic_snapshot() -> None:
    target = _target()
    payload = target.to_dict()

    assert payload["schema_version"] == "target_capabilities_v1"
    assert set(payload) == {
        "schema_version",
        "target_class",
        "logical_qubit_capacity",
        "physical_qubit_capacity",
        "native_gates",
        "measurement_results",
        "artifact_profiles",
        "topology",
        "control_flow",
        "supports_mid_circuit_measurement",
        "supports_reset",
        "supports_timing",
        "supports_pulse",
        "supports_noise",
        "supports_parameter_binding",
        "maximum_shots",
        "maximum_program_operations",
        "ancilla_policy",
        "maximum_compiler_ancillas",
        "calibration_snapshot_hash",
        "calibration_valid_until",
        "display_label",
    }
    for absent in (
        "provider",
        "target_id",
        "captured_at",
        "source",
        "support_status",
        "fact_exposure",
        "evidence_level",
        "evidence_refs",
        "extensions",
    ):
        assert absent not in payload


@pytest.mark.parametrize("unknown_value", ["unknown", "unmeasured", "unsupported"])
def test_target_boolean_flags_cannot_encode_four_state_support(
    unknown_value: str,
) -> None:
    payload = _target().to_dict()
    payload["supports_noise"] = unknown_value

    with pytest.raises(ValueError, match="must be boolean"):
        TargetCapabilities.from_dict(payload)


def test_target_comparison_is_requirement_coverage_not_discovery_evidence() -> None:
    required = _target()
    available = replace(
        required,
        supports_noise=True,
        maximum_shots=1024,
        display_label="live target label",
    )

    comparison = compare_target_capabilities(required, available)

    assert comparison.compatible is True
    assert comparison.differences == ()
    assert available.semantic_fingerprint != required.semantic_fingerprint
    assert replace(available, display_label="another label").semantic_fingerprint == (
        available.semantic_fingerprint
    )


def test_capability_shaped_types_are_distinct_lossy_domain_projections() -> None:
    assert _field_names(CapabilityContract) == (
        "version",
        "backend",
        "devices",
        "dtypes",
        "modes",
        "supports_autograd",
        "supports_distributed",
    )
    assert _field_names(BackendCapabilities) == (
        "name",
        "tensor_backend",
        "devices",
        "dtypes",
        "supports_autograd",
        "supports_distributed",
        "supports_statevector",
        "supports_density_matrix",
        "supports_mps",
        "preferred_device",
        "accelerators",
    )
    assert _field_names(CloudBackendProfile) == (
        "provider",
        "name",
        "n_wires",
        "basis_gates",
        "coupling_map",
        "supports_openqasm",
        "supports_qcis",
        "supports_dynamic_circuits",
        "max_classical_bits",
        "is_simulator",
        "metadata",
        "dynamic_dialect",
    )
    assert not hasattr(BackendCapabilities, "from_dict")
    assert not hasattr(CloudBackendProfile, "from_dict")


def test_extension_capability_messages_are_negotiation_not_target_snapshots() -> None:
    assert _field_names(CapabilityRequest) == (
        "required",
        "dtype",
        "device_type",
        "require_gradients",
    )
    assert _field_names(CapabilityResponse) == (
        "accepted",
        "supported",
        "blockers",
    )
    assert not hasattr(CapabilityRequest, "to_dict")
    assert not hasattr(CapabilityResponse, "to_dict")


def test_probe_source_is_required_before_passed_evidence_is_verified() -> None:
    common = {
        "provider": "characterization",
        "device_type": "flagos",
        "profile_hash": "a" * 64,
        "operator": "matmul",
        "dtype": "complex128",
        "passed": True,
        "forward": True,
        "backward": True,
    }

    declared = CapabilityEvidence(probe_source="provider_declaration", **common)
    observed = CapabilityEvidence(probe_source="runtime_probe", **common)

    assert declared.is_verified is False
    assert observed.is_verified is True
    assert declared.evidence_id != observed.evidence_id


def test_platform_discovery_preserves_unknown_facts_without_certifying_support() -> (
    None
):
    device = PlatformDevice(
        device_type="flagos",
        index=0,
        name="unverified-device",
        provider="characterization",
        available=True,
    )
    memory = MemorySnapshot()

    assert device.available is True
    assert device.memory_bytes is None
    assert memory == MemorySnapshot(
        allocated_bytes=None,
        reserved_bytes=None,
        free_bytes=None,
        total_bytes=None,
    )
    assert not hasattr(device, "evidence_level")
    assert not hasattr(device, "support_status")


def test_public_exports_do_not_make_private_target_snapshot_authoritative() -> None:
    assert fq.BackendCapabilities is BackendCapabilities
    assert fq.CloudBackendProfile is CloudBackendProfile
    assert not hasattr(fq, "TargetCapabilities")

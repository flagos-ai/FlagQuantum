"""Focused contracts for application-level preflight workflows."""

import torch

import flagquantum as fq
import flagquantum.deployment as deployment
from flagquantum.deployment import CloudBackendProfile
from flagquantum.services import (
    capabilities,
    preflight_deployment,
    preflight_execution,
)


def test_execution_preflight_is_machine_readable() -> None:
    theta = torch.tensor(0.2, requires_grad=True)
    circuit = fq.Circuit(2).ry(0, theta=theta).cx(0, 1)

    preflight = preflight_execution(
        circuit, require_gradients=True, target="expectation"
    )

    assert preflight.validation.valid
    assert "autograd" in preflight.validation.capabilities
    assert preflight.validation.to_dict()["metadata"]["n_wires"] == 2
    assert preflight.executable
    assert preflight.selected_backend
    assert preflight.to_dict()["validation"]["valid"] is True


def test_deployment_preflight_rejects_backend_capacity() -> None:
    circuit = fq.Circuit(3).h(0)
    backend = CloudBackendProfile.simulator(2)

    report = preflight_deployment(circuit, backend=backend)

    assert not report.approved_for_submission
    assert report.validation.errors[0].code == "BACKEND_CAPACITY_EXCEEDED"


def test_deployment_preflight_rejects_unsupported_backend_gate() -> None:
    backend = CloudBackendProfile(
        provider="test", name="restricted", n_wires=2, basis_gates=("x", "cx")
    )

    report = preflight_deployment(fq.Circuit(2).h(0).cx(0, 1), backend=backend)

    assert not report.approved_for_submission
    assert report.validation.errors[0].code == "BACKEND_GATE_UNSUPPORTED"


def test_deployment_preflight_builds_and_seals_package() -> None:
    circuit = fq.Circuit(2).h(0).cx(0, 1)
    backend = CloudBackendProfile.simulator(2)

    report = preflight_deployment(circuit, backend=backend, shots=64)

    assert report.approved_for_submission
    assert report.package is not None
    assert deployment.validate_deployment_package(report.package) is report.package
    assert "package" not in report.to_dict()


def test_deployment_preflight_fails_closed_for_invalid_shots() -> None:
    report = preflight_deployment(
        fq.Circuit(1).h(0), backend=CloudBackendProfile.simulator(1), shots=0
    )

    assert not report.approved_for_submission
    assert report.blockers[0].code == "INVALID_SHOT_COUNT"


def test_capability_manifest_is_versioned() -> None:
    manifest = capabilities()

    assert manifest["schema"] == "flagquantum_service_capabilities_v1"
    assert manifest["version"] == fq.__version__
    assert manifest["contracts"]["deployment_preflight"] is True

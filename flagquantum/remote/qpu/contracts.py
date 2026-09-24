"""Remote task contracts for quantum execution providers."""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from ...deployment.cloud import (
    CloudBackendProfile,
    DeploymentPackage,
    DeploymentPackageIdentityError,
    validate_deployment_package,
)

DEPLOYMENT_SUBMISSION_RECEIPT_SCHEMA = "flagquantum_submission_receipt_v1"


@dataclass(frozen=True)
class ProviderTaskHandle:
    provider: str
    task_id: str
    backend_name: str
    payload: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DeploymentResult:
    handle: ProviderTaskHandle
    counts: Mapping[str, int]
    shots: int
    metadata: Mapping[str, Any] = field(default_factory=dict)


def build_submission_receipt(
    package: DeploymentPackage,
    payload: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Attach immutable package identity to a provider submission receipt."""

    validate_deployment_package(package)
    return dict(payload or {}) | {
        "deployment_receipt_schema": DEPLOYMENT_SUBMISSION_RECEIPT_SCHEMA,
        "deployment_package_schema": package.metadata["deployment_package_schema"],
        "deployment_program_format": package.metadata["deployment_program_format"],
        "routing_evidence_sha256": package.metadata["routing_evidence_sha256"],
        "deployment_artifact_sha256": package.metadata["deployment_artifact_sha256"],
    }


def build_result_metadata(
    handle: ProviderTaskHandle,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Carry submission identity into provider result metadata."""

    identity_keys = (
        "deployment_receipt_schema",
        "deployment_package_schema",
        "deployment_program_format",
        "routing_evidence_sha256",
        "deployment_artifact_sha256",
    )
    identity = {key: handle.payload.get(key) for key in identity_keys}
    if identity["deployment_receipt_schema"] != DEPLOYMENT_SUBMISSION_RECEIPT_SCHEMA:
        raise DeploymentPackageIdentityError(
            "deployment handle is missing a supported submission receipt"
        )
    for key in identity_keys[1:]:
        if not identity[key]:
            raise DeploymentPackageIdentityError(
                f"deployment handle is missing identity field {key!r}"
            )
    return dict(metadata or {}) | identity


def validate_deployment_result(result: DeploymentResult) -> DeploymentResult:
    """Validate the receipt-to-result identity chain and shot accounting."""

    expected = build_result_metadata(result.handle)
    for key, value in expected.items():
        if result.metadata.get(key) != value:
            raise DeploymentPackageIdentityError(
                f"deployment result identity mismatch for {key!r}"
            )
    if result.shots != sum(int(count) for count in result.counts.values()):
        raise DeploymentPackageIdentityError(
            "deployment result shots do not match its counts"
        )
    return result


class QuantumProvider:
    """Minimal adapter protocol for externally controlled quantum targets."""

    provider: str = "provider"

    def list_devices(
        self, n_qubits: int | None = None
    ) -> tuple[CloudBackendProfile, ...]:
        legacy_method = type(self).discover_backends
        if legacy_method is not QuantumProvider.discover_backends:
            return legacy_method(self, n_wires=n_qubits)
        raise NotImplementedError(f"{self.provider} list_devices is not implemented")

    def discover_backends(
        self, n_wires: int | None = None
    ) -> tuple[CloudBackendProfile, ...]:
        """Deprecated compatibility alias for :meth:`list_devices`."""

        warnings.warn(
            "discover_backends() is deprecated; use list_devices(). Removal is "
            "planned for 0.4.0 after the 0.3.x migration window.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.list_devices(n_qubits=n_wires)

    def submit(self, package: DeploymentPackage) -> ProviderTaskHandle:
        raise NotImplementedError(f"{self.provider} submit is not implemented")

    def query_status(self, handle: ProviderTaskHandle) -> str:
        raise NotImplementedError(f"{self.provider} query_status is not implemented")

    def fetch_result(self, handle: ProviderTaskHandle) -> DeploymentResult:
        raise NotImplementedError(f"{self.provider} fetch_result is not implemented")

    def run(self, package: DeploymentPackage) -> DeploymentResult:
        handle = self.submit(package)
        status = self.query_status(handle)
        if status not in {"Finished", "Completed", "Done"}:
            raise RuntimeError(
                f"Deployment task {handle.task_id} ended with status {status!r}."
            )
        return validate_deployment_result(self.fetch_result(handle))


__all__ = (
    "DEPLOYMENT_SUBMISSION_RECEIPT_SCHEMA",
    "DeploymentResult",
    "ProviderTaskHandle",
    "QuantumProvider",
    "build_result_metadata",
    "build_submission_receipt",
    "validate_deployment_result",
)

"""FieldQuantum cloud simulator execution provider."""

from __future__ import annotations

from typing import Any

from ...deployment.cloud import (
    CloudBackendProfile,
    DeploymentPackage,
    DeploymentResult,
    ProviderTaskHandle,
    build_result_metadata,
    build_submission_receipt,
    validate_deployment_package,
)
from .http import HttpQuantumProvider, ProviderCredentials
from .result_parsing import _extract_counts


class FieldQuantumProvider(HttpQuantumProvider):
    """Adapter for the FieldQuantum cloud simulator service."""

    def __init__(
        self,
        *,
        base_url: str = "https://fieldquantum.tech",
        token: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            provider="fieldquantum",
            base_url=base_url,
            credentials=ProviderCredentials(token=token),
            default_backend="fieldquantum_sim",
            **kwargs,
        )

    def discover_backends(
        self, n_wires: int | None = None
    ) -> tuple[CloudBackendProfile, ...]:
        return (
            CloudBackendProfile(
                provider="fieldquantum",
                name="fieldquantum_sim",
                n_wires=n_wires or self.default_n_wires,
                is_simulator=True,
                metadata={"source": "static"},
            ),
        )

    def submit(self, package: DeploymentPackage) -> ProviderTaskHandle:
        validate_deployment_package(package)
        response = self.transport.post_json(
            self._url("/task/run"),
            {"mode": "sample", "qasm": package.qasm, "shots": package.shots},
            self._headers(),
            self.timeout,
        )
        task_id = response.get("task_id")
        if task_id is None:
            raise RuntimeError(
                "fieldquantum submit response does not contain a task id."
            )
        return ProviderTaskHandle(
            "fieldquantum",
            str(task_id),
            package.backend.name,
            build_submission_receipt(package, response),
        )

    def query_status(self, handle: ProviderTaskHandle) -> str:
        response = self.transport.get_json(
            self._url("/task/status/{task_id}", task_id=handle.task_id),
            self._headers(),
            self.timeout,
        )
        raw = str(response.get("status", "unknown")).lower()
        return {
            "submitted": "Running",
            "pending": "Running",
            "queued": "Running",
            "running": "Running",
            "finished": "Finished",
            "failed": "Failed",
            "error": "Failed",
        }.get(raw, "Running")

    def fetch_result(self, handle: ProviderTaskHandle) -> DeploymentResult:
        response = self.transport.get_json(
            self._url("/task/result/{task_id}", task_id=handle.task_id),
            self._headers(),
            self.timeout,
        )
        if str(response.get("status", "")).lower() in {"failed", "error"}:
            raise RuntimeError(f"fieldquantum task {handle.task_id} failed.")
        counts = _extract_counts(response, provider="fieldquantum")
        return DeploymentResult(
            handle=handle,
            counts=counts,
            shots=sum(counts.values()),
            metadata=build_result_metadata(handle, response),
        )


__all__ = ["FieldQuantumProvider"]

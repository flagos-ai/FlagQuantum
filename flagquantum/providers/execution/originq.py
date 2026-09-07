"""Origin Quantum SDK execution provider."""

from __future__ import annotations

from typing import Any, Mapping

from ...deployment.cloud import (
    CloudBackendProfile,
    DeploymentPackage,
    DeploymentResult,
    ProviderTaskHandle,
    QuantumProvider,
    build_result_metadata,
    build_submission_receipt,
    validate_deployment_package,
)
from .result_parsing import _flip_counts


class OriginQProvider(QuantumProvider):
    """OriginQ adapter placeholder for the provider's SDK-based contract."""

    provider = "originq"

    def __init__(
        self,
        *,
        base_url: str = "http://pyqanda-admin.qpanda.cn",
        token: str | None = None,
        sdk: Any = None,
        default_backend: str = "originq",
        default_n_wires: int = 32,
        **_: Any,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.sdk = sdk
        self.default_backend = default_backend
        self.default_n_wires = int(default_n_wires)

    def discover_backends(
        self, n_wires: int | None = None
    ) -> tuple[CloudBackendProfile, ...]:
        if self.sdk is None:
            return (
                CloudBackendProfile(
                    provider=self.provider,
                    name=self.default_backend,
                    n_wires=n_wires or self.default_n_wires,
                    metadata={"source": "sdk-required", "url": self.base_url},
                ),
            )
        rows = self.sdk.backends()
        names = rows.keys() if isinstance(rows, Mapping) else rows
        return tuple(
            CloudBackendProfile(
                provider=self.provider,
                name=str(name),
                n_wires=n_wires or self.default_n_wires,
            )
            for name in names
        )

    def _missing_sdk(self) -> NotImplementedError:
        return NotImplementedError(
            "originq uses the provider SDK contract; pass an sdk adapter to OriginQProvider."
        )

    def submit(self, package: DeploymentPackage) -> ProviderTaskHandle:
        validate_deployment_package(package)
        if self.sdk is None:
            raise self._missing_sdk()
        job = self.sdk.run(
            package.qasm, backend=package.backend.name, shots=package.shots
        )
        task_id = job.job_id() if hasattr(job, "job_id") else getattr(job, "id", None)
        if task_id is None:
            raise RuntimeError(
                "originq sdk submit response does not contain a task id."
            )
        return ProviderTaskHandle(
            "originq",
            str(task_id),
            package.backend.name,
            build_submission_receipt(package, {"job": job}),
        )

    def query_status(self, handle: ProviderTaskHandle) -> str:
        job = handle.payload.get("job")
        if job is None:
            raise self._missing_sdk()
        status = str(job.status())
        return {"FINISHED": "Finished", "FAILED": "Failed", "WAITING": "Running"}.get(
            status.upper(), status
        )

    def fetch_result(self, handle: ProviderTaskHandle) -> DeploymentResult:
        job = handle.payload.get("job")
        if job is None:
            raise self._missing_sdk()
        result = job.result()
        counts = result.get_counts() if hasattr(result, "get_counts") else result
        normalized = _flip_counts(counts)
        return DeploymentResult(
            handle=handle,
            counts=normalized,
            shots=sum(normalized.values()),
            metadata=build_result_metadata(handle, {"raw": result}),
        )


__all__ = ["OriginQProvider"]

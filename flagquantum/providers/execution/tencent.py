"""Tencent Quantum Cloud execution provider."""

from __future__ import annotations

from typing import Any, Mapping

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
from .result_parsing import _extract_counts, _strip_barrier


class TencentQuantumProvider(HttpQuantumProvider):
    """Adapter for Tencent Quantum Cloud."""

    def __init__(
        self,
        *,
        base_url: str = "https://quantum.tencent.com/cloud/quk",
        token: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            provider="tencent",
            base_url=base_url,
            credentials=ProviderCredentials(token=token),
            default_backend="tianji_s2",
            **kwargs,
        )

    def _headers(self) -> dict[str, str]:
        headers = super()._headers()
        headers.setdefault("user-agent", "Mozilla/5.0")
        return headers

    def discover_backends(
        self, n_wires: int | None = None
    ) -> tuple[CloudBackendProfile, ...]:
        try:
            payload = self.transport.post_json(
                self._url("/device/find"), {}, self._headers(), self.timeout
            )
        except Exception:
            return super().discover_backends(n_wires)
        profiles = []
        for row in payload.get("devices", []):
            if not isinstance(row, Mapping):
                continue
            name = str(row.get("id", row.get("name", self.default_backend)))
            qubits = int(
                row.get("n_wires", row.get("bits", n_wires or self.default_n_wires))
            )
            if n_wires is None or qubits >= n_wires:
                profiles.append(
                    CloudBackendProfile(
                        provider="tencent",
                        name=name,
                        n_wires=qubits,
                        metadata=dict(row),
                    )
                )
        return tuple(profiles) or super().discover_backends(n_wires)

    def submit(self, package: DeploymentPackage) -> ProviderTaskHandle:
        validate_deployment_package(package)
        response = self.transport.post_json(
            self._url("/task/submit"),
            {
                "device": f"{package.backend.name}?o={package.metadata.get('qos_option', 2)}",
                "shots": package.shots,
                "source": _strip_barrier(package.qasm),
                "version": "1",
                "lang": "OPENQASM",
                "prior": int(package.metadata.get("priority", 1)),
            },
            self._headers(),
            self.timeout,
        )
        task_id = None
        for row in response.get("tasks", []):
            if isinstance(row, Mapping) and "id" in row and "err" not in row:
                task_id = row["id"]
                break
        if task_id is None:
            task_id = response.get("task_id", response.get("id"))
        if task_id is None:
            raise RuntimeError("tencent submit response does not contain a task id.")
        return ProviderTaskHandle(
            "tencent",
            str(task_id),
            package.backend.name,
            build_submission_receipt(package, response),
        )

    def _task_detail(self, task_id: str) -> Mapping[str, Any]:
        response = self.transport.post_json(
            self._url("/task/detail"), {"id": task_id}, self._headers(), self.timeout
        )
        task = response.get("task", response)
        return task if isinstance(task, Mapping) else response

    def query_status(self, handle: ProviderTaskHandle) -> str:
        state = str(self._task_detail(handle.task_id).get("state", "pending")).lower()
        return {
            "completed": "Finished",
            "failed": "Failed",
            "pending": "Running",
            "scheduled": "Running",
        }.get(state, "Running")

    def fetch_result(self, handle: ProviderTaskHandle) -> DeploymentResult:
        detail = self._task_detail(handle.task_id)
        result = detail.get("result", detail.get("results", detail))
        counts = _extract_counts(
            result if isinstance(result, Mapping) else detail, provider="tencent"
        )
        return DeploymentResult(
            handle=handle,
            counts=counts,
            shots=sum(counts.values()),
            metadata=build_result_metadata(handle, detail),
        )


__all__ = ["TencentQuantumProvider"]

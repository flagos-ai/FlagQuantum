"""Shared HTTP transport and provider implementation for quantum services."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, cast
from urllib import parse, request

from ..deployment.cloud import (
    CloudBackendProfile,
    DeploymentPackage,
    validate_deployment_package,
)
from .contracts import (
    DeploymentResult,
    ProviderTaskHandle,
    QuantumProvider,
    build_result_metadata,
    build_submission_receipt,
)
from .result_parsing import _extract_counts


class QuantumCloudTransport(Protocol):
    """Small transport protocol used by provider adapters."""

    def post_json(
        self,
        url: str,
        payload: Mapping[str, Any],
        headers: Mapping[str, str],
        timeout: float,
    ) -> Mapping[str, Any]: ...

    def get_json(
        self,
        url: str,
        headers: Mapping[str, str],
        timeout: float,
    ) -> Mapping[str, Any]: ...

    def post_form(
        self,
        url: str,
        payload: Mapping[str, Any],
        headers: Mapping[str, str],
        timeout: float,
    ) -> Mapping[str, Any]: ...


class UrllibTransport:
    """Standard-library JSON/form transport."""

    def post_json(
        self,
        url: str,
        payload: Mapping[str, Any],
        headers: Mapping[str, str],
        timeout: float,
    ) -> Mapping[str, Any]:
        data = json.dumps(dict(payload)).encode("utf-8")
        req = request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json", **dict(headers)},
            method="POST",
        )
        with request.urlopen(req, timeout=timeout) as response:
            return cast(Mapping[str, Any], json.loads(response.read().decode("utf-8")))

    def get_json(
        self,
        url: str,
        headers: Mapping[str, str],
        timeout: float,
    ) -> Mapping[str, Any]:
        req = request.Request(url, headers=dict(headers), method="GET")
        with request.urlopen(req, timeout=timeout) as response:
            return cast(Mapping[str, Any], json.loads(response.read().decode("utf-8")))

    def post_form(
        self,
        url: str,
        payload: Mapping[str, Any],
        headers: Mapping[str, str],
        timeout: float,
    ) -> Mapping[str, Any]:
        data = parse.urlencode(dict(payload)).encode("utf-8")
        req = request.Request(
            url,
            data=data,
            headers={
                "Content-Type": "application/x-www-form-urlencoded",
                **dict(headers),
            },
            method="POST",
        )
        with request.urlopen(req, timeout=timeout) as response:
            return cast(Mapping[str, Any], json.loads(response.read().decode("utf-8")))


@dataclass(frozen=True)
class ProviderEndpoints:
    """Provider endpoint paths relative to ``base_url``."""

    submit: str = "/jobs"
    status: str = "/jobs/{task_id}/status"
    result: str = "/jobs/{task_id}/result"
    backends: str = "/backends"


@dataclass(frozen=True)
class ProviderCredentials:
    """Provider authentication material."""

    token: str | None = None
    user_id: str | None = None
    extra_headers: Mapping[str, str] = field(default_factory=dict)


class HttpQuantumProvider(QuantumProvider):
    """Provider adapter for OpenQASM-style HTTP quantum-cloud APIs."""

    provider = "http"

    def __init__(
        self,
        *,
        provider: str,
        base_url: str,
        credentials: ProviderCredentials | None = None,
        endpoints: ProviderEndpoints | None = None,
        transport: QuantumCloudTransport | None = None,
        timeout: float = 30.0,
        default_backend: str = "default",
        default_n_wires: int = 32,
    ) -> None:
        self.provider = provider
        self.base_url = base_url.rstrip("/")
        self.credentials = credentials or ProviderCredentials()
        self.endpoints = endpoints or ProviderEndpoints()
        self.transport = transport or UrllibTransport()
        self.timeout = float(timeout)
        self.default_backend = default_backend
        self.default_n_wires = int(default_n_wires)

    def _url(self, path: str, **values: Any) -> str:
        return self.base_url + path.format(**values)

    def _headers(self) -> dict[str, str]:
        headers = dict(self.credentials.extra_headers)
        if self.credentials.token:
            headers.setdefault("Authorization", f"Bearer {self.credentials.token}")
        if self.credentials.user_id:
            headers.setdefault("X-User-Id", self.credentials.user_id)
        return headers

    def discover_backends(
        self, n_wires: int | None = None
    ) -> tuple[CloudBackendProfile, ...]:
        try:
            payload = self.transport.get_json(
                self._url(self.endpoints.backends),
                self._headers(),
                self.timeout,
            )
        except Exception:
            return (
                CloudBackendProfile(
                    provider=self.provider,
                    name=self.default_backend,
                    n_wires=n_wires or self.default_n_wires,
                    metadata={"source": "fallback"},
                ),
            )
        rows = payload.get("backends", payload.get("data", []))
        profiles = []
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            qubits = int(
                row.get(
                    "n_wires",
                    row.get(
                        "nqubits", row.get("qubits", n_wires or self.default_n_wires)
                    ),
                )
            )
            if n_wires is not None and qubits < n_wires:
                continue
            profiles.append(
                CloudBackendProfile(
                    provider=self.provider,
                    name=str(row.get("name", row.get("backend", self.default_backend))),
                    n_wires=qubits,
                    basis_gates=tuple(row.get("basis_gates", ())),
                    is_simulator=bool(row.get("is_simulator", False)),
                    metadata=dict(row),
                )
            )
        if not profiles:
            profiles.append(
                CloudBackendProfile(
                    provider=self.provider,
                    name=self.default_backend,
                    n_wires=n_wires or self.default_n_wires,
                    metadata={"source": "empty-list-fallback"},
                )
            )
        return tuple(profiles)

    def _submit_payload(self, package: DeploymentPackage) -> dict[str, Any]:
        return {
            "schema": package.metadata.get("deployment_package_schema"),
            "name": package.name,
            "backend": package.backend.name,
            "shots": package.shots,
            "qasm": package.qasm,
            "qasm_version": package.qasm_version,
            "routing_evidence_sha256": package.metadata.get("routing_evidence_sha256"),
            "deployment_artifact_sha256": package.metadata.get(
                "deployment_artifact_sha256"
            ),
            "metadata": dict(package.metadata),
        }

    def submit(self, package: DeploymentPackage) -> ProviderTaskHandle:
        validate_deployment_package(package)
        response = self.transport.post_json(
            self._url(self.endpoints.submit),
            self._submit_payload(package),
            self._headers(),
            self.timeout,
        )
        task_id = response.get("task_id", response.get("job_id", response.get("id")))
        if task_id is None:
            raise RuntimeError(
                f"{self.provider} submit response does not contain a task id."
            )
        return ProviderTaskHandle(
            provider=self.provider,
            task_id=str(task_id),
            backend_name=package.backend.name,
            payload=build_submission_receipt(package, response),
        )

    def query_status(self, handle: ProviderTaskHandle) -> str:
        response = self.transport.get_json(
            self._url(self.endpoints.status, task_id=handle.task_id),
            self._headers(),
            self.timeout,
        )
        return str(response.get("status", response.get("state", "Unknown")))

    def fetch_result(self, handle: ProviderTaskHandle) -> DeploymentResult:
        response = self.transport.get_json(
            self._url(self.endpoints.result, task_id=handle.task_id),
            self._headers(),
            self.timeout,
        )
        counts = _extract_counts(response, provider=self.provider)
        return DeploymentResult(
            handle=handle,
            counts=counts,
            shots=sum(counts.values()),
            metadata=build_result_metadata(handle, response),
        )


__all__ = [
    "HttpQuantumProvider",
    "ProviderCredentials",
    "ProviderEndpoints",
    "QuantumCloudTransport",
    "UrllibTransport",
]

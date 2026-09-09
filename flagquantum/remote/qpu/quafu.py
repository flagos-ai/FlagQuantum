"""Quafu SQC execution provider."""

from __future__ import annotations

import hashlib
import os
import time
from typing import Any, Mapping
from urllib import parse

from ...deployment.cloud import (
    CloudBackendProfile,
    DeploymentPackage,
    validate_deployment_package,
)
from .contracts import (
    DEPLOYMENT_SUBMISSION_RECEIPT_SCHEMA,
    DeploymentResult,
    ProviderTaskHandle,
    build_result_metadata,
    build_submission_receipt,
    validate_deployment_result,
)
from .http import HttpQuantumProvider, ProviderCredentials
from .result_parsing import _extract_counts


class QuafuProvider(HttpQuantumProvider):
    """HTTP adapter for the Quafu SQC cloud ``quafusqc`` contract.

    Authentication defaults to the provider-specific ``QUAFU_API_TOKEN`` environment
    variable
    used by ``quafusqc``. Tokens passed explicitly always take precedence.
    """

    def __init__(
        self,
        *,
        base_url: str = "https://quafu-sqc.baqis.ac.cn",
        token: str | None = None,
        poll_interval: float = 0.2,
        result_timeout: float = 300.0,
        reverse_result_bits: bool = False,
        **kwargs: Any,
    ) -> None:
        resolved_token = token or os.getenv("QUAFU_API_TOKEN")
        super().__init__(
            provider="quafu",
            base_url=base_url,
            credentials=ProviderCredentials(token=resolved_token),
            default_backend="quafu",
            **kwargs,
        )
        if poll_interval < 0:
            raise ValueError("poll_interval must be non-negative")
        if result_timeout < 0:
            raise ValueError("result_timeout must be non-negative")
        self.poll_interval = float(poll_interval)
        self.result_timeout = float(result_timeout)
        self.reverse_result_bits = bool(reverse_result_bits)

    def verify(self) -> Any:
        """Verify the configured token using the official platform endpoint."""

        if not self.credentials.token:
            raise RuntimeError(
                "A Quafu token is required; pass token=... or set QUAFU_API_TOKEN."
            )
        return self.transport.get_json(
            self._url("/task/verify"), self._headers(), self.timeout
        )

    def _headers(self) -> dict[str, str]:
        headers = dict(self.credentials.extra_headers)
        if self.credentials.token:
            headers["token"] = self.credentials.token
        return headers

    def discover_backends(
        self, n_wires: int | None = None
    ) -> tuple[CloudBackendProfile, ...]:
        """Discover chips through ``Task.status(0)``.

        The SQC status response reports queue availability but not qubit counts,
        so the requested width (or the configured conservative fallback) is
        retained in each profile and the limitation is exposed in metadata.
        """

        response = self.transport.get_json(
            self._url("/task/status/0"), self._headers(), self.timeout
        )
        rows: Any = response
        if isinstance(response, Mapping):
            rows = response.get("data", response.get("status", response))
        if not isinstance(rows, Mapping):
            raise RuntimeError("quafu backend status response is not a mapping")
        profiles = []
        for name, queue_status in rows.items():
            profiles.append(
                CloudBackendProfile(
                    provider="quafu",
                    name=str(name),
                    n_wires=n_wires or self.default_n_wires,
                    metadata={
                        "source": "quafu-task-status",
                        "queue_status": queue_status,
                        "n_wires_source": "requested-or-fallback",
                    },
                )
            )
        if not profiles:
            raise RuntimeError("quafu backend status response contains no chips")
        return tuple(profiles)

    def fetch_chip_info(self, chip: str) -> Mapping[str, Any]:
        """Fetch the current QuarkCircuit calibration and topology snapshot."""

        name = str(chip).strip()
        if not name:
            raise ValueError("Quafu chip name cannot be empty")
        response = self.transport.get_json(
            self._url(f"/task/backendtest/{parse.quote(name, safe='')}1"),
            self._headers(),
            self.timeout,
        )
        if not isinstance(response, Mapping):
            raise RuntimeError("quafu chip-info response is not a mapping")
        required = {"calibration_time", "qubits_info", "couplers_info"}
        missing = sorted(required.difference(response))
        if missing:
            raise RuntimeError(
                "quafu chip-info response is missing " + ", ".join(missing)
            )
        return response

    def submit(self, package: DeploymentPackage) -> ProviderTaskHandle:
        validate_deployment_package(package)
        query = parse.urlencode(
            {"name": package.name, "chip": package.backend.name, "shots": package.shots}
        )
        response = self.transport.post_json(
            self._url(f"/task/run/?{query}"),
            {
                "circuit": package.qasm,
                "compile": bool(package.metadata.get("provider_compile", True)),
                "options": dict(package.metadata.get("provider_options", {})),
            },
            self._headers(),
            self.timeout,
        )
        task_id = (
            response
            if isinstance(response, int)
            else response.get("task_id", response.get("tid", response.get("id")))
        )
        if task_id is None:
            raise RuntimeError("quafu submit response does not contain a task id.")
        return ProviderTaskHandle(
            "quafu",
            str(task_id),
            package.backend.name,
            build_submission_receipt(
                package, response if isinstance(response, Mapping) else {}
            ),
        )

    def submit_qasm(
        self,
        qasm: str,
        *,
        chip: str,
        name: str,
        shots: int,
    ) -> ProviderTaskHandle:
        """Submit OpenQASM 2.0 while requesting no provider compilation.

        Quafu may still lower gates or remap qubits. The returned ``transpiled``
        program, when present, is the authoritative execution representation.
        """

        program = str(qasm)
        backend = str(chip).strip()
        task_name = str(name).strip()
        if not program.lstrip().startswith("OPENQASM 2.0;"):
            raise ValueError("Quafu QASM submission requires OpenQASM 2.0")
        if not backend or not task_name:
            raise ValueError("Quafu QASM submission requires chip and name")
        if int(shots) <= 0 or int(shots) % 1024:
            raise ValueError("Quafu shots must be a positive multiple of 1024")
        digest = hashlib.sha256(program.encode()).hexdigest()
        query = parse.urlencode(
            {"name": task_name, "chip": backend, "shots": int(shots)}
        )
        response = self.transport.post_json(
            self._url(f"/task/run/?{query}"),
            {"circuit": program, "compile": False, "options": {}},
            self._headers(),
            self.timeout,
        )
        task_id = (
            response
            if isinstance(response, int)
            else response.get("task_id", response.get("tid", response.get("id")))
        )
        if task_id is None:
            raise RuntimeError("quafu submit response does not contain a task id.")
        receipt = {
            "deployment_receipt_schema": DEPLOYMENT_SUBMISSION_RECEIPT_SCHEMA,
            "deployment_package_schema": "flagquantum_submitted_qasm_v1",
            "deployment_program_format": "openqasm-2",
            "routing_evidence_sha256": digest,
            "deployment_artifact_sha256": digest,
            "submitted_qasm_sha256": digest,
            "compile": False,
        }
        if isinstance(response, Mapping):
            receipt["submit"] = dict(response)
        return ProviderTaskHandle("quafu", str(task_id), backend, receipt)

    def query_status(self, handle: ProviderTaskHandle) -> str:
        response = self.transport.get_json(
            self._url("/task/status/{task_id}", task_id=handle.task_id),
            self._headers(),
            self.timeout,
        )
        if isinstance(response, Mapping):
            status = str(
                response.get(
                    "status",
                    response.get("state", response.get("task_status", "Running")),
                )
            )
        else:
            status = str(response)
        return {
            "completed": "Finished",
            "finished": "Finished",
            "done": "Finished",
            "failed": "Failed",
            "cancelled": "Cancelled",
            "canceled": "Cancelled",
        }.get(status.lower(), status)

    def cancel(self, handle: ProviderTaskHandle) -> Any:
        """Request cancellation through ``Task.cancel(tid)``."""

        return self.transport.get_json(
            self._url("/task/cancel/{task_id}", task_id=handle.task_id),
            self._headers(),
            self.timeout,
        )

    def fetch_result(self, handle: ProviderTaskHandle) -> DeploymentResult:
        response = self.transport.get_json(
            self._url("/task/result/{task_id}", task_id=handle.task_id),
            self._headers(),
            self.timeout,
        )
        if not response:
            raise RuntimeError(
                f"quafu task {handle.task_id} has no result yet; query status and retry"
            )
        counts = _extract_counts(
            response, provider="quafu", flip=self.reverse_result_bits
        )
        return DeploymentResult(
            handle=handle,
            counts=counts,
            shots=sum(counts.values()),
            metadata=build_result_metadata(handle, response),
        )

    def run(self, package: DeploymentPackage) -> DeploymentResult:
        """Poll an asynchronous Quafu task until it reaches a terminal state."""

        handle = self.submit(package)
        deadline = time.monotonic() + self.result_timeout
        while True:
            status = self.query_status(handle)
            if status in {"Finished", "Completed", "Done"}:
                try:
                    return validate_deployment_result(self.fetch_result(handle))
                except RuntimeError as exc:
                    if "has no result yet" not in str(exc):
                        raise
            if status in {"Failed", "Cancelled", "Canceled"}:
                raise RuntimeError(
                    f"Deployment task {handle.task_id} ended with status {status!r}."
                )
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Quafu task {handle.task_id} did not finish within "
                    f"{self.result_timeout:g} seconds."
                )
            time.sleep(self.poll_interval)


__all__ = ["QuafuProvider"]

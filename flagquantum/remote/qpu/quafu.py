"""Quafu SQC execution provider."""

from __future__ import annotations

import hashlib
import os
import re
import time
import uuid
from collections.abc import Mapping, Sequence
from typing import Any
from urllib import error, parse

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

_QREG_DECLARATION = re.compile(r"\bqreg\s+q\s*\[\s*(\d+)\s*\]\s*;")
_QUBIT_REFERENCE = re.compile(r"\bq\s*\[\s*(\d+)\s*\]")
_TASK_API_PROTOCOL = "task_api_v1"
_SQC_PROTOCOL = "sqc_legacy"


def _service_options(n_wires: int, raw_qubits: Sequence[int]) -> dict[str, Any]:
    """Validate the optional physical mapping before any remote side effect."""
    if not isinstance(raw_qubits, Sequence) or isinstance(raw_qubits, (str, bytes)):
        raise ValueError("target_qubits must be a sequence of physical qubits")
    qubits = list(raw_qubits)
    if qubits and (
        len(qubits) != n_wires
        or any(type(qubit) is not int or qubit < 0 for qubit in qubits)
        or len(set(qubits)) != len(qubits)
    ):
        raise ValueError(
            "target_qubits must map every logical wire to a unique nonnegative integer"
        )
    return {"compiler": "quarkcircuit", "target_qubits": qubits}


def _submission_options(
    qasm: str,
    *,
    n_wires: int,
    options: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the selected service-compilation or precompiled submission contract."""

    if options.get("compiler") == "quarkcircuit":
        resolved = _service_options(n_wires, options.get("target_qubits", ()))
        declaration = _QREG_DECLARATION.search(qasm)
        if declaration is None or int(declaration.group(1)) != n_wires:
            raise ValueError("Quafu QASM must declare logical qreg q[N]")
        if any(
            int(index) >= n_wires
            for index in _QUBIT_REFERENCE.findall(
                _QREG_DECLARATION.sub("", qasm, count=1)
            )
        ):
            raise ValueError("Quafu QASM references an out-of-range logical wire")
        return {**options, **resolved}

    if "compiler" not in options or options["compiler"] is not None:
        raise ValueError("precompiled Quafu submissions require compiler=None")
    raw_qubits = options.get("target_qubits")
    if not isinstance(raw_qubits, Sequence) or isinstance(raw_qubits, (str, bytes)):
        raise ValueError("precompiled Quafu submissions require target_qubits")
    target_qubits = [int(qubit) for qubit in raw_qubits]
    if (
        len(target_qubits) != n_wires
        or len(set(target_qubits)) != n_wires
        or any(qubit < 0 for qubit in target_qubits)
    ):
        raise ValueError(
            "target_qubits must contain one unique physical qubit per logical wire"
        )
    declaration = _QREG_DECLARATION.search(qasm)
    if declaration is None or int(declaration.group(1)) != n_wires:
        raise ValueError(
            "Quafu QASM must declare logical qreg q[N] matching target_qubits"
        )
    body = _QREG_DECLARATION.sub("", qasm, count=1)
    if any(int(index) >= n_wires for index in _QUBIT_REFERENCE.findall(body)):
        raise ValueError("Quafu QASM may reference only logical q[0] through q[N-1]")
    resolved = dict(options)
    resolved["compiler"] = None
    resolved["target_qubits"] = target_qubits
    return resolved


class QuafuProvider(HttpQuantumProvider):
    """HTTP adapter for the current and legacy Quafu task APIs.

    ``QUAFU_API_KEY`` enables the current task API and
    ``QUAFU_TASK_SERVER_URL`` overrides its base URL. When the current API is
    unavailable before submission, the provider falls back to the legacy SQC
    contract configured by ``QUAFU_API_TOKEN``. Explicit credentials take
    precedence over environment variables.
    """

    def __init__(
        self,
        *,
        base_url: str = "https://quafu-sqc.baqis.ac.cn",
        token: str | None = None,
        task_server_url: str | None = None,
        api_key: str | None = None,
        poll_interval: float = 0.2,
        result_timeout: float = 300.0,
        reverse_result_bits: bool = False,
        **kwargs: Any,
    ) -> None:
        resolved_token = token or os.getenv("QUAFU_API_TOKEN")
        self.task_server_url = (
            task_server_url
            or os.getenv("QUAFU_TASK_SERVER_URL")
            or "https://quafu.com.cn/api/v1"
        ).rstrip("/")
        task_server = parse.urlparse(self.task_server_url)
        if (
            task_server.scheme != "https"
            or not task_server.netloc
            or task_server.username is not None
            or task_server.password is not None
        ):
            raise ValueError("Quafu task_server_url must be an HTTPS URL")
        self.api_key = api_key or os.getenv("QUAFU_API_KEY")
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

    def _task_api_url(self, path: str, **values: Any) -> str:
        return self.task_server_url + path.format(**values)

    def _task_api_headers(self) -> dict[str, str]:
        if not self.api_key:
            return {}
        return {"Authorization": f"Bearer {self.api_key}"}

    @staticmethod
    def _protocol(handle: ProviderTaskHandle) -> str:
        protocol = handle.payload.get("quafu_protocol", _SQC_PROTOCOL)
        return protocol if protocol == _TASK_API_PROTOCOL else _SQC_PROTOCOL

    def _task_api_is_healthy(self) -> bool:
        if not self.api_key:
            return False
        try:
            response = self.transport.get_json(
                self._task_api_url("/healthz"), {}, self.timeout
            )
        except Exception:
            return False
        return isinstance(response, Mapping) and response.get("healthy") is True

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
        """Discover current task-API devices, with legacy SQC fallback.

        The current API reports device capacity and status directly. The legacy
        SQC response has no qubit counts, so fallback profiles retain the
        requested width or configured conservative default.
        """

        try:
            response = self.transport.get_json(
                self._task_api_url("/devices"), {}, self.timeout
            )
            return self._task_api_backends(response, n_wires=n_wires)
        except Exception:
            return self._legacy_backends(n_wires=n_wires)

    def _task_api_backends(
        self, response: Mapping[str, Any], *, n_wires: int | None
    ) -> tuple[CloudBackendProfile, ...]:
        raw_devices = response.get("devices")
        if not isinstance(raw_devices, Sequence) or isinstance(
            raw_devices, (str, bytes)
        ):
            raise RuntimeError("quafu task API devices response is not a sequence")
        rows = list(raw_devices)
        simulator = response.get("simulator")
        if isinstance(simulator, Mapping):
            rows.append(simulator)
        profiles = []
        for row in rows:
            if not isinstance(row, Mapping):
                raise RuntimeError("quafu task API device is not a mapping")
            name = row.get("name")
            capacity = row.get("n_qubits", row.get("max_qubits"))
            if not isinstance(name, str) or not name.strip():
                raise RuntimeError("quafu task API device has no name")
            if type(capacity) is not int or capacity <= 0:
                capacity = n_wires or self.default_n_wires
            profiles.append(
                CloudBackendProfile(
                    provider="quafu",
                    name=name,
                    n_qubits=capacity,
                    basis_gates=tuple(map(str, row.get("basis_gates", ()))),
                    is_simulator=name == "sim",
                    metadata={
                        "source": "quafu-task-api-devices",
                        "status": row.get("status"),
                        "queue": row.get("queue"),
                        "calibration_id": row.get("calibration_id"),
                    },
                )
            )
        if not profiles:
            raise RuntimeError("quafu task API devices response contains no devices")
        return tuple(profiles)

    def _legacy_backends(
        self, *, n_wires: int | None
    ) -> tuple[CloudBackendProfile, ...]:
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
                    n_qubits=n_wires or self.default_n_wires,
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
        options = _submission_options(
            package.qasm,
            n_wires=package.n_wires,
            options=dict(package.metadata.get("provider_options", {})),
        )
        if (
            options.get("compiler") == "quarkcircuit"
            and not options.get("target_qubits")
            and self._task_api_is_healthy()
        ):
            try:
                return self._submit_task_api(package)
            except error.HTTPError as exc:
                if 400 <= exc.code < 500:
                    # A client rejection means the platform did not accept the
                    # task. Server and transport failures after POST remain
                    # ambiguous and must not create a second hardware task.
                    pass
                else:
                    raise
        return self._submit_legacy(package, options)

    def _submit_task_api(self, package: DeploymentPackage) -> ProviderTaskHandle:
        client_ref = str(uuid.uuid4())
        response = self.transport.post_json(
            self._task_api_url("/jobs"),
            {
                "mode": "single",
                "target": package.backend.name,
                "shots": package.shots,
                "circuits": [{"qasm": package.qasm}],
                "client_ref": client_ref,
            },
            self._task_api_headers(),
            self.timeout,
        )
        job_id = response.get("job_id") if isinstance(response, Mapping) else None
        if not isinstance(job_id, str) or not job_id.strip():
            raise RuntimeError("quafu task API response does not contain a job id.")
        receipt = build_submission_receipt(package, response)
        receipt["quafu_protocol"] = _TASK_API_PROTOCOL
        receipt["client_ref"] = client_ref
        return ProviderTaskHandle("quafu", job_id, package.backend.name, receipt)

    def _submit_legacy(
        self, package: DeploymentPackage, options: Mapping[str, Any]
    ) -> ProviderTaskHandle:
        query = parse.urlencode(
            {"name": package.name, "chip": package.backend.name, "shots": package.shots}
        )
        response = self.transport.post_json(
            self._url(f"/task/run/?{query}"),
            {
                "circuit": package.qasm,
                "options": options,
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
        receipt = build_submission_receipt(
            package, response if isinstance(response, Mapping) else {}
        )
        receipt["quafu_protocol"] = _SQC_PROTOCOL
        return ProviderTaskHandle(
            "quafu",
            str(task_id),
            package.backend.name,
            receipt,
        )

    def submit_qasm(
        self,
        qasm: str,
        *,
        chip: str,
        name: str,
        shots: int,
        target_qubits: Sequence[int],
    ) -> ProviderTaskHandle:
        """Submit precompiled logical OpenQASM 2.0 to ordered physical qubits.

        ``q[i]`` remains logical wire ``i``; ``target_qubits[i]`` identifies the
        physical qubit selected for that wire. Quafu receives ``compiler=None``.
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
        options = _submission_options(
            program,
            n_wires=len(target_qubits),
            options={"compiler": None, "target_qubits": target_qubits},
        )
        digest = hashlib.sha256(program.encode()).hexdigest()
        query = parse.urlencode(
            {"name": task_name, "chip": backend, "shots": int(shots)}
        )
        response = self.transport.post_json(
            self._url(f"/task/run/?{query}"),
            {"circuit": program, "options": options},
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
            "compiler": None,
            "target_qubits": options["target_qubits"],
            "quafu_protocol": _SQC_PROTOCOL,
        }
        if isinstance(response, Mapping):
            receipt["submit"] = dict(response)
        return ProviderTaskHandle("quafu", str(task_id), backend, receipt)

    def query_status(self, handle: ProviderTaskHandle) -> str:
        if self._protocol(handle) == _TASK_API_PROTOCOL:
            response = self.transport.get_json(
                self._task_api_url("/jobs/{task_id}/status", task_id=handle.task_id),
                self._task_api_headers(),
                self.timeout,
            )
        else:
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
        """Request cancellation through the API that accepted the task."""

        if self._protocol(handle) == _TASK_API_PROTOCOL:
            return self.transport.post_json(
                self._task_api_url("/jobs/{task_id}/cancel", task_id=handle.task_id),
                {},
                self._task_api_headers(),
                self.timeout,
            )

        return self.transport.get_json(
            self._url("/task/cancel/{task_id}", task_id=handle.task_id),
            self._headers(),
            self.timeout,
        )

    def fetch_result(self, handle: ProviderTaskHandle) -> DeploymentResult:
        protocol = self._protocol(handle)
        if protocol == _TASK_API_PROTOCOL:
            response = self.transport.get_json(
                self._task_api_url("/jobs/{task_id}/results", task_id=handle.task_id),
                self._task_api_headers(),
                self.timeout,
            )
        else:
            response = self.transport.get_json(
                self._url("/task/result/{task_id}", task_id=handle.task_id),
                self._headers(),
                self.timeout,
            )
        if not response:
            raise RuntimeError(
                f"quafu task {handle.task_id} has no result yet; query status and retry"
            )
        result_payload: Mapping[str, Any] = response
        if protocol == _TASK_API_PROTOCOL:
            rows = response.get("counts")
            if (
                isinstance(rows, Sequence)
                and not isinstance(rows, (str, bytes))
                and len(rows) == 1
                and rows[0] is None
            ):
                raise RuntimeError(
                    f"quafu task {handle.task_id} has no result yet; "
                    "query status and retry"
                )
            if (
                not isinstance(rows, Sequence)
                or isinstance(rows, (str, bytes))
                or len(rows) != 1
                or not isinstance(rows[0], Mapping)
            ):
                raise RuntimeError("quafu task API result does not contain counts.")
            result_payload = {"counts": rows[0]}
        counts = _extract_counts(
            result_payload, provider="quafu", flip=self.reverse_result_bits
        )
        return DeploymentResult(
            handle=handle,
            counts=counts,
            shots=sum(counts.values()),
            metadata=build_result_metadata(
                handle, dict(response) | {"quafu_protocol": protocol}
            ),
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

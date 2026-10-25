"""Quafu SDK execution provider with a staged read-only legacy bridge."""

from __future__ import annotations

import hashlib
import os
import re
import time
from collections.abc import Mapping, Sequence
from typing import Any
from urllib import parse

from ...deployment.cloud import (
    CloudBackendProfile,
    DeploymentPackage,
    validate_deployment_package,
)
from ...errors import ValidationError
from ._quafu_calibration import qsteed_chip_info
from ._quafu_physical import (
    backend_snapshot_sha256,
    prepare_physical_qasm,
    validate_physical_result,
)
from ._quafu_sdk import _QuafuSdkAdapter
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
_SDK_PROTOCOL = "quafu_sdk_v1"
_MAX_SHOTS = 8192


def _quafu_backend_name(target: str) -> str:
    """Return the backend portion of a root target or an adapter backend name."""

    provider, separator, backend = str(target).partition(":")
    return (
        backend.strip() if separator and provider.lower() == "quafu" else target.strip()
    )


def _is_simulator_target(target: str) -> bool:
    backend = _quafu_backend_name(target)
    return backend == "sim" or backend.endswith("-sim")


def _validate_quafu_shots(target: str, shots: int | None) -> None:
    """Validate the maintained service's simulator and hardware shot ranges."""

    if type(shots) is not int:
        raise ValueError("Quafu shots must be an integer")
    if _is_simulator_target(target):
        if not 1 <= shots <= _MAX_SHOTS:
            raise ValueError("Quafu simulator shots must be between 1 and 8192")
        return
    if not 1024 <= shots <= _MAX_SHOTS or shots % 1024:
        raise ValueError(
            "Quafu hardware shots must be a multiple of 1024 between 1024 and 8192"
        )


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
    return {"compiler": "qiskit", "target_qubits": qubits}


def _submission_options(
    qasm: str,
    *,
    n_wires: int,
    options: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the selected service-compilation or precompiled submission contract."""

    if options.get("compiler") in {"qiskit", "quarkcircuit"}:
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
    """FlagQuantum adapter for Quafu discovery and remote execution.

    Read-only discovery, calibration, and every static execution lifecycle use
    the official Quafu SDK. The legacy HTTP transport remains private to
    ``fetch_chip_info()`` until the SDK exposes the complete Twin noise schema.
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
        sdk_adapter = kwargs.pop("_sdk_adapter", None)
        if sdk_adapter is not None and not isinstance(sdk_adapter, _QuafuSdkAdapter):
            raise TypeError("_sdk_adapter must be a private Quafu SDK adapter")
        self._sdk_read_adapter = sdk_adapter
        self._sdk_submission_adapter = sdk_adapter
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

    def _read_sdk_adapter(self) -> _QuafuSdkAdapter:
        if self._sdk_read_adapter is None:
            base_url = self.task_server_url.removesuffix("/api/v1")
            self._sdk_read_adapter = _QuafuSdkAdapter.for_discovery(base_url=base_url)
        return self._sdk_read_adapter

    def _submission_sdk_adapter(self) -> _QuafuSdkAdapter:
        if self._sdk_submission_adapter is None:
            base_url = self.task_server_url.removesuffix("/api/v1")
            self._sdk_submission_adapter = _QuafuSdkAdapter.for_submission(
                api_key=self.api_key,
                base_url=base_url,
            )
        return self._sdk_submission_adapter

    @staticmethod
    def _protocol(handle: ProviderTaskHandle) -> str:
        protocol = handle.payload.get("quafu_protocol")
        if protocol != _SDK_PROTOCOL:
            raise ValidationError(
                "This Quafu receipt uses a retired transport protocol; only "
                "quafu_sdk_v1 jobs can be queried, cancelled, or restored."
            )
        return _SDK_PROTOCOL

    def _headers(self) -> dict[str, str]:
        headers = dict(self.credentials.extra_headers)
        if self.credentials.token:
            headers["token"] = self.credentials.token
        return headers

    def list_devices(
        self, n_qubits: int | None = None
    ) -> tuple[CloudBackendProfile, ...]:
        """Discover authoritative devices through the official Quafu SDK."""

        return self._sdk_backends(self._read_sdk_adapter())

    @staticmethod
    def _sdk_backends(
        adapter: _QuafuSdkAdapter,
    ) -> tuple[CloudBackendProfile, ...]:
        profiles = []
        for row in adapter.devices():
            name = row.get("name")
            if not isinstance(name, str) or not name.strip():
                raise RuntimeError("Quafu SDK device has no name")
            details = adapter.backend_metadata(name)
            capacity = details.get("n_qubits", row.get("n_qubits"))
            if type(capacity) is not int or capacity <= 0:
                raise RuntimeError(
                    f"Quafu SDK metadata for {name!r} has no positive n_qubits"
                )
            raw_gates = details.get("basis_gates", ())
            if not isinstance(raw_gates, Sequence) or isinstance(
                raw_gates, (str, bytes)
            ):
                raise RuntimeError(
                    f"Quafu SDK metadata for {name!r} has invalid basis_gates"
                )
            kind = row.get("kind")
            if kind not in {"real", "sim"}:
                raise RuntimeError(f"Quafu SDK device {name!r} has invalid kind")
            limits = details.get("limits", row.get("limits"))
            if limits is not None and not isinstance(limits, Mapping):
                raise RuntimeError(
                    f"Quafu SDK metadata for {name!r} has invalid limits"
                )
            metadata_fields = (
                "status",
                "queue",
                "max_qubits",
                "calibration_id",
                "calibrated_at",
                "calibration_source",
            )
            profiles.append(
                CloudBackendProfile(
                    provider="quafu",
                    name=name,
                    n_qubits=capacity,
                    basis_gates=tuple(map(str, raw_gates)),
                    is_simulator=kind == "sim",
                    metadata={
                        "source": "quafu-sdk",
                        "kind": kind,
                        "limits": limits,
                        **{
                            key: details[key] if key in details else row[key]
                            for key in metadata_fields
                            if key in details or key in row
                        },
                    },
                )
            )
        if not profiles:
            raise RuntimeError("Quafu SDK devices result contains no devices")
        return tuple(profiles)

    def fetch_calibration(
        self,
        device: str,
        *,
        calibration_id: str | None = None,
    ) -> Mapping[str, Any]:
        """Fetch a current or immutable calibration snapshot through the SDK.

        Pass the calibration identity preserved by a job receipt to restore
        the exact historical snapshot. Calibration discovery is public and
        does not require a provider credential.
        """

        name = str(device).strip()
        if not name:
            raise ValueError("Quafu device name cannot be empty")
        sdk_response = self._read_sdk_adapter().calibration(
            name, calibration_id=calibration_id
        )
        return self._validate_sdk_calibration(
            sdk_response,
            device=name,
            calibration_id=calibration_id,
        )

    @staticmethod
    def _validate_sdk_calibration(
        response: Mapping[str, Any],
        *,
        device: str,
        calibration_id: str | None,
    ) -> Mapping[str, Any]:
        required = {"id", "chip", "calibrated_at", "qubits", "couplers"}
        missing = sorted(required.difference(response))
        if missing:
            raise RuntimeError(
                "quafu calibration response is missing " + ", ".join(missing)
            )
        if response["chip"] != device:
            raise RuntimeError(
                f"quafu calibration belongs to {response['chip']!r}, not {device!r}"
            )
        if calibration_id is not None and response["id"] != calibration_id:
            raise RuntimeError(
                "quafu historical calibration identity does not match the request"
            )
        return response

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

    def _qsteed_chip_info(self, chip: str) -> Mapping[str, Any]:
        """Resolve QSteed topology through the maintained SDK read path."""

        return qsteed_chip_info(self.fetch_calibration(chip))

    def submit(self, package: DeploymentPackage) -> ProviderTaskHandle:
        validate_deployment_package(package)
        _validate_quafu_shots(package.backend.name, package.shots)
        options = _submission_options(
            package.qasm,
            n_wires=package.n_qubits,
            options=dict(package.metadata.get("provider_options", {})),
        )
        if options.get("compiler") in {"qiskit", "quarkcircuit"} and not options.get(
            "target_qubits"
        ):
            return self._submit_sdk_logical(package)
        if options.get("compiler") is None:
            return self._submit_sdk_physical(package, options)
        raise ValueError(
            "Quafu SDK service compilation does not accept target_qubits; use "
            "compiler='qsteed' for an explicit physical mapping"
        )

    def _submit_sdk_logical(self, package: DeploymentPackage) -> ProviderTaskHandle:
        response = self._submission_sdk_adapter().submit_logical(
            package.qasm,
            target=package.backend.name,
            shots=package.shots,
            name=package.name,
        )
        receipt = build_submission_receipt(package, response)
        receipt["quafu_protocol"] = _SDK_PROTOCOL
        receipt["service_compiler"] = "qiskit"
        return ProviderTaskHandle(
            "quafu", str(response["job_id"]), package.backend.name, receipt
        )

    def _submit_sdk_physical(
        self, package: DeploymentPackage, options: Mapping[str, Any]
    ) -> ProviderTaskHandle:
        response, physical_qasm, measurements, backend_snapshot, calibration_id = (
            self._submit_sdk_mapped_qasm(
                package.qasm,
                backend=package.backend.name,
                shots=package.shots,
                name=package.name,
                target_qubits=options["target_qubits"],
            )
        )
        target_qubits = tuple(int(qubit) for qubit in options["target_qubits"])
        receipt = build_submission_receipt(package, response)
        receipt.update(
            {
                "quafu_protocol": _SDK_PROTOCOL,
                "service_compiler": "none",
                "target_qubits": list(target_qubits),
                "source_qasm_sha256": hashlib.sha256(package.qasm.encode()).hexdigest(),
                "submitted_qasm_sha256": hashlib.sha256(
                    physical_qasm.encode()
                ).hexdigest(),
                "backend_snapshot_sha256": backend_snapshot_sha256(backend_snapshot),
                "calibration_id": calibration_id,
                "measurement_mapping": [dict(item) for item in measurements],
            }
        )
        return ProviderTaskHandle(
            "quafu", str(response["job_id"]), package.backend.name, receipt
        )

    def _submit_sdk_mapped_qasm(
        self,
        qasm: str,
        *,
        backend: str,
        shots: int,
        name: str,
        target_qubits: Sequence[int],
    ) -> tuple[
        Mapping[str, Any],
        str,
        tuple[dict[str, int], ...],
        Mapping[str, Any],
        str,
    ]:
        """Validate, materialize, and submit one physical SDK program."""

        adapter = self._submission_sdk_adapter()
        backend_snapshot = adapter.backend_metadata(backend)
        calibration_id = backend_snapshot.get("calibration_id")
        if not isinstance(calibration_id, str) or not calibration_id.strip():
            raise RuntimeError(
                "Quafu SDK backend metadata has no current calibration identity"
            )
        mapping = tuple(int(qubit) for qubit in target_qubits)
        physical_qasm, measurements = prepare_physical_qasm(
            qasm,
            target_qubits=mapping,
            backend_metadata=backend_snapshot,
        )
        response = adapter.submit_physical(
            physical_qasm,
            target=backend,
            shots=shots,
            name=name,
        )
        return (
            response,
            physical_qasm,
            measurements,
            backend_snapshot,
            calibration_id.strip(),
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

        ``q[i]`` remains logical qubit ``i``; ``target_qubits[i]`` identifies the
        physical qubit selected for that qubit. Quafu receives ``compiler=None``.
        """

        program = str(qasm)
        backend = str(chip).strip()
        task_name = str(name).strip()
        if not program.lstrip().startswith("OPENQASM 2.0;"):
            raise ValueError("Quafu QASM submission requires OpenQASM 2.0")
        if not backend or not task_name:
            raise ValueError("Quafu QASM submission requires chip and name")
        _validate_quafu_shots(backend, shots)
        options = _submission_options(
            program,
            n_wires=len(target_qubits),
            options={"compiler": None, "target_qubits": target_qubits},
        )
        digest = hashlib.sha256(program.encode()).hexdigest()
        response, physical_qasm, measurements, backend_snapshot, calibration_id = (
            self._submit_sdk_mapped_qasm(
                program,
                backend=backend,
                shots=shots,
                name=task_name,
                target_qubits=options["target_qubits"],
            )
        )
        receipt = {
            "deployment_receipt_schema": DEPLOYMENT_SUBMISSION_RECEIPT_SCHEMA,
            "deployment_package_schema": "flagquantum_submitted_qasm_v1",
            "deployment_program_format": "openqasm-2",
            "routing_evidence_sha256": digest,
            "deployment_artifact_sha256": digest,
            "source_qasm_sha256": digest,
            "submitted_qasm_sha256": hashlib.sha256(physical_qasm.encode()).hexdigest(),
            "compiler": None,
            "service_compiler": "none",
            "target_qubits": options["target_qubits"],
            "quafu_protocol": _SDK_PROTOCOL,
            "backend_snapshot_sha256": backend_snapshot_sha256(backend_snapshot),
            "calibration_id": calibration_id,
            "measurement_mapping": [dict(item) for item in measurements],
        }
        return ProviderTaskHandle("quafu", str(response["job_id"]), backend, receipt)

    def query_status(self, handle: ProviderTaskHandle) -> str:
        self._protocol(handle)
        return self._submission_sdk_adapter().job_status(handle.task_id)

    def cancel(self, handle: ProviderTaskHandle) -> Any:
        """Request cancellation through the API that accepted the task."""

        self._protocol(handle)
        return self._submission_sdk_adapter().cancel_job(handle.task_id)

    def fetch_result(self, handle: ProviderTaskHandle) -> DeploymentResult:
        protocol = self._protocol(handle)
        response: Mapping[str, Any] = self._submission_sdk_adapter().job_result(
            handle.task_id
        )
        if not response:
            raise RuntimeError(
                f"quafu task {handle.task_id} has no result yet; query status and retry"
            )
        counts = _extract_counts(
            response, provider="quafu", flip=self.reverse_result_bits
        )
        if response.get("bit_order") != "c0_rightmost":
            raise RuntimeError(
                "Quafu SDK result does not declare c0_rightmost bit order"
            )
        returned = response.get("shots_returned")
        if type(returned) is not int or returned != sum(counts.values()):
            raise RuntimeError(
                "Quafu SDK returned-shot count does not match result counts"
            )
        if handle.payload.get("service_compiler") == "none":
            validate_physical_result(
                expected_measurements=handle.payload.get("measurement_mapping"),
                expected_calibration_id=handle.payload.get("calibration_id"),
                response=response,
            )
        executed = response.get("executed_qasm")
        if isinstance(executed, str) and executed:
            response = dict(response) | {
                "executed_qasm_sha256": hashlib.sha256(executed.encode()).hexdigest()
            }
        return DeploymentResult(
            handle=handle,
            counts=counts,
            shots=sum(counts.values()),
            metadata=build_result_metadata(
                handle,
                dict(response)
                | {
                    "quafu_protocol": protocol,
                    "service_compiler": handle.payload.get("service_compiler"),
                    "source_qasm_sha256": handle.payload.get("source_qasm_sha256"),
                    "submitted_qasm_sha256": handle.payload.get(
                        "submitted_qasm_sha256"
                    ),
                },
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

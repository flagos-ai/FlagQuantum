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
_TASK_API_PROTOCOL = "task_api_v1"
_SQC_PROTOCOL = "sqc_legacy"
_SDK_PROTOCOL = "quafu_sdk_v1"
_TASK_API_ROUTING_TARGETS = {"all-race", "all-redispatch"}
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


def _requires_task_api(target: str) -> bool:
    backend = _quafu_backend_name(target)
    return _is_simulator_target(backend) or backend in _TASK_API_ROUTING_TARGETS


def _validate_quafu_shots(target: str, shots: int | None) -> None:
    """Validate the task API's simulator and hardware shot ranges."""

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
    """FlagQuantum adapter for Quafu discovery and remote execution.

    Read-only discovery, calibration, and static single-circuit task lifecycles
    use the official Quafu SDK. An explicitly injected HTTP transport retains
    the compatibility path used by offline contract tests while collection
    migration remains staged.
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
        transport_supplied = "transport" in kwargs
        sdk_adapter = kwargs.pop("_sdk_adapter", None)
        if sdk_adapter is not None and not isinstance(sdk_adapter, _QuafuSdkAdapter):
            raise TypeError("_sdk_adapter must be a private Quafu SDK adapter")
        self._sdk_read_adapter = sdk_adapter
        self._sdk_submission_adapter = sdk_adapter
        self._prefer_sdk_reads = sdk_adapter is not None or not transport_supplied
        self._prefer_sdk_lifecycle = sdk_adapter is not None or not transport_supplied
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
        protocol = handle.payload.get("quafu_protocol", _SQC_PROTOCOL)
        if protocol in {_SDK_PROTOCOL, _TASK_API_PROTOCOL}:
            return str(protocol)
        return _SQC_PROTOCOL

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

    def _require_legacy_credential(self, target: str) -> None:
        """Refuse an unauthenticated POST to the legacy SQC platform.

        The legacy SQC contract authenticates with the ``token`` header that
        :meth:`_headers` builds from ``QUAFU_API_TOKEN``. Without it the request
        still leaves the machine, and the platform's transport-level rejection is
        what reaches the caller -- a report of the platform's verdict, not of the
        missing prerequisite. Both entry points that post to that endpoint call
        this before assembling a request, so an unconfigured credential fails at
        the earliest knowable stage instead of surfacing as a rejected
        submission.
        """

        if self.credentials.token:
            return
        remedy = (
            "A Quafu credential is required to submit to the legacy SQC platform: "
            "pass token=... or set QUAFU_API_TOKEN."
        )
        if self.api_key:
            raise ValidationError(
                f"{remedy} QUAFU_API_KEY is set, but the task API is not answering "
                f"for target {target!r}, and that key does not authorize the legacy "
                "endpoint. No request was sent."
            )
        raise ValidationError(
            f"{remedy} Neither QUAFU_API_KEY nor QUAFU_API_TOKEN is configured, so a "
            f"request for target {target!r} would carry no authentication header. "
            "No request was sent."
        )

    def list_devices(
        self, n_qubits: int | None = None
    ) -> tuple[CloudBackendProfile, ...]:
        """Discover devices through the official Quafu SDK.

        SDK data is normalized into FlagQuantum-owned backend profiles. During
        the staged migration, only an explicitly injected HTTP transport uses
        the compatibility path retained for offline contract tests.
        """

        if self._prefer_sdk_reads:
            return self._sdk_backends(self._read_sdk_adapter())
        try:
            response = self.transport.get_json(
                self._task_api_url("/devices"), {}, self.timeout
            )
            return self._task_api_backends(response, n_qubits=n_qubits)
        except Exception:
            return self._legacy_backends(n_qubits=n_qubits)

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

    def _task_api_backends(
        self, response: Mapping[str, Any], *, n_qubits: int | None
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
                capacity = n_qubits or self.default_n_qubits
            metadata_fields = (
                "status",
                "queue",
                "simulator_available",
                "calibration_id",
                "calibrated_at",
                "calibration_source",
                "data_fetched_at",
                "data_origin",
                "live_status_at",
                "live_status_stale",
                "usable",
            )
            profiles.append(
                CloudBackendProfile(
                    provider="quafu",
                    name=name,
                    n_qubits=capacity,
                    basis_gates=tuple(map(str, row.get("basis_gates", ()))),
                    is_simulator=name == "sim",
                    metadata={
                        "source": "quafu-task-api-devices",
                        **{key: row[key] for key in metadata_fields if key in row},
                    },
                )
            )
        if not profiles:
            raise RuntimeError("quafu task API devices response contains no devices")
        return tuple(profiles)

    def _legacy_backends(
        self, *, n_qubits: int | None
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
                    n_qubits=n_qubits or self.default_n_qubits,
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
        if self._prefer_sdk_reads:
            sdk_response = self._read_sdk_adapter().calibration(
                name, calibration_id=calibration_id
            )
            return self._validate_sdk_calibration(
                sdk_response,
                device=name,
                calibration_id=calibration_id,
            )
        expected_identity = None
        if calibration_id is None:
            path = "/devices/{device}/calibration"
            values = {"device": parse.quote(name, safe="")}
        else:
            expected_identity = str(calibration_id).strip()
            if not expected_identity:
                raise ValueError("Quafu calibration_id cannot be empty")
            path = "/calibrations/{calibration_id}"
            values = {
                "calibration_id": parse.quote(expected_identity, safe=""),
            }
        response = self.transport.get_json(
            self._task_api_url(path, **values),
            {},
            self.timeout,
        )
        if not isinstance(response, Mapping):
            raise RuntimeError("quafu calibration response is not a mapping")
        required = {
            "id",
            "chip",
            "calibrated_at",
            "qubits",
            "couplers",
        }
        missing = sorted(required.difference(response))
        if missing:
            raise RuntimeError(
                "quafu calibration response is missing " + ", ".join(missing)
            )
        if response["chip"] != name:
            raise RuntimeError(
                f"quafu calibration belongs to {response['chip']!r}, not {name!r}"
            )
        if expected_identity is not None and response["id"] != expected_identity:
            raise RuntimeError(
                "quafu historical calibration identity does not match the request"
            )
        return response

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

        if not self._prefer_sdk_reads:
            return self.fetch_chip_info(chip)
        return qsteed_chip_info(self.fetch_calibration(chip))

    def submit(self, package: DeploymentPackage) -> ProviderTaskHandle:
        validate_deployment_package(package)
        _validate_quafu_shots(package.backend.name, package.shots)
        options = _submission_options(
            package.qasm,
            n_wires=package.n_qubits,
            options=dict(package.metadata.get("provider_options", {})),
        )
        if (
            self._prefer_sdk_lifecycle
            and options.get("compiler") == "quarkcircuit"
            and not options.get("target_qubits")
        ):
            return self._submit_sdk_logical(package)
        if self._prefer_sdk_lifecycle and options.get("compiler") is None:
            return self._submit_sdk_physical(package, options)
        task_api_required = _requires_task_api(package.backend.name)
        if task_api_required and (
            options.get("compiler") != "quarkcircuit" or options.get("target_qubits")
        ):
            # The exception class here is unchanged: this refusal is about the
            # option combination the caller passed, not about a credential, and
            # `FQ-QUAFU-CREDENTIAL-FAIL-CLOSED-20260930.md` enumerates only the
            # hardware and the routing-target classes. Widening an unapproved API
            # change is not this patch's business.
            raise ValueError(
                f"Quafu target {package.backend.name!r} requires logical-circuit "
                "submission through the task API"
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
                    if task_api_required:
                        raise
                else:
                    raise
        if task_api_required:
            raise ValidationError(
                f"Quafu target {package.backend.name!r} requires the task API, and "
                "the task API needs a credential this provider does not have: pass "
                "api_key=... or set QUAFU_API_KEY. The legacy SQC platform is not an "
                "equivalent route to this target, so there is no fallback to take."
            )
        return self._submit_legacy(package, options)

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
        adapter = self._submission_sdk_adapter()
        backend_snapshot = adapter.backend_metadata(package.backend.name)
        calibration_id = backend_snapshot.get("calibration_id")
        if not isinstance(calibration_id, str) or not calibration_id.strip():
            raise RuntimeError(
                "Quafu SDK backend metadata has no current calibration identity"
            )
        target_qubits = tuple(int(qubit) for qubit in options["target_qubits"])
        physical_qasm, measurements = prepare_physical_qasm(
            package.qasm,
            target_qubits=target_qubits,
            backend_metadata=backend_snapshot,
        )
        response = adapter.submit_physical(
            physical_qasm,
            target=package.backend.name,
            shots=package.shots,
            name=package.name,
        )
        receipt = build_submission_receipt(package, response)
        receipt.update(
            {
                "quafu_protocol": _SDK_PROTOCOL,
                "service_compiler": "none",
                "target_qubits": list(target_qubits),
                "submitted_qasm_sha256": hashlib.sha256(
                    physical_qasm.encode()
                ).hexdigest(),
                "backend_snapshot_sha256": backend_snapshot_sha256(backend_snapshot),
                "calibration_id": calibration_id.strip(),
                "measurement_mapping": [dict(item) for item in measurements],
            }
        )
        return ProviderTaskHandle(
            "quafu", str(response["job_id"]), package.backend.name, receipt
        )

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
        self._require_legacy_credential(package.backend.name)
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
        self._require_legacy_credential(backend)
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
        protocol = self._protocol(handle)
        if protocol == _SDK_PROTOCOL:
            return self._submission_sdk_adapter().job_status(handle.task_id)
        if protocol == _TASK_API_PROTOCOL:
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

        protocol = self._protocol(handle)
        if protocol == _SDK_PROTOCOL:
            return self._submission_sdk_adapter().cancel_job(handle.task_id)
        if protocol == _TASK_API_PROTOCOL:
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
        response: Mapping[str, Any]
        if protocol == _SDK_PROTOCOL:
            response = self._submission_sdk_adapter().job_result(handle.task_id)
        elif protocol == _TASK_API_PROTOCOL:
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
        if protocol == _SDK_PROTOCOL:
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
                    "executed_qasm_sha256": hashlib.sha256(
                        executed.encode()
                    ).hexdigest()
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

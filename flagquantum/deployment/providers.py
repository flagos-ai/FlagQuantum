"""Quantum-cloud provider adapters.

The adapters here intentionally depend only on the Python standard library.
They keep platform-specific endpoint, authentication, and payload details
behind the FlagQuantum deployment contract.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, cast
from urllib import parse, request

from .cloud import (
    CloudBackendProfile,
    DeploymentPackage,
    DeploymentResult,
    ProviderTaskHandle,
    QuantumProvider,
    build_result_metadata,
    build_submission_receipt,
    validate_deployment_package,
    validate_deployment_result,
)


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


@dataclass(frozen=True)
class BraketSubmissionPreview:
    """Validated, non-submitting description of an Amazon Braket task."""

    device_arn: str
    backend: CloudBackendProfile
    shots: int
    program: str
    compatible: bool
    blockers: tuple[str, ...] = ()

    def summary(self) -> dict[str, Any]:
        return {
            "device_arn": self.device_arn,
            "backend": self.backend.name,
            "shots": self.shots,
            "program_format": "openqasm-3",
            "dynamic_dialect": self.backend.dynamic_dialect,
            "compatible": self.compatible,
            "blockers": self.blockers,
        }


def _normalize_counts(counts: Mapping[str, Any]) -> dict[str, int]:
    return {str(key): int(round(float(value))) for key, value in counts.items()}


def _flip_counts(counts: Mapping[str, Any]) -> dict[str, int]:
    return {str(key)[::-1]: int(round(float(value))) for key, value in counts.items()}


def _unwrap_result_envelope(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    result = payload.get("result", payload)
    if (
        isinstance(result, Mapping)
        and "ok" in result
        and isinstance(result.get("result"), Mapping)
    ):
        return cast(Mapping[str, Any], result["result"])
    if isinstance(result, Mapping):
        return result
    return payload


def _extract_counts(
    payload: Mapping[str, Any], *, provider: str, flip: bool = False
) -> dict[str, int]:
    current: Any = _unwrap_result_envelope(payload)
    if isinstance(current, Mapping) and "counts" in current:
        current = current["counts"]
    elif isinstance(current, Mapping) and "count" in current:
        current = current["count"]
    elif isinstance(current, Mapping) and isinstance(current.get("data"), Mapping):
        current = current["data"]
        if "counts" in current:
            current = current["counts"]
        elif "count" in current:
            current = current["count"]
    if not isinstance(current, Mapping):
        raise RuntimeError(f"{provider} result response does not contain counts.")
    return _flip_counts(current) if flip else _normalize_counts(current)


def _strip_barrier(qasm: str) -> str:
    return "\n".join(
        line for line in qasm.splitlines() if not line.strip().startswith("barrier")
    )


def _format_circuit_source(source: str) -> str:
    return "\n".join(line.strip() for line in str(source).splitlines() if line.strip())


def _property_value(value: Any, *path: str, default: Any = None) -> Any:
    current = value
    for key in path:
        if isinstance(current, Mapping):
            current = current.get(key, default)
        else:
            current = getattr(current, key, default)
        if current is default:
            break
    return current


def _first_property(value: Any, paths: tuple[tuple[str, ...], ...]) -> Any:
    marker = object()
    for path in paths:
        found = _property_value(value, *path, default=marker)
        if found is not marker and found is not None:
            return found
    return None


def _braket_coupling_map(properties: Any, n_wires: int) -> Any:
    from ..compilation.routing import CouplingMap

    graph = _first_property(
        properties,
        (
            ("paradigm", "connectivity", "connectivityGraph"),
            ("paradigm", "connectivity", "connectivity_graph"),
        ),
    )
    edges: set[tuple[int, int]] = set()
    if isinstance(graph, Mapping):
        for source, targets in graph.items():
            for target in targets or ():
                left, right = sorted((int(source), int(target)))
                if left != right:
                    edges.add((left, right))
    if not edges:
        return None
    return CouplingMap(n_wires=n_wires, edges=tuple(sorted(edges)))


def _braket_dynamic_groups(properties: Any) -> tuple[tuple[int, ...], ...]:
    raw = _first_property(
        properties,
        (
            ("paradigm", "dynamicQubitGroups"),
            ("paradigm", "dynamic_qubit_groups"),
            ("provider", "dynamicQubitGroups"),
            ("provider", "dynamic_qubit_groups"),
        ),
    )
    if not raw:
        return ()
    return tuple(tuple(int(wire) for wire in group) for group in raw)


def braket_backend_profile(
    device: Any,
    *,
    dynamic_qubit_groups: tuple[tuple[int, ...], ...] | None = None,
) -> CloudBackendProfile:
    """Build a fail-closed backend profile from an ``AwsDevice``-like object."""

    properties = getattr(device, "properties", None)
    if properties is None:
        raise ValueError("Braket device does not expose properties")
    n_wires = _first_property(
        properties,
        (
            ("paradigm", "qubitCount"),
            ("paradigm", "qubit_count"),
            ("service", "qubitCount"),
        ),
    )
    if n_wires is None:
        raise ValueError("Braket device properties do not declare a qubit count")
    n_wires = int(n_wires)
    basis = _first_property(
        properties,
        (
            ("paradigm", "nativeGateSet"),
            ("paradigm", "native_gate_set"),
        ),
    ) or ()
    arn = str(getattr(device, "arn", getattr(device, "device_arn", "")))
    name = str(getattr(device, "name", arn or "braket-device"))
    provider_name = str(
        _first_property(properties, (("provider", "providerName"),))
        or _first_property(properties, (("provider", "provider_name"),))
        or ""
    )
    is_iqm = "iqm" in f"{provider_name} {name} {arn}".lower()
    groups = (
        tuple(tuple(int(wire) for wire in group) for group in dynamic_qubit_groups)
        if dynamic_qubit_groups is not None
        else _braket_dynamic_groups(properties)
    )
    metadata = {
        "device_arn": arn,
        "device_provider": provider_name,
        "capability_source": "AwsDevice.properties",
    }
    if groups:
        metadata["dynamic_qubit_groups"] = groups
    return CloudBackendProfile(
        provider="amazon-braket",
        name=name,
        n_wires=n_wires,
        basis_gates=tuple(str(gate).lower() for gate in basis),
        coupling_map=_braket_coupling_map(properties, n_wires),
        supports_openqasm=True,
        supports_dynamic_circuits=is_iqm,
        max_classical_bits=None,
        is_simulator="simulator" in f"{name} {arn}".lower(),
        metadata=metadata,
        dynamic_dialect="braket_iqm" if is_iqm else None,
    )


def _extract_cqlib_counts(items: Any) -> dict[str, int]:
    result_items = [items] if isinstance(items, Mapping) else list(items or [])
    merged: dict[str, int] = {}
    for item in result_items:
        if not isinstance(item, Mapping):
            continue
        matrix = item.get("resultStatus")
        if isinstance(matrix, list):
            for row in matrix[1:]:
                if not isinstance(row, list):
                    continue
                try:
                    bits = [int(value) for value in row]
                except Exception:
                    continue
                if bits and all(value in (0, 1) for value in bits):
                    bitstring = "".join(str(value) for value in bits)
                    merged[bitstring] = merged.get(bitstring, 0) + 1
        count = item.get("count")
        if isinstance(count, Mapping):
            for bit, value in count.items():
                key = str(bit)
                merged[key] = merged.get(key, 0) + int(round(float(value)))
    if not merged:
        raise RuntimeError("failed to extract counts from platform result payload")
    return merged


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


class QuafuProvider(HttpQuantumProvider):
    """Adapter for Quafu SQC cloud."""

    def __init__(
        self,
        *,
        base_url: str = "https://quafu-sqc.baqis.ac.cn",
        token: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            provider="quafu",
            base_url=base_url,
            credentials=ProviderCredentials(token=token),
            default_backend="quafu",
            **kwargs,
        )

    def _headers(self) -> dict[str, str]:
        headers = dict(self.credentials.extra_headers)
        if self.credentials.token:
            headers["token"] = self.credentials.token
        return headers

    def submit(self, package: DeploymentPackage) -> ProviderTaskHandle:
        validate_deployment_package(package)
        query = parse.urlencode(
            {"name": package.name, "chip": package.backend.name, "shots": package.shots}
        )
        response = self.transport.post_json(
            self._url(f"/task/run/?{query}"),
            {
                "circuit": package.qasm,
                "compile": bool(package.metadata.get("provider_compile", False)),
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

    def query_status(self, handle: ProviderTaskHandle) -> str:
        response = self.transport.get_json(
            self._url("/task/status/{task_id}", task_id=handle.task_id),
            self._headers(),
            self.timeout,
        )
        status = str(
            response.get(
                "status", response.get("state", response.get("task_status", "Running"))
            )
        )
        return {
            "completed": "Finished",
            "finished": "Finished",
            "failed": "Failed",
        }.get(status.lower(), status)

    def fetch_result(self, handle: ProviderTaskHandle) -> DeploymentResult:
        response = self.transport.get_json(
            self._url("/task/result/{task_id}", task_id=handle.task_id),
            self._headers(),
            self.timeout,
        )
        counts = _extract_counts(response, provider="quafu", flip=True)
        return DeploymentResult(
            handle=handle,
            counts=counts,
            shots=sum(counts.values()),
            metadata=build_result_metadata(handle, response),
        )


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


@dataclass(frozen=True)
class _CqlibSpec:
    provider: str
    domain: str
    login_path: str
    machine_list_path: str
    submit_path: str
    query_path: str
    default_backend: str


class _CqlibProvider(QuantumProvider):
    def __init__(
        self,
        *,
        spec: _CqlibSpec,
        token: str | None = None,
        transport: QuantumCloudTransport | None = None,
        timeout: float = 60.0,
        default_n_wires: int = 32,
        access_token: str | None = None,
    ) -> None:
        self.provider = spec.provider
        self.spec = spec
        self.token = token
        self.access_token = access_token
        self.transport = transport or UrllibTransport()
        self.timeout = float(timeout)
        self.default_n_wires = int(default_n_wires)

    @property
    def base_url(self) -> str:
        return f"https://{self.spec.domain}"

    def _url(self, path: str) -> str:
        return self.base_url + path

    def _login(self) -> str:
        if self.access_token:
            return self.access_token
        if not self.token:
            raise ValueError(f"{self.provider} token cannot be empty.")
        payload = {
            "grant_type": "openId",
            "openId": self.token,
            "account_type": "member",
        }
        response = self.transport.post_form(
            self._url(self.spec.login_path), payload, {}, self.timeout
        )
        data = response.get("data", {})
        if (
            response.get("code", 0) != 0
            or not isinstance(data, Mapping)
            or not data.get("access_token")
        ):
            raise RuntimeError(f"{self.provider} login failed.")
        self.access_token = str(data["access_token"])
        return self.access_token

    def _headers(self) -> dict[str, str]:
        access_token = self._login()
        return {"basicToken": access_token, "Authorization": f"Bearer {access_token}"}

    def discover_backends(
        self, n_wires: int | None = None
    ) -> tuple[CloudBackendProfile, ...]:
        try:
            response = self.transport.get_json(
                self._url(self.spec.machine_list_path), self._headers(), self.timeout
            )
            rows = response.get("data", [])
        except Exception:
            rows = []
        profiles = []
        for row in rows if isinstance(rows, list) else []:
            if not isinstance(row, Mapping):
                continue
            name = str(
                row.get(
                    "machineName",
                    row.get("qcCode", row.get("name", self.spec.default_backend)),
                )
            )
            qubits = int(
                row.get("n_wires", row.get("qubits", n_wires or self.default_n_wires))
            )
            if n_wires is None or qubits >= n_wires:
                profiles.append(
                    CloudBackendProfile(
                        provider=self.provider,
                        name=name,
                        n_wires=qubits,
                        supports_openqasm=False,
                        supports_qcis=True,
                        metadata=dict(row),
                    )
                )
        return tuple(profiles) or (
            CloudBackendProfile(
                provider=self.provider,
                name=self.spec.default_backend,
                n_wires=n_wires or self.default_n_wires,
                supports_openqasm=False,
                supports_qcis=True,
                metadata={"source": "fallback"},
            ),
        )

    def submit(self, package: DeploymentPackage) -> ProviderTaskHandle:
        validate_deployment_package(package)
        qcis = package.metadata.get("qcis")
        if not isinstance(qcis, str) or not qcis.strip():
            raise ValueError(
                f"{self.provider} requires package.metadata['qcis'] because this platform is QCIS-native."
            )
        payload = {
            "exp_id": package.metadata.get("experiment_id"),
            "lab_id": package.metadata.get("lab_id"),
            "inputCode": [_format_circuit_source(qcis).upper()],
            "languageCode": "qcis",
            "name": package.name,
            "shots": package.shots,
            "source": "SDK",
            "computerCode": package.backend.name,
            "experimentDetailName": "1",
            "is_verify": bool(package.metadata.get("is_verify", True)),
        }
        response = self.transport.post_json(
            self._url(self.spec.submit_path), payload, self._headers(), self.timeout
        )
        data = response.get("data", {})
        query_ids = data.get("query_ids") if isinstance(data, Mapping) else None
        if isinstance(query_ids, list) and query_ids:
            task_id = ",".join(str(value) for value in query_ids)
        elif query_ids:
            task_id = str(query_ids)
        else:
            raise RuntimeError(
                f"{self.provider} submit response does not contain query_ids."
            )
        return ProviderTaskHandle(
            self.provider,
            task_id,
            package.backend.name,
            build_submission_receipt(
                package, {"query_ids": query_ids, "submit": dict(response)}
            ),
        )

    def _query_result_items(
        self, handle: ProviderTaskHandle
    ) -> list[Mapping[str, Any]]:
        query_ids = handle.payload.get("query_ids", handle.task_id.split(","))
        response = self.transport.post_json(
            self._url(self.spec.query_path),
            {"query_ids": query_ids},
            self._headers(),
            self.timeout,
        )
        data = response.get("data", {})
        items = (
            data.get("experimentResultModelList") if isinstance(data, Mapping) else None
        )
        return (
            [item for item in items if isinstance(item, Mapping)]
            if isinstance(items, list)
            else []
        )

    def query_status(self, handle: ProviderTaskHandle) -> str:
        return "Finished" if self._query_result_items(handle) else "Running"

    def fetch_result(self, handle: ProviderTaskHandle) -> DeploymentResult:
        items = self._query_result_items(handle)
        counts = _extract_cqlib_counts(items)
        return DeploymentResult(
            handle=handle,
            counts=counts,
            shots=sum(counts.values()),
            metadata=build_result_metadata(handle, {"result_items": items}),
        )


class TianyanProvider(_CqlibProvider):
    def __init__(self, *, token: str | None = None, **kwargs: Any) -> None:
        super().__init__(
            spec=_CqlibSpec(
                provider="tianyan",
                domain="qc.zdxlz.com",
                login_path="/qccp-auth/oauth2/sdk/opnId",
                machine_list_path="/qccp-quantum/sdk/quantumComputer/list",
                submit_path="/qccp-quantum/sdk/experiment/temporary/save",
                query_path="/qccp-quantum/sdk/experiment/result/find",
                default_backend="tianyan176",
            ),
            token=token,
            **kwargs,
        )


class GuodunProvider(_CqlibProvider):
    def __init__(self, *, token: str | None = None, **kwargs: Any) -> None:
        super().__init__(
            spec=_CqlibSpec(
                provider="guodun",
                domain="quantumctek-cloud.com",
                login_path="/api-uaa/oauth/token",
                machine_list_path="/experiment/sdk/quantumComputer/list",
                submit_path="/experiment/sdk/experiment/temporary/save",
                query_path="/experiment/sdk/experiment/result/find",
                default_backend="gd_qc1",
            ),
            token=token,
            **kwargs,
        )


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


class AmazonBraketProvider(QuantumProvider):
    """Amazon Braket OpenQASM adapter with an explicit no-submit preflight."""

    provider = "amazon-braket"

    def __init__(
        self,
        device: Any,
        *,
        dynamic_qubit_groups: tuple[tuple[int, ...], ...] | None = None,
        program_factory: Any | None = None,
    ) -> None:
        if isinstance(device, str):
            try:
                from braket.aws import AwsDevice
            except ImportError as exc:
                raise ImportError(
                    "AmazonBraketProvider requires amazon-braket-sdk when a device "
                    "ARN is supplied"
                ) from exc
            device = AwsDevice(device)
        self.device = device
        self.backend = braket_backend_profile(
            device,
            dynamic_qubit_groups=dynamic_qubit_groups,
        )
        self._program_factory = program_factory

    @property
    def device_arn(self) -> str:
        return str(self.backend.metadata.get("device_arn", ""))

    def discover_backends(
        self, n_wires: int | None = None
    ) -> tuple[CloudBackendProfile, ...]:
        if n_wires is not None and self.backend.n_wires < int(n_wires):
            return ()
        return (self.backend,)

    def dry_run(self, package: DeploymentPackage) -> BraketSubmissionPreview:
        """Validate locally and return the exact payload without creating a task."""

        blockers: list[str] = []
        try:
            validate_deployment_package(package)
        except Exception as exc:
            blockers.append(f"invalid_deployment_package:{exc}")
        if package.backend.provider != self.provider:
            blockers.append("backend_provider_is_not_amazon_braket")
        package_arn = str(package.backend.metadata.get("device_arn", ""))
        if self.device_arn and package_arn != self.device_arn:
            blockers.append("deployment_package_targets_a_different_device")
        if package.qasm_version != 3.0:
            blockers.append("amazon_braket_provider_requires_openqasm_3")
        if (
            package.backend.dynamic_dialect == "braket_iqm"
            and not package.backend.metadata.get("dynamic_qubit_groups")
        ):
            blockers.append("braket_iqm_dynamic_qubit_groups_are_required")
        return BraketSubmissionPreview(
            device_arn=self.device_arn,
            backend=package.backend,
            shots=package.shots,
            program=package.qasm,
            compatible=not blockers,
            blockers=tuple(blockers),
        )

    def _program(self, source: str) -> Any:
        if self._program_factory is not None:
            return self._program_factory(source=source)
        try:
            from braket.ir.openqasm import Program
        except ImportError as exc:
            raise ImportError(
                "Amazon Braket submission requires amazon-braket-sdk"
            ) from exc
        return Program(source=source)

    def submit(self, package: DeploymentPackage) -> ProviderTaskHandle:
        preview = self.dry_run(package)
        if not preview.compatible:
            raise RuntimeError(
                "Amazon Braket preflight failed: " + ", ".join(preview.blockers)
            )
        task = self.device.run(self._program(preview.program), shots=preview.shots)
        task_id = getattr(task, "id", None)
        if callable(task_id):
            task_id = task_id()
        if task_id is None:
            raise RuntimeError("Amazon Braket task does not expose an id")
        return ProviderTaskHandle(
            provider=self.provider,
            task_id=str(task_id),
            backend_name=package.backend.name,
            payload=build_submission_receipt(package, {"task": task}),
        )

    def query_status(self, handle: ProviderTaskHandle) -> str:
        task = handle.payload.get("task")
        if task is None:
            raise RuntimeError("Amazon Braket task object is missing from the handle")
        state = str(task.state()).upper()
        return {
            "COMPLETED": "Finished",
            "FAILED": "Failed",
            "CANCELLED": "Failed",
            "CANCELLING": "Running",
            "CREATED": "Running",
            "QUEUED": "Running",
            "RUNNING": "Running",
        }.get(state, state.title())

    def fetch_result(self, handle: ProviderTaskHandle) -> DeploymentResult:
        task = handle.payload.get("task")
        if task is None:
            raise RuntimeError("Amazon Braket task object is missing from the handle")
        raw = task.result()
        counts = getattr(raw, "measurement_counts", None)
        if counts is None:
            raise RuntimeError("Amazon Braket result does not contain measurement counts")
        normalized = _normalize_counts(counts)
        return DeploymentResult(
            handle=handle,
            counts=normalized,
            shots=sum(normalized.values()),
            metadata=build_result_metadata(
                handle,
                {
                    "device_arn": self.device_arn,
                    "mid_circuit_measurements_returned": False
                    if self.backend.dynamic_dialect == "braket_iqm"
                    else None,
                },
            ),
        )

    def run(self, package: DeploymentPackage) -> DeploymentResult:
        """Submit and block through ``task.result()``, matching Braket semantics."""

        handle = self.submit(package)
        result = self.fetch_result(handle)
        return validate_deployment_result(result)


__all__ = [
    "AmazonBraketProvider",
    "BraketSubmissionPreview",
    "FieldQuantumProvider",
    "GuodunProvider",
    "HttpQuantumProvider",
    "OriginQProvider",
    "ProviderCredentials",
    "ProviderEndpoints",
    "QuafuProvider",
    "QuantumCloudTransport",
    "TencentQuantumProvider",
    "TianyanProvider",
    "UrllibTransport",
    "braket_backend_profile",
]

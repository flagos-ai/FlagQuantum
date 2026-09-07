"""Standard-library quantum-cloud adapters behind the deployment contract."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from ..providers.execution import result_parsing as _result_parsing
from ..providers.execution.braket import (
    AmazonBraketProvider,
    BraketSubmissionPreview,
    braket_backend_profile,
)
from ..providers.execution.http import (
    HttpQuantumProvider,
    ProviderCredentials,
    ProviderEndpoints,
    QuantumCloudTransport,
    UrllibTransport,
)
from ..providers.execution.originq import OriginQProvider
from ..providers.execution.quafu import QuafuProvider
from .cloud import (
    CloudBackendProfile,
    DeploymentPackage,
    DeploymentResult,
    ProviderTaskHandle,
    QuantumProvider,
    build_result_metadata,
    build_submission_receipt,
    validate_deployment_package,
)

_extract_counts = _result_parsing._extract_counts
_flip_counts = _result_parsing._flip_counts
_format_circuit_source = _result_parsing._format_circuit_source
_normalize_counts = _result_parsing._normalize_counts
_strip_barrier = _result_parsing._strip_barrier
_unwrap_result_envelope = _result_parsing._unwrap_result_envelope


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

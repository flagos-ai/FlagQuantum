"""Execution providers for the shared CQLib cloud protocol."""

from __future__ import annotations

from dataclasses import dataclass
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
from .http import QuantumCloudTransport, UrllibTransport
from .result_parsing import _format_circuit_source


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


__all__ = [
    "GuodunProvider",
    "TianyanProvider",
]

"""Private loading and credential boundary for the maintained Quafu SDK."""

from __future__ import annotations

import importlib
import os
from collections.abc import Mapping, Sequence
from importlib import metadata
from types import ModuleType
from typing import Any, NoReturn

from ...errors import ExecutionError, ValidationError

_DISTRIBUTION = "quafu-sdk"
_EXTRA = "quafu"


def _load_quafu_sdk() -> ModuleType:
    try:
        metadata.version(_DISTRIBUTION)
    except metadata.PackageNotFoundError:
        raise RuntimeError(
            "Quafu execution requires the optional quafu-sdk package; "
            "install FlagQuantum with the 'quafu' extra."
        ) from None

    try:
        sdk = importlib.import_module("quafu")
    except Exception as exc:
        if type(exc).__name__ == "ConflictError":
            raise RuntimeError(
                "quafu-sdk cannot share an environment with pyquafu because "
                "both install the 'quafu' package; use a clean environment."
            ) from None
        raise
    if not callable(getattr(sdk, "Client", None)):
        raise RuntimeError(
            "The installed 'quafu' package is not the supported quafu-sdk; "
            "use a clean environment and install FlagQuantum with the "
            f"'{_EXTRA}' extra."
        )
    return sdk


def _submission_api_key(explicit: str | None) -> str:
    value = explicit if explicit is not None else os.getenv("QUAFU_API_KEY")
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(
            "Quafu submission requires api_key=... or QUAFU_API_KEY; "
            "FlagQuantum does not start interactive SDK login."
        )
    return value.strip()


def _plain_value(value: Any, *, field: str) -> Any:
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise RuntimeError(f"Quafu SDK {field} contains a non-string key")
        return {
            key: _plain_value(item, field=f"{field}.{key}")
            for key, item in value.items()
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [
            _plain_value(item, field=f"{field}[{index}]")
            for index, item in enumerate(value)
        ]
    raise RuntimeError(
        f"Quafu SDK {field} contains unsupported {type(value).__name__} data"
    )


def _plain_mapping(value: Any, *, field: str) -> dict[str, Any]:
    raw = getattr(value, "raw", value)
    normalized = _plain_value(raw, field=field)
    if not isinstance(normalized, dict):
        raise RuntimeError(f"Quafu SDK {field} is not a mapping")
    return normalized


class _QuafuSdkAdapter:
    """Keep SDK clients and their native objects inside the Remote boundary."""

    def __init__(self, client: Any) -> None:
        self._client = client
        self._jobs: dict[str, Any] = {}

    @classmethod
    def for_discovery(
        cls,
        *,
        base_url: str | None = None,
        client: Any | None = None,
    ) -> _QuafuSdkAdapter:
        if client is not None:
            return cls(client)
        sdk = _load_quafu_sdk()
        return cls(sdk.Client(base_url=base_url))

    @classmethod
    def for_submission(
        cls,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        client: Any | None = None,
    ) -> _QuafuSdkAdapter:
        if client is not None:
            if api_key is not None or base_url is not None:
                raise TypeError(
                    "An injected Quafu SDK client cannot be combined with "
                    "api_key or base_url"
                )
            return cls(client)
        key = _submission_api_key(api_key)
        sdk = _load_quafu_sdk()
        return cls(sdk.Client(api_key=key, base_url=base_url))

    def devices(self) -> tuple[dict[str, Any], ...]:
        rows = self._client.devices()
        if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
            raise RuntimeError("Quafu SDK devices result is not a sequence")
        return tuple(
            _plain_mapping(row, field=f"devices[{index}]")
            for index, row in enumerate(rows)
        )

    def backend_metadata(self, target: str) -> dict[str, Any]:
        return _plain_mapping(
            self._client.metadata(target), field=f"metadata[{target}]"
        )

    def calibration(
        self,
        device: str,
        *,
        calibration_id: str | None = None,
    ) -> dict[str, Any]:
        value = (
            self._client.device_calibration(device)
            if calibration_id is None
            else self._client.calibration(calibration_id)
        )
        return _plain_mapping(value, field="calibration")

    def submit_logical(
        self,
        qasm: str,
        *,
        target: str,
        shots: int,
        name: str,
    ) -> dict[str, Any]:
        try:
            job = self._client.submit(
                qasm,
                target=target,
                shots=shots,
                compiler="qiskit",
                name=name,
            )
        except Exception as exc:
            _raise_sdk_failure(exc, operation="submission")
        job_id = getattr(job, "id", None)
        if not isinstance(job_id, str) or not job_id.strip():
            raise RuntimeError("Quafu SDK submission returned no job id")
        n_tasks = getattr(job, "n_tasks", None)
        if n_tasks is not None and n_tasks != 1:
            raise RuntimeError("Quafu SDK single submission returned multiple tasks")
        self._jobs[job_id] = job
        return {
            "job_id": job_id,
            "n_tasks": n_tasks,
            "target": target,
        }

    def job_status(self, job_id: str) -> str:
        job = self._job(job_id)
        try:
            status = job.status()
        except Exception as exc:
            _raise_sdk_failure(exc, operation="status")
        if not isinstance(status, str) or not status.strip():
            raise RuntimeError("Quafu SDK job returned no status")
        return status

    def cancel_job(self, job_id: str) -> dict[str, Any]:
        job = self._job(job_id)
        try:
            response = job.cancel()
        except Exception as exc:
            _raise_sdk_failure(exc, operation="cancellation")
        return _plain_mapping(response, field="cancel")

    def job_result(self, job_id: str) -> dict[str, Any]:
        job = self._job(job_id)
        try:
            result = job.result(timeout=0)
        except Exception as exc:
            _raise_sdk_failure(exc, operation="result")
        counts = _plain_value(getattr(result, "counts", None), field="result.counts")
        if not isinstance(counts, dict):
            raise RuntimeError("Quafu SDK result contains no counts")
        fields = (
            "status",
            "task_id",
            "target",
            "backend",
            "bit_order",
            "n_clbits",
            "shots_requested",
            "shots_returned",
            "measurements",
            "executed_qasm",
            "compiler",
            "calibration_id",
        )
        return {
            "counts": counts,
            **{
                field: _plain_value(
                    getattr(result, field, None), field=f"result.{field}"
                )
                for field in fields
            },
        }

    def _job(self, job_id: str) -> Any:
        if not isinstance(job_id, str) or not job_id.strip():
            raise ValueError("Quafu job id cannot be empty")
        cached = self._jobs.get(job_id)
        if cached is not None:
            return cached
        try:
            job = self._client.job(job_id)
        except Exception as exc:
            _raise_sdk_failure(exc, operation="restore")
        self._jobs[job_id] = job
        return job


def _raise_sdk_failure(exc: Exception, *, operation: str) -> NoReturn:
    name = type(exc).__name__
    code = getattr(exc, "code", None)
    category = getattr(exc, "category", None)
    detail = ", ".join(
        f"{key}={value}"
        for key, value in (("code", code), ("category", category))
        if isinstance(value, str) and value
    )
    message = f"Quafu SDK {operation} failed with {name}"
    if detail:
        message += f" ({detail})"
    if name in {
        "AuthError",
        "BadRequest",
        "CredentialsError",
        "LoginExpired",
        "LoginRevoked",
        "NotLoggedIn",
        "PrecheckError",
        "QuotaError",
    }:
        raise ValidationError(message) from None
    if name == "Timeout":
        raise TimeoutError(message) from None
    raise ExecutionError(message) from None


__all__: tuple[str, ...] = ()

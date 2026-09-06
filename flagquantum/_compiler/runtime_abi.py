"""Provider-neutral private runtime ABI for sealed executable artifacts."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from enum import Enum
from typing import Protocol, runtime_checkable

from .executable_artifact import SealedExecutableArtifact

_SHA256 = re.compile(r"[0-9a-f]{64}")


def _digest(value: dict[str, object]) -> str:
    encoded = json.dumps(
        value, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _require_digest(value: str, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


class AdapterKind(str, Enum):
    LOCAL_SYNC = "local_sync"
    OFFLINE_ASYNC_MOCK = "offline_async_mock"


class ExecutionState(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @property
    def terminal(self) -> bool:
        return self in {
            ExecutionState.SUCCEEDED,
            ExecutionState.FAILED,
            ExecutionState.CANCELLED,
        }


class RuntimeCallStatus(str, Enum):
    OK = "ok"
    INVALID_REQUEST = "invalid_request"
    UNKNOWN_HANDLE = "unknown_handle"
    NOT_READY = "not_ready"
    TERMINAL = "terminal"
    EXECUTION_FAILED = "execution_failed"


@dataclass(frozen=True, order=True)
class RuntimeParameterBinding:
    name: str
    value: float

    def __post_init__(self) -> None:
        name = str(self.name).strip()
        if not name:
            raise ValueError("runtime parameter name cannot be empty")
        if isinstance(self.value, bool) or not isinstance(self.value, (int, float)):
            raise ValueError("runtime parameter value must be a real scalar")
        value = float(self.value)
        if not math.isfinite(value):
            raise ValueError("runtime parameter value must be finite")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "value", value)

    def to_dict(self) -> dict[str, object]:
        return {"name": self.name, "value": self.value}


@dataclass(frozen=True)
class RuntimeBindings:
    shots: int | None = None
    parameter_bindings: tuple[RuntimeParameterBinding, ...] = ()

    def __post_init__(self) -> None:
        if self.shots is not None and (
            isinstance(self.shots, bool)
            or not isinstance(self.shots, int)
            or self.shots <= 0
        ):
            raise ValueError("runtime shots must be a positive integer")
        bindings = tuple(sorted(self.parameter_bindings))
        if any(not isinstance(item, RuntimeParameterBinding) for item in bindings):
            raise ValueError("runtime bindings must use RuntimeParameterBinding")
        names = tuple(item.name for item in bindings)
        if len(names) != len(set(names)):
            raise ValueError("runtime parameter bindings must be name-unique")
        object.__setattr__(self, "parameter_bindings", bindings)

    def to_dict(self) -> dict[str, object]:
        return {
            "shots": self.shots,
            "parameter_bindings": [item.to_dict() for item in self.parameter_bindings],
        }


@dataclass(frozen=True)
class ExecutionBinding:
    artifact_identity: str
    adapter_identity: str
    adapter_kind: AdapterKind
    options: RuntimeBindings
    binding_identity: str

    def __post_init__(self) -> None:
        _require_digest(self.artifact_identity, "artifact identity")
        _require_digest(self.adapter_identity, "adapter identity")
        _require_digest(self.binding_identity, "binding identity")
        if not isinstance(self.adapter_kind, AdapterKind):
            raise ValueError("adapter kind must use the closed enum")
        if not isinstance(self.options, RuntimeBindings):
            raise ValueError("runtime bindings must use RuntimeBindings")
        if self.binding_identity != _digest(self.identity_dict()):
            raise ValueError("execution binding identity does not match its content")

    def identity_dict(self) -> dict[str, object]:
        return {
            "artifact_identity": self.artifact_identity,
            "adapter_identity": self.adapter_identity,
            "adapter_kind": self.adapter_kind.value,
            "options": self.options.to_dict(),
        }


def create_execution_binding(
    artifact: SealedExecutableArtifact,
    adapter_identity: str,
    adapter_kind: AdapterKind,
    options: RuntimeBindings,
) -> ExecutionBinding:
    if not executable_artifact_envelope_is_valid(artifact):
        raise ValueError("execution binding requires a valid sealed artifact")
    if not isinstance(adapter_kind, AdapterKind):
        raise ValueError("adapter kind must use the closed enum")
    if not isinstance(options, RuntimeBindings):
        raise ValueError("runtime bindings must use RuntimeBindings")
    values = {
        "artifact_identity": artifact.artifact_identity,
        "adapter_identity": _require_digest(adapter_identity, "adapter identity"),
        "adapter_kind": adapter_kind.value,
        "options": options.to_dict(),
    }
    return ExecutionBinding(
        artifact.artifact_identity,
        adapter_identity,
        adapter_kind,
        options,
        _digest(values),
    )


@dataclass(frozen=True)
class ExecutionHandle:
    token: str
    artifact_identity: str
    binding_identity: str

    def __post_init__(self) -> None:
        _require_digest(self.token, "execution handle token")
        _require_digest(self.artifact_identity, "artifact identity")
        _require_digest(self.binding_identity, "binding identity")


@dataclass(frozen=True)
class ExternalExecutionIdentity:
    """Execution-only identity; never part of compiler or portable artifact state."""

    provider: str
    backend_id: str
    job_id: str

    def __post_init__(self) -> None:
        for name in ("provider", "backend_id", "job_id"):
            value = str(getattr(self, name)).strip()
            if not value:
                raise ValueError(f"external {name} cannot be empty")
            object.__setattr__(self, name, value)

    def to_dict(self) -> dict[str, str]:
        return {
            "provider": self.provider,
            "backend_id": self.backend_id,
            "job_id": self.job_id,
        }


@dataclass(frozen=True)
class SubmissionReceipt:
    handle: ExecutionHandle
    binding: ExecutionBinding
    external_identity: ExternalExecutionIdentity | None
    receipt_identity: str

    def __post_init__(self) -> None:
        if not isinstance(self.handle, ExecutionHandle):
            raise ValueError("receipt handle must use ExecutionHandle")
        if not isinstance(self.binding, ExecutionBinding):
            raise ValueError("receipt binding must use ExecutionBinding")
        if self.external_identity is not None and not isinstance(
            self.external_identity, ExternalExecutionIdentity
        ):
            raise ValueError("external identity has the wrong type")
        _require_digest(self.receipt_identity, "receipt identity")
        if self.handle.artifact_identity != self.binding.artifact_identity:
            raise ValueError("receipt artifact identity chain is broken")
        if self.handle.binding_identity != self.binding.binding_identity:
            raise ValueError("receipt binding identity chain is broken")
        if self.receipt_identity != _digest(self.identity_dict()):
            raise ValueError("receipt identity does not match its content")

    def identity_dict(self) -> dict[str, object]:
        return {
            "handle_token": self.handle.token,
            "artifact_identity": self.binding.artifact_identity,
            "binding_identity": self.binding.binding_identity,
            "external_identity": (
                None
                if self.external_identity is None
                else self.external_identity.to_dict()
            ),
        }


def create_submission_receipt(
    handle: ExecutionHandle,
    binding: ExecutionBinding,
    external_identity: ExternalExecutionIdentity | None = None,
) -> SubmissionReceipt:
    if not isinstance(handle, ExecutionHandle) or not isinstance(
        binding, ExecutionBinding
    ):
        raise ValueError("submission receipt requires a handle and binding")
    if external_identity is not None and not isinstance(
        external_identity, ExternalExecutionIdentity
    ):
        raise ValueError("external identity has the wrong type")
    values = {
        "handle_token": handle.token,
        "artifact_identity": binding.artifact_identity,
        "binding_identity": binding.binding_identity,
        "external_identity": (
            None if external_identity is None else external_identity.to_dict()
        ),
    }
    return SubmissionReceipt(handle, binding, external_identity, _digest(values))


@dataclass(frozen=True)
class RuntimeResult:
    receipt_identity: str
    artifact_identity: str
    binding_identity: str
    payload: bytes
    payload_content_hash: str
    result_identity: str

    def __post_init__(self) -> None:
        for name in (
            "receipt_identity",
            "artifact_identity",
            "binding_identity",
            "payload_content_hash",
            "result_identity",
        ):
            _require_digest(getattr(self, name), name.replace("_", " "))
        if not isinstance(self.payload, bytes):
            raise ValueError("runtime result payload must be immutable bytes")
        if self.payload_content_hash != hashlib.sha256(self.payload).hexdigest():
            raise ValueError("runtime result payload hash does not match its content")
        if self.result_identity != _digest(self.identity_dict()):
            raise ValueError("runtime result identity does not match its content")

    def identity_dict(self) -> dict[str, object]:
        return {
            "receipt_identity": self.receipt_identity,
            "artifact_identity": self.artifact_identity,
            "binding_identity": self.binding_identity,
            "payload_content_hash": self.payload_content_hash,
        }


def create_runtime_result(receipt: SubmissionReceipt, payload: bytes) -> RuntimeResult:
    if not isinstance(receipt, SubmissionReceipt):
        raise ValueError("runtime result requires a submission receipt")
    if not isinstance(payload, bytes):
        raise ValueError("runtime result payload must be immutable bytes")
    content_hash = hashlib.sha256(payload).hexdigest()
    values = {
        "receipt_identity": receipt.receipt_identity,
        "artifact_identity": receipt.binding.artifact_identity,
        "binding_identity": receipt.binding.binding_identity,
        "payload_content_hash": content_hash,
    }
    return RuntimeResult(
        receipt.receipt_identity,
        receipt.binding.artifact_identity,
        receipt.binding.binding_identity,
        payload,
        content_hash,
        _digest(values),
    )


@dataclass(frozen=True)
class RuntimeDiagnostic:
    status: RuntimeCallStatus
    message: str

    def __post_init__(self) -> None:
        if not isinstance(self.status, RuntimeCallStatus):
            raise ValueError("runtime diagnostic status must use the closed enum")
        message = str(self.message).strip()
        if not message:
            raise ValueError("runtime diagnostic message cannot be empty")
        object.__setattr__(self, "message", message)


@dataclass(frozen=True)
class SubmitResult:
    status: RuntimeCallStatus
    receipt: SubmissionReceipt | None = None
    diagnostics: tuple[RuntimeDiagnostic, ...] = ()

    def __post_init__(self) -> None:
        _validate_call_result(
            self.status,
            success_value=self.receipt,
            diagnostics=self.diagnostics,
            label="submit",
        )

    @property
    def ok(self) -> bool:
        return self.status is RuntimeCallStatus.OK


@dataclass(frozen=True)
class StatusResult:
    status: RuntimeCallStatus
    state: ExecutionState | None = None
    diagnostics: tuple[RuntimeDiagnostic, ...] = ()

    def __post_init__(self) -> None:
        _validate_call_result(
            self.status,
            success_value=self.state,
            diagnostics=self.diagnostics,
            label="status",
            allow_failure_value=True,
        )

    @property
    def ok(self) -> bool:
        return self.status is RuntimeCallStatus.OK


@dataclass(frozen=True)
class CancelResult:
    status: RuntimeCallStatus
    state: ExecutionState | None = None
    acknowledged: bool = False
    diagnostics: tuple[RuntimeDiagnostic, ...] = ()

    def __post_init__(self) -> None:
        _validate_call_result(
            self.status,
            success_value=self.state,
            diagnostics=self.diagnostics,
            label="cancel",
            allow_failure_value=True,
        )
        if not isinstance(self.acknowledged, bool):
            raise ValueError("cancel acknowledgement must be boolean")

    @property
    def ok(self) -> bool:
        return self.status is RuntimeCallStatus.OK


@dataclass(frozen=True)
class FetchResult:
    status: RuntimeCallStatus
    result: RuntimeResult | None = None
    diagnostics: tuple[RuntimeDiagnostic, ...] = ()

    def __post_init__(self) -> None:
        _validate_call_result(
            self.status,
            success_value=self.result,
            diagnostics=self.diagnostics,
            label="result",
        )

    @property
    def ok(self) -> bool:
        return self.status is RuntimeCallStatus.OK


def _validate_call_result(
    status: RuntimeCallStatus,
    *,
    success_value: object | None,
    diagnostics: tuple[RuntimeDiagnostic, ...],
    label: str,
    allow_failure_value: bool = False,
) -> None:
    if not isinstance(status, RuntimeCallStatus):
        raise ValueError(f"{label} status must use the closed enum")
    if any(not isinstance(item, RuntimeDiagnostic) for item in diagnostics):
        raise ValueError(f"{label} diagnostics have the wrong type")
    if status is RuntimeCallStatus.OK:
        if success_value is None or diagnostics:
            raise ValueError(f"successful {label} requires a value and no diagnostics")
    elif not diagnostics or (success_value is not None and not allow_failure_value):
        raise ValueError(f"failed {label} requires diagnostics and no success value")


def executable_artifact_envelope_is_valid(
    artifact: SealedExecutableArtifact,
) -> bool:
    if not isinstance(artifact, SealedExecutableArtifact):
        return False
    if artifact.payload_content_hash != hashlib.sha256(artifact.payload).hexdigest():
        return False
    return artifact.artifact_identity == _digest(artifact.identity_dict())


@runtime_checkable
class RuntimeAdapter(Protocol):
    adapter_identity: str
    adapter_kind: AdapterKind

    def submit(
        self,
        artifact: SealedExecutableArtifact,
        options: RuntimeBindings = RuntimeBindings(),
    ) -> SubmitResult: ...

    def status(self, handle: ExecutionHandle) -> StatusResult: ...

    def cancel(self, handle: ExecutionHandle) -> CancelResult: ...

    def result(self, handle: ExecutionHandle) -> FetchResult: ...


__all__ = [
    "AdapterKind",
    "CancelResult",
    "ExecutionBinding",
    "ExecutionHandle",
    "RuntimeBindings",
    "ExecutionState",
    "ExternalExecutionIdentity",
    "FetchResult",
    "RuntimeAdapter",
    "RuntimeCallStatus",
    "RuntimeDiagnostic",
    "RuntimeParameterBinding",
    "RuntimeResult",
    "StatusResult",
    "SubmissionReceipt",
    "SubmitResult",
    "create_execution_binding",
    "create_runtime_result",
    "create_submission_receipt",
    "executable_artifact_envelope_is_valid",
]

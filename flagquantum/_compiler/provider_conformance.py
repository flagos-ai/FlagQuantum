"""Private, provider-neutral conformance harness for Phase 3 target families."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from enum import Enum
from typing import Protocol, runtime_checkable

from .executable_artifact import (
    SealedExecutableArtifact,
    verify_executable_artifact,
)
from .runtime_abi import (
    AdapterKind,
    ExecutionHandle,
    ExecutionState,
    RuntimeAdapter,
    RuntimeBindings,
    RuntimeCallStatus,
    RuntimeDiagnostic,
    StatusResult,
)
from .runtime_adapters import LocalSyncRuntimeAdapter, OfflineAsyncMockRuntimeAdapter
from .target_capabilities import (
    ArtifactFormat,
    TargetCapabilities,
    TargetClass,
)
from .target_ir import TargetIR

_NAMESPACE = re.compile(r"[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)+")
_KEY = re.compile(r"[a-z][a-z0-9_]*")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_FORBIDDEN_EXTENSION_TERMS = frozenset(
    {
        "account",
        "access_key",
        "api_key",
        "artifact_identity",
        "authorization",
        "backend",
        "binding_identity",
        "credential",
        "cookie",
        "endpoint",
        "job",
        "password",
        "private_key",
        "price",
        "provider",
        "queue",
        "quota",
        "receipt_identity",
        "secret",
        "session",
        "token",
        "url",
        "user",
    }
)
_FORBIDDEN_VALUE_MARKERS = (
    "access_token",
    "api_key",
    "authorization:",
    "bearer ",
    "credential=",
    "http://",
    "https://",
    "password=",
    "private_key",
    "secret=",
)
_CONFORMANCE_CHECKS = (
    "artifact_verified",
    "submitted",
    "initial_status_observed",
    "settled",
    "terminal_success_observed",
    "result_identity_verified",
    "terminal_reads_idempotent",
    "terminal_cancel_rejected",
)


def _digest(value: dict[str, object]) -> str:
    encoded = json.dumps(
        value, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


class ConformanceTargetFamily(str, Enum):
    LOCAL_RUNTIME_PLAN = "local_runtime_plan"
    SYNTHETIC_QASM_TEXT = "synthetic_qasm_text"
    SYNTHETIC_NON_QASM_ARTIFACT = "synthetic_non_qasm_artifact"


class ConformanceStatus(str, Enum):
    PASSED = "passed"
    FAILED = "failed"


@dataclass(frozen=True, order=True)
class ProviderExtensionEntry:
    key: str
    value: str | int | float | bool

    def __post_init__(self) -> None:
        key = str(self.key).strip().lower()
        if _KEY.fullmatch(key) is None:
            raise ValueError("provider extension key must be lowercase snake case")
        if any(term in key for term in _FORBIDDEN_EXTENSION_TERMS):
            raise ValueError(
                "provider extension key is reserved or execution-sensitive"
            )
        value = self.value
        if not isinstance(value, (str, int, float, bool)):
            raise ValueError(
                "provider extension value must be an immutable JSON scalar"
            )
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("provider extension numeric value must be finite")
        if isinstance(value, str):
            value = value.strip()
            if not value:
                raise ValueError("provider extension string value cannot be empty")
            lowered = value.lower()
            if any(marker in lowered for marker in _FORBIDDEN_VALUE_MARKERS):
                raise ValueError("provider extension value contains sensitive data")
        object.__setattr__(self, "key", key)
        object.__setattr__(self, "value", value)

    def to_dict(self) -> dict[str, object]:
        return {"key": self.key, "value": self.value}


@dataclass(frozen=True, order=True)
class ProviderExtension:
    """Namespaced non-semantic metadata excluded from every execution identity."""

    namespace: str
    entries: tuple[ProviderExtensionEntry, ...]

    def __post_init__(self) -> None:
        namespace = str(self.namespace).strip().lower()
        if _NAMESPACE.fullmatch(namespace) is None:
            raise ValueError("provider extension requires a dotted lowercase namespace")
        entries = tuple(sorted(self.entries, key=lambda item: item.key))
        if not entries or any(
            not isinstance(item, ProviderExtensionEntry) for item in entries
        ):
            raise ValueError("provider extension entries must be non-empty and typed")
        keys = tuple(item.key for item in entries)
        if len(keys) != len(set(keys)):
            raise ValueError("provider extension keys must be unique")
        object.__setattr__(self, "namespace", namespace)
        object.__setattr__(self, "entries", entries)

    def to_dict(self) -> dict[str, object]:
        return {
            "namespace": self.namespace,
            "entries": [item.to_dict() for item in self.entries],
        }


@dataclass(frozen=True)
class ConformanceCase:
    family: ConformanceTargetFamily
    target: TargetCapabilities
    target_ir: TargetIR
    artifact: SealedExecutableArtifact
    extensions: tuple[ProviderExtension, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.family, ConformanceTargetFamily):
            raise ValueError("conformance family must use the closed enum")
        if not isinstance(self.target, TargetCapabilities):
            raise ValueError("conformance target must use TargetCapabilities")
        if not isinstance(self.target_ir, TargetIR):
            raise ValueError("conformance program must use TargetIR")
        if not isinstance(self.artifact, SealedExecutableArtifact):
            raise ValueError("conformance artifact must be sealed")
        extensions = tuple(sorted(self.extensions, key=lambda item: item.namespace))
        if any(not isinstance(item, ProviderExtension) for item in extensions):
            raise ValueError("conformance extensions must use ProviderExtension")
        namespaces = tuple(item.namespace for item in extensions)
        if len(namespaces) != len(set(namespaces)):
            raise ValueError("provider extension namespaces must be unique")
        self._validate_family_profile()
        if not verify_executable_artifact(
            self.artifact, self.target_ir, self.target
        ).ok:
            raise ValueError("conformance artifact does not match TargetIR and target")
        object.__setattr__(self, "extensions", extensions)

    def _validate_family_profile(self) -> None:
        expected = {
            ConformanceTargetFamily.LOCAL_RUNTIME_PLAN: (
                TargetClass.LOCAL_RUNTIME,
                frozenset({ArtifactFormat.RUNTIME_PLAN}),
            ),
            ConformanceTargetFamily.SYNTHETIC_QASM_TEXT: (
                TargetClass.QASM_TEXT,
                frozenset(
                    {ArtifactFormat.OPENQASM_2, ArtifactFormat.OPENQASM_3_STATIC}
                ),
            ),
            ConformanceTargetFamily.SYNTHETIC_NON_QASM_ARTIFACT: (
                TargetClass.NON_QASM_ARTIFACT,
                frozenset({ArtifactFormat.QCIS_1}),
            ),
        }
        target_class, formats = expected[self.family]
        if self.target.target_class is not target_class:
            raise ValueError("conformance family and target class do not match")
        if self.artifact.profile.format not in formats:
            raise ValueError("conformance family and artifact profile do not match")


@runtime_checkable
class ConformanceDriver(Protocol):
    adapter: RuntimeAdapter

    def settle(self, handle: ExecutionHandle, payload: bytes) -> StatusResult: ...


class LocalConformanceDriver:
    def __init__(self, adapter_identity: str, payload: bytes) -> None:
        if not isinstance(payload, bytes):
            raise ValueError("conformance result payload must be immutable bytes")
        self._payload = payload
        self.adapter = LocalSyncRuntimeAdapter(
            adapter_identity,
            lambda _artifact, _options: payload,
        )

    def settle(self, handle: ExecutionHandle, payload: bytes) -> StatusResult:
        status = self.adapter.status(handle)
        if payload != self._payload or status.state is not ExecutionState.SUCCEEDED:
            return StatusResult(
                RuntimeCallStatus.TERMINAL,
                status.state,
                diagnostics=status.diagnostics
                or (
                    _runtime_diagnostic(
                        RuntimeCallStatus.TERMINAL,
                        "local conformance payload or terminal state does not match",
                    ),
                ),
            )
        return status


class OfflineConformanceDriver:
    def __init__(self, adapter_identity: str) -> None:
        self.adapter = OfflineAsyncMockRuntimeAdapter(adapter_identity)

    def settle(self, handle: ExecutionHandle, payload: bytes) -> StatusResult:
        started = self.adapter.start(handle)
        if not started.ok:
            return started
        return self.adapter.complete(handle, payload)


def _runtime_diagnostic(status: RuntimeCallStatus, message: str) -> RuntimeDiagnostic:
    return RuntimeDiagnostic(status, message)


@dataclass(frozen=True)
class ConformanceDiagnostic:
    stage: str
    message: str

    def __post_init__(self) -> None:
        stage = str(self.stage).strip()
        message = str(self.message).strip()
        if not stage or not message:
            raise ValueError("conformance diagnostic fields cannot be empty")
        object.__setattr__(self, "stage", stage)
        object.__setattr__(self, "message", message)


@dataclass(frozen=True)
class ConformanceReport:
    family: ConformanceTargetFamily
    artifact_identity: str
    binding_identity: str
    receipt_identity: str
    result_identity: str
    checks: tuple[str, ...]
    extension_namespaces: tuple[str, ...]
    conformance_identity: str

    def __post_init__(self) -> None:
        if not isinstance(self.family, ConformanceTargetFamily):
            raise ValueError("conformance report family must use the closed enum")
        for name in (
            "artifact_identity",
            "binding_identity",
            "receipt_identity",
            "result_identity",
            "conformance_identity",
        ):
            if _SHA256.fullmatch(getattr(self, name)) is None:
                raise ValueError(f"conformance report {name} must be lowercase SHA-256")
        if self.checks != _CONFORMANCE_CHECKS:
            raise ValueError(
                "conformance report must contain the complete ordered suite"
            )
        if self.extension_namespaces != tuple(sorted(self.extension_namespaces)):
            raise ValueError("conformance extension namespaces must be ordered")
        if len(self.extension_namespaces) != len(set(self.extension_namespaces)):
            raise ValueError("conformance extension namespaces must be unique")
        if any(
            _NAMESPACE.fullmatch(item) is None for item in self.extension_namespaces
        ):
            raise ValueError("conformance extension namespace is invalid")
        if self.conformance_identity != _digest(self.semantic_identity_dict()):
            raise ValueError("conformance identity does not match semantic evidence")

    def semantic_identity_dict(self) -> dict[str, object]:
        return {
            "family": self.family.value,
            "artifact_identity": self.artifact_identity,
            "binding_identity": self.binding_identity,
            "receipt_identity": self.receipt_identity,
            "result_identity": self.result_identity,
            "checks": list(self.checks),
        }


@dataclass(frozen=True)
class ConformanceResult:
    status: ConformanceStatus
    report: ConformanceReport | None = None
    diagnostics: tuple[ConformanceDiagnostic, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.status, ConformanceStatus):
            raise ValueError("conformance status must use the closed enum")
        if any(
            not isinstance(item, ConformanceDiagnostic) for item in self.diagnostics
        ):
            raise ValueError("conformance diagnostics have the wrong type")
        if self.status is ConformanceStatus.PASSED:
            if self.report is None or self.diagnostics:
                raise ValueError("passed conformance requires a report only")
        elif self.report is not None or not self.diagnostics:
            raise ValueError("failed conformance requires diagnostics only")

    @property
    def ok(self) -> bool:
        return self.status is ConformanceStatus.PASSED


def _failure(stage: str, message: str) -> ConformanceResult:
    return ConformanceResult(
        ConformanceStatus.FAILED,
        diagnostics=(ConformanceDiagnostic(stage, message),),
    )


def run_conformance_case(
    case: ConformanceCase,
    driver: ConformanceDriver,
    *,
    result_payload: bytes = b"flagquantum-conformance-result-v1",
) -> ConformanceResult:
    """Run one identical lifecycle contract against any authorized target family."""

    if not isinstance(case, ConformanceCase) or not isinstance(
        driver, ConformanceDriver
    ):
        return _failure("input", "conformance requires a typed case and driver")
    expected_kind = (
        AdapterKind.LOCAL_SYNC
        if case.family is ConformanceTargetFamily.LOCAL_RUNTIME_PLAN
        else AdapterKind.OFFLINE_ASYNC_MOCK
    )
    if driver.adapter.adapter_kind is not expected_kind:
        return _failure(
            "compatibility", "target family and runtime adapter do not match"
        )
    submitted = driver.adapter.submit(case.artifact, RuntimeBindings())
    if not submitted.ok or submitted.receipt is None:
        return _failure("submit", "runtime rejected the conformance artifact")
    receipt = submitted.receipt
    initial = driver.adapter.status(receipt.handle)
    if not initial.ok:
        return _failure("initial_status", "runtime did not expose an initial state")
    settled = driver.settle(receipt.handle, result_payload)
    if not settled.ok or settled.state is not ExecutionState.SUCCEEDED:
        return _failure("settle", "runtime did not reach successful terminal state")
    terminal = driver.adapter.status(receipt.handle)
    fetched = driver.adapter.result(receipt.handle)
    repeated = driver.adapter.result(receipt.handle)
    cancelled = driver.adapter.cancel(receipt.handle)
    if terminal.state is not ExecutionState.SUCCEEDED:
        return _failure("terminal_status", "runtime terminal state changed")
    if not fetched.ok or fetched.result is None:
        return _failure("result", "runtime did not return an identity-bound result")
    if not repeated.ok or repeated.result != fetched.result:
        return _failure("idempotency", "terminal result read is not idempotent")
    result = fetched.result
    if (
        receipt.binding.artifact_identity != case.artifact.artifact_identity
        or result.artifact_identity != case.artifact.artifact_identity
        or result.binding_identity != receipt.binding.binding_identity
        or result.receipt_identity != receipt.receipt_identity
    ):
        return _failure("identity", "artifact-to-result identity chain is broken")
    if cancelled.status is not RuntimeCallStatus.TERMINAL or cancelled.acknowledged:
        return _failure("terminal_cancel", "successful execution accepted cancellation")
    semantic = {
        "family": case.family.value,
        "artifact_identity": case.artifact.artifact_identity,
        "binding_identity": receipt.binding.binding_identity,
        "receipt_identity": receipt.receipt_identity,
        "result_identity": result.result_identity,
        "checks": list(_CONFORMANCE_CHECKS),
    }
    report = ConformanceReport(
        case.family,
        case.artifact.artifact_identity,
        receipt.binding.binding_identity,
        receipt.receipt_identity,
        result.result_identity,
        _CONFORMANCE_CHECKS,
        tuple(item.namespace for item in case.extensions),
        _digest(semantic),
    )
    return ConformanceResult(ConformanceStatus.PASSED, report)


__all__ = [
    "ConformanceCase",
    "ConformanceDiagnostic",
    "ConformanceDriver",
    "ConformanceReport",
    "ConformanceResult",
    "ConformanceStatus",
    "ConformanceTargetFamily",
    "LocalConformanceDriver",
    "OfflineConformanceDriver",
    "ProviderExtension",
    "ProviderExtensionEntry",
    "run_conformance_case",
]

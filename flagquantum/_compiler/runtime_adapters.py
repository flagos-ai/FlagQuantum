"""Private local and offline adapters implementing the Phase 3 runtime ABI."""

from __future__ import annotations

import hashlib
import re
import threading
from collections.abc import Callable
from dataclasses import dataclass

from .executable_artifact import SealedExecutableArtifact
from .runtime_abi import (
    AdapterKind,
    CancelResult,
    ExecutionHandle,
    ExecutionState,
    FetchResult,
    RuntimeBindings,
    RuntimeCallStatus,
    RuntimeDiagnostic,
    RuntimeResult,
    StatusResult,
    SubmissionReceipt,
    SubmitResult,
    create_execution_binding,
    create_runtime_result,
    create_submission_receipt,
    executable_artifact_envelope_is_valid,
)
from .target_capabilities import ArtifactFormat

_SHA256 = re.compile(r"[0-9a-f]{64}")


def _diagnostic(
    status: RuntimeCallStatus, message: str
) -> tuple[RuntimeDiagnostic, ...]:
    return (RuntimeDiagnostic(status, message),)


@dataclass
class _ExecutionRecord:
    receipt: SubmissionReceipt
    state: ExecutionState
    result: RuntimeResult | None = None


class _MemoryRuntimeAdapter:
    adapter_kind: AdapterKind

    def __init__(self, adapter_identity: str) -> None:
        if (
            not isinstance(adapter_identity, str)
            or _SHA256.fullmatch(adapter_identity) is None
        ):
            raise ValueError("adapter identity must be a lowercase SHA-256 digest")
        self.adapter_identity = adapter_identity
        self._records: dict[str, _ExecutionRecord] = {}
        self._ordinal = 0
        self._lock = threading.RLock()

    def _create_record(
        self,
        artifact: SealedExecutableArtifact,
        options: RuntimeBindings,
        state: ExecutionState,
    ) -> _ExecutionRecord:
        binding = create_execution_binding(
            artifact,
            self.adapter_identity,
            self.adapter_kind,
            options,
        )
        self._ordinal += 1
        token_source = (
            f"{binding.binding_identity}:{self._ordinal}:{self.adapter_kind.value}"
        )
        handle = ExecutionHandle(
            hashlib.sha256(token_source.encode("utf-8")).hexdigest(),
            artifact.artifact_identity,
            binding.binding_identity,
        )
        receipt = create_submission_receipt(handle, binding)
        record = _ExecutionRecord(receipt, state)
        self._records[handle.token] = record
        return record

    def _lookup(self, handle: ExecutionHandle) -> _ExecutionRecord | None:
        if not isinstance(handle, ExecutionHandle):
            return None
        record = self._records.get(handle.token)
        if record is None or record.receipt.handle != handle:
            return None
        return record

    def status(self, handle: ExecutionHandle) -> StatusResult:
        with self._lock:
            record = self._lookup(handle)
            if record is None:
                return StatusResult(
                    RuntimeCallStatus.UNKNOWN_HANDLE,
                    diagnostics=_diagnostic(
                        RuntimeCallStatus.UNKNOWN_HANDLE,
                        "execution handle is unknown or its identity chain changed",
                    ),
                )
            return StatusResult(RuntimeCallStatus.OK, record.state)

    def cancel(self, handle: ExecutionHandle) -> CancelResult:
        with self._lock:
            record = self._lookup(handle)
            if record is None:
                return CancelResult(
                    RuntimeCallStatus.UNKNOWN_HANDLE,
                    diagnostics=_diagnostic(
                        RuntimeCallStatus.UNKNOWN_HANDLE,
                        "execution handle is unknown or its identity chain changed",
                    ),
                )
            if record.state in {ExecutionState.QUEUED, ExecutionState.RUNNING}:
                record.state = ExecutionState.CANCELLED
                return CancelResult(RuntimeCallStatus.OK, record.state, True)
            if record.state is ExecutionState.CANCELLED:
                return CancelResult(RuntimeCallStatus.OK, record.state, True)
            return CancelResult(
                RuntimeCallStatus.TERMINAL,
                record.state,
                False,
                _diagnostic(
                    RuntimeCallStatus.TERMINAL,
                    "completed execution cannot transition to cancelled",
                ),
            )

    def result(self, handle: ExecutionHandle) -> FetchResult:
        with self._lock:
            record = self._lookup(handle)
            if record is None:
                return FetchResult(
                    RuntimeCallStatus.UNKNOWN_HANDLE,
                    diagnostics=_diagnostic(
                        RuntimeCallStatus.UNKNOWN_HANDLE,
                        "execution handle is unknown or its identity chain changed",
                    ),
                )
            if record.state is ExecutionState.SUCCEEDED:
                if record.result is None:  # pragma: no cover - internal invariant.
                    return FetchResult(
                        RuntimeCallStatus.EXECUTION_FAILED,
                        diagnostics=_diagnostic(
                            RuntimeCallStatus.EXECUTION_FAILED,
                            "successful execution has no identity-bound result",
                        ),
                    )
                return FetchResult(RuntimeCallStatus.OK, record.result)
            if record.state in {ExecutionState.QUEUED, ExecutionState.RUNNING}:
                return FetchResult(
                    RuntimeCallStatus.NOT_READY,
                    diagnostics=_diagnostic(
                        RuntimeCallStatus.NOT_READY,
                        "execution result is not ready",
                    ),
                )
            status = (
                RuntimeCallStatus.EXECUTION_FAILED
                if record.state is ExecutionState.FAILED
                else RuntimeCallStatus.TERMINAL
            )
            return FetchResult(
                status,
                diagnostics=_diagnostic(
                    status,
                    f"execution ended in {record.state.value} state without a result",
                ),
            )

    def _validate_submission(
        self,
        artifact: SealedExecutableArtifact,
        options: RuntimeBindings,
    ) -> SubmitResult | None:
        if not executable_artifact_envelope_is_valid(artifact):
            return SubmitResult(
                RuntimeCallStatus.INVALID_REQUEST,
                diagnostics=_diagnostic(
                    RuntimeCallStatus.INVALID_REQUEST,
                    "runtime requires an intrinsically valid sealed artifact",
                ),
            )
        if not isinstance(options, RuntimeBindings):
            return SubmitResult(
                RuntimeCallStatus.INVALID_REQUEST,
                diagnostics=_diagnostic(
                    RuntimeCallStatus.INVALID_REQUEST,
                    "runtime bindings must use RuntimeBindings",
                ),
            )
        return None


class LocalSyncRuntimeAdapter(_MemoryRuntimeAdapter):
    """Synchronous adapter for existing canonical runtime-plan artifacts."""

    adapter_kind = AdapterKind.LOCAL_SYNC

    def __init__(
        self,
        adapter_identity: str,
        executor: Callable[[SealedExecutableArtifact, RuntimeBindings], bytes],
    ) -> None:
        super().__init__(adapter_identity)
        if not callable(executor):
            raise ValueError("local runtime executor must be callable")
        self._executor = executor

    def submit(
        self,
        artifact: SealedExecutableArtifact,
        options: RuntimeBindings = RuntimeBindings(),
    ) -> SubmitResult:
        invalid = self._validate_submission(artifact, options)
        if invalid is not None:
            return invalid
        if artifact.profile.format is not ArtifactFormat.RUNTIME_PLAN:
            return SubmitResult(
                RuntimeCallStatus.INVALID_REQUEST,
                diagnostics=_diagnostic(
                    RuntimeCallStatus.INVALID_REQUEST,
                    "local synchronous adapter accepts runtime-plan artifacts only",
                ),
            )
        with self._lock:
            record = self._create_record(artifact, options, ExecutionState.RUNNING)
            try:
                payload = self._executor(artifact, options)
                record.result = create_runtime_result(record.receipt, payload)
            except (
                Exception
            ):  # noqa: BLE001 - adapter boundary sanitizes executor failure.
                record.state = ExecutionState.FAILED
            else:
                record.state = ExecutionState.SUCCEEDED
            return SubmitResult(RuntimeCallStatus.OK, record.receipt)


class OfflineAsyncMockRuntimeAdapter(_MemoryRuntimeAdapter):
    """Deterministic, network-free asynchronous lifecycle test double."""

    adapter_kind = AdapterKind.OFFLINE_ASYNC_MOCK

    def submit(
        self,
        artifact: SealedExecutableArtifact,
        options: RuntimeBindings = RuntimeBindings(),
    ) -> SubmitResult:
        invalid = self._validate_submission(artifact, options)
        if invalid is not None:
            return invalid
        with self._lock:
            record = self._create_record(artifact, options, ExecutionState.QUEUED)
            return SubmitResult(RuntimeCallStatus.OK, record.receipt)

    def start(self, handle: ExecutionHandle) -> StatusResult:
        with self._lock:
            record = self._lookup(handle)
            if record is None:
                return self.status(handle)
            if record.state is ExecutionState.QUEUED:
                record.state = ExecutionState.RUNNING
                return StatusResult(RuntimeCallStatus.OK, record.state)
            if record.state is ExecutionState.RUNNING:
                return StatusResult(RuntimeCallStatus.OK, record.state)
            return StatusResult(
                RuntimeCallStatus.TERMINAL,
                record.state,
                _diagnostic(
                    RuntimeCallStatus.TERMINAL,
                    "terminal execution cannot transition to running",
                ),
            )

    def complete(self, handle: ExecutionHandle, payload: bytes) -> StatusResult:
        with self._lock:
            record = self._lookup(handle)
            if record is None:
                return self.status(handle)
            if record.state is ExecutionState.SUCCEEDED:
                if record.result is not None and record.result.payload == payload:
                    return StatusResult(RuntimeCallStatus.OK, record.state)
                return StatusResult(
                    RuntimeCallStatus.TERMINAL,
                    record.state,
                    _diagnostic(
                        RuntimeCallStatus.TERMINAL,
                        "successful execution cannot accept a different result",
                    ),
                )
            if record.state is not ExecutionState.RUNNING:
                return StatusResult(
                    RuntimeCallStatus.TERMINAL,
                    record.state,
                    _diagnostic(
                        RuntimeCallStatus.TERMINAL,
                        "only a running execution can complete",
                    ),
                )
            try:
                record.result = create_runtime_result(record.receipt, payload)
            except ValueError as exc:
                return StatusResult(
                    RuntimeCallStatus.INVALID_REQUEST,
                    record.state,
                    _diagnostic(RuntimeCallStatus.INVALID_REQUEST, str(exc)),
                )
            record.state = ExecutionState.SUCCEEDED
            return StatusResult(RuntimeCallStatus.OK, record.state)

    def fail(self, handle: ExecutionHandle) -> StatusResult:
        with self._lock:
            record = self._lookup(handle)
            if record is None:
                return self.status(handle)
            if record.state in {ExecutionState.QUEUED, ExecutionState.RUNNING}:
                record.state = ExecutionState.FAILED
                return StatusResult(RuntimeCallStatus.OK, record.state)
            if record.state is ExecutionState.FAILED:
                return StatusResult(RuntimeCallStatus.OK, record.state)
            return StatusResult(
                RuntimeCallStatus.TERMINAL,
                record.state,
                _diagnostic(
                    RuntimeCallStatus.TERMINAL,
                    "terminal execution cannot transition to failed",
                ),
            )


__all__ = ["LocalSyncRuntimeAdapter", "OfflineAsyncMockRuntimeAdapter"]

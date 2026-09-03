from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError, fields, replace

import pytest

import flagquantum as fq
from flagquantum._compiler.executable_artifact import seal_executable_artifact
from flagquantum._compiler.runtime_abi import (
    CancelResult,
    ExecutionOptions,
    ExecutionState,
    ExternalExecutionIdentity,
    FetchResult,
    RuntimeAdapter,
    RuntimeCallStatus,
    RuntimeParameterBinding,
    StatusResult,
    SubmissionReceipt,
    SubmitResult,
    create_submission_receipt,
)
from flagquantum._compiler.runtime_adapters import (
    LocalSyncRuntimeAdapter,
    OfflineAsyncMockRuntimeAdapter,
)
from flagquantum._compiler.target_capabilities import (
    ArtifactFormat,
    ArtifactProfile,
    GateCapability,
    MeasurementResult,
    TargetCapabilities,
    TargetClass,
)
from flagquantum._compiler.target_ir import TargetIR, TargetOperation

pytestmark = pytest.mark.unit

ADAPTER_IDENTITY = "d" * 64
COMPILATION_IDENTITY = "c" * 64


def _artifact():
    profile = ArtifactProfile(ArtifactFormat.RUNTIME_PLAN, "1.0")
    target = TargetCapabilities(
        target_class=TargetClass.LOCAL_RUNTIME,
        logical_qubit_capacity=1,
        physical_qubit_capacity=1,
        native_gates=(GateCapability("rx"),),
        measurement_results=(MeasurementResult.STATE,),
        artifact_profiles=(profile,),
        maximum_program_operations=10,
    )
    target_ir = TargetIR(
        "a" * 64,
        target.semantic_fingerprint,
        (0,),
        (TargetOperation("rx", (0,), {"theta": 0.25}),),
    )
    payload = json.dumps(
        target_ir.canonical(),
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    sealed = seal_executable_artifact(
        target_ir,
        target,
        profile,
        payload,
        compilation_identity=COMPILATION_IDENTITY,
    ).artifact
    assert sealed is not None
    return sealed


def test_local_sync_lifecycle_binds_receipt_and_result_to_artifact() -> None:
    artifact = _artifact()
    options = ExecutionOptions(
        shots=16,
        parameter_bindings=(RuntimeParameterBinding("theta", 0.5),),
    )
    adapter = LocalSyncRuntimeAdapter(
        ADAPTER_IDENTITY,
        lambda submitted, request: (
            f"{submitted.artifact_identity}:{request.shots}".encode()
        ),
    )

    assert isinstance(adapter, RuntimeAdapter)
    submitted = adapter.submit(artifact, options)
    assert submitted.ok and submitted.receipt is not None
    receipt = submitted.receipt
    assert receipt.binding.artifact_identity == artifact.artifact_identity
    assert receipt.binding.options == options
    assert receipt.external_identity is None
    assert adapter.status(receipt.handle).state is ExecutionState.SUCCEEDED

    fetched = adapter.result(receipt.handle)
    assert fetched.ok and fetched.result is not None
    assert fetched.result.artifact_identity == artifact.artifact_identity
    assert fetched.result.binding_identity == receipt.binding.binding_identity
    assert fetched.result.receipt_identity == receipt.receipt_identity
    assert fetched.result.payload.endswith(b":16")

    cancelled = adapter.cancel(receipt.handle)
    assert cancelled.status is RuntimeCallStatus.TERMINAL
    assert cancelled.acknowledged is False
    assert adapter.status(receipt.handle).state is ExecutionState.SUCCEEDED


def test_local_failure_is_terminal_and_sanitizes_executor_exception() -> None:
    def fail_with_secret(*_args):
        raise RuntimeError("credential=do-not-leak")

    adapter = LocalSyncRuntimeAdapter(ADAPTER_IDENTITY, fail_with_secret)
    submitted = adapter.submit(_artifact())
    assert submitted.ok and submitted.receipt is not None

    fetched = adapter.result(submitted.receipt.handle)
    assert fetched.status is RuntimeCallStatus.EXECUTION_FAILED
    assert fetched.result is None
    assert "do-not-leak" not in repr(fetched.diagnostics)


def test_offline_async_success_lifecycle_and_terminal_idempotency() -> None:
    adapter = OfflineAsyncMockRuntimeAdapter(ADAPTER_IDENTITY)
    submitted = adapter.submit(_artifact())
    assert submitted.ok and submitted.receipt is not None
    handle = submitted.receipt.handle

    assert adapter.status(handle).state is ExecutionState.QUEUED
    assert adapter.result(handle).status is RuntimeCallStatus.NOT_READY
    assert adapter.start(handle).state is ExecutionState.RUNNING
    assert adapter.start(handle).state is ExecutionState.RUNNING
    assert adapter.complete(handle, b"result").state is ExecutionState.SUCCEEDED
    assert adapter.complete(handle, b"result").status is RuntimeCallStatus.OK
    assert adapter.complete(handle, b"different").status is RuntimeCallStatus.TERMINAL
    assert adapter.result(handle).result.payload == b"result"


def test_offline_async_cancel_and_failure_are_idempotent_terminal_states() -> None:
    adapter = OfflineAsyncMockRuntimeAdapter(ADAPTER_IDENTITY)
    cancelled = adapter.submit(_artifact()).receipt
    failed = adapter.submit(_artifact()).receipt
    assert cancelled is not None and failed is not None

    first_cancel = adapter.cancel(cancelled.handle)
    second_cancel = adapter.cancel(cancelled.handle)
    assert first_cancel.ok and second_cancel.ok
    assert first_cancel.acknowledged and second_cancel.acknowledged
    assert adapter.start(cancelled.handle).status is RuntimeCallStatus.TERMINAL

    assert adapter.start(failed.handle).state is ExecutionState.RUNNING
    assert adapter.fail(failed.handle).state is ExecutionState.FAILED
    assert adapter.fail(failed.handle).state is ExecutionState.FAILED
    assert adapter.result(failed.handle).status is RuntimeCallStatus.EXECUTION_FAILED


def test_unknown_and_identity_tampered_handles_fail_closed() -> None:
    adapter = OfflineAsyncMockRuntimeAdapter(ADAPTER_IDENTITY)
    receipt = adapter.submit(_artifact()).receipt
    assert receipt is not None
    tampered = replace(receipt.handle, artifact_identity="e" * 64)
    unknown = replace(receipt.handle, token="f" * 64)

    for handle in (tampered, unknown):
        assert adapter.status(handle).status is RuntimeCallStatus.UNKNOWN_HANDLE
        assert adapter.cancel(handle).status is RuntimeCallStatus.UNKNOWN_HANDLE
        assert adapter.result(handle).status is RuntimeCallStatus.UNKNOWN_HANDLE


def test_invalid_artifact_and_options_are_rejected_before_record_creation() -> None:
    artifact = _artifact()
    adapter = OfflineAsyncMockRuntimeAdapter(ADAPTER_IDENTITY)
    tampered = replace(artifact, payload=artifact.payload + b"changed")

    assert adapter.submit(tampered).status is RuntimeCallStatus.INVALID_REQUEST
    assert (
        adapter.submit(artifact, object()).status is RuntimeCallStatus.INVALID_REQUEST
    )
    with pytest.raises(ValueError, match="adapter identity"):
        OfflineAsyncMockRuntimeAdapter("not-a-digest")


def test_execution_only_external_identity_is_receipt_scoped() -> None:
    artifact = _artifact()
    adapter = OfflineAsyncMockRuntimeAdapter(ADAPTER_IDENTITY)
    receipt = adapter.submit(artifact).receipt
    assert receipt is not None
    external = ExternalExecutionIdentity(
        "anonymous-provider",
        "anonymous-backend",
        "anonymous-job",
    )
    external_receipt = create_submission_receipt(
        receipt.handle,
        receipt.binding,
        external,
    )

    assert external_receipt.external_identity == external
    assert external_receipt.receipt_identity != receipt.receipt_identity
    assert {item.name for item in fields(SubmissionReceipt)} == {
        "handle",
        "binding",
        "external_identity",
        "receipt_identity",
    }
    portable_names = {item.name for item in fields(type(artifact))}
    assert portable_names.isdisjoint({"provider", "backend_id", "job_id"})


def test_call_results_reject_contradictory_success_states() -> None:
    with pytest.raises(ValueError, match="successful submit"):
        SubmitResult(RuntimeCallStatus.OK)
    with pytest.raises(ValueError, match="successful status"):
        StatusResult(RuntimeCallStatus.OK)
    with pytest.raises(ValueError, match="successful cancel"):
        CancelResult(RuntimeCallStatus.OK)
    with pytest.raises(ValueError, match="successful result"):
        FetchResult(RuntimeCallStatus.OK)


def test_runtime_values_are_immutable_and_parallel_handles_are_unique() -> None:
    artifact = _artifact()
    adapter = OfflineAsyncMockRuntimeAdapter(ADAPTER_IDENTITY)

    with ThreadPoolExecutor(max_workers=8) as executor:
        receipts = list(
            executor.map(lambda _: adapter.submit(artifact).receipt, range(64))
        )
    assert all(receipt is not None for receipt in receipts)
    assert len({receipt.handle.token for receipt in receipts}) == 64
    with pytest.raises(FrozenInstanceError):
        receipts[0].handle.token = "0" * 64


def test_runtime_adapter_remains_private() -> None:
    for name in (
        "RuntimeAdapter",
        "ExecutionBinding",
        "LocalSyncRuntimeAdapter",
        "OfflineAsyncMockRuntimeAdapter",
    ):
        assert not hasattr(fq, name)

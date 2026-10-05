from __future__ import annotations

import json
import os
import stat
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from flagquantum.remote.kaiwu import (
    KaiwuTaskReceipt,
    KaiwuTaskResult,
    new_receipt,
    restore_kaiwu_job,
    submit_kaiwu_task,
)
from flagquantum.remote.kaiwu.contracts import (
    FrozenIsingMatrix,
    KaiwuTaskMode,
)

pytestmark = pytest.mark.unit

_MATRIX = ((0.0, 1.0), (1.0, 0.0))


class _FakeClient:
    def __init__(self, statuses: list[str] | None = None) -> None:
        self.statuses = statuses or ["Completed"]
        self.submit_calls = 0
        self.status_calls = 0
        self.result_calls = 0

    def submit(
        self,
        matrix: FrozenIsingMatrix,
        *,
        task_name: str,
        mode: KaiwuTaskMode,
        requested_samples: int,
        project_no: str | None,
    ) -> KaiwuTaskReceipt:
        self.submit_calls += 1
        return new_receipt(
            task_name=task_name,
            matrix=matrix,
            mode=mode,
            requested_samples=requested_samples,
            project_no=project_no,
            provider_task_id="provider-task-123",
            provider_target="SPQC-test",
        )

    def query_status(self, receipt: KaiwuTaskReceipt, matrix: FrozenIsingMatrix) -> str:
        del receipt, matrix
        index = min(self.status_calls, len(self.statuses) - 1)
        self.status_calls += 1
        return self.statuses[index]

    def fetch_result(
        self, receipt: KaiwuTaskReceipt, matrix: FrozenIsingMatrix
    ) -> KaiwuTaskResult:
        del matrix
        self.result_calls += 1
        samples = tuple((1, -1) for _ in range(receipt.requested_samples))
        return KaiwuTaskResult(
            receipt=receipt,
            samples=samples,
            energies=tuple(2.0 for _ in samples),
            raw_status="Completed",
            metadata={"fallback_occurred": False},
        )


def test_submit_status_and_result_never_resubmit() -> None:
    client = _FakeClient()
    job = submit_kaiwu_task(
        _MATRIX,
        client=client,
        task_name="bounded-sampling",
        mode="sampling",
        requested_samples=10,
        project_no="CPQC-test",
    )

    assert job.status() == "succeeded"
    result = job.result()
    assert len(result.samples) == 10
    assert result.metadata["fallback_occurred"] is False
    assert client.submit_calls == 1
    assert client.result_calls == 1


def test_wait_polls_to_completion_without_resubmission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _FakeClient(["Queued", "Running", "Completed"])
    job = submit_kaiwu_task(_MATRIX, client=client, task_name="wait-test")
    monkeypatch.setattr("flagquantum.remote.kaiwu.jobs.time.sleep", lambda _: None)

    result = job.wait(timeout=10.0, poll_interval=0.1)

    assert result.receipt.task_name == "wait-test"
    assert client.status_calls == 3
    assert client.submit_calls == 1


def test_timeout_preserves_recoverable_identity_without_fetch_or_resubmit() -> None:
    client = _FakeClient(["Queued"])
    job = submit_kaiwu_task(_MATRIX, client=client, task_name="timeout-test")

    with pytest.raises(TimeoutError, match="no cancellation or resubmission"):
        job.wait(timeout=0.0)

    assert job.receipt.provider_task_id == "provider-task-123"
    assert client.submit_calls == 1
    assert client.result_calls == 0


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("timeout", True, "timeout must be finite"),
        ("timeout", "1", "timeout must be finite"),
        ("timeout", complex(1, 0), "timeout must be finite"),
        ("poll_interval", False, "poll_interval must be finite"),
        ("poll_interval", "1", "poll_interval must be finite"),
        ("poll_interval", complex(1, 0), "poll_interval must be finite"),
    ),
)
def test_wait_rejects_invalid_scalar_controls(
    field: str, value: object, message: str
) -> None:
    job = submit_kaiwu_task(_MATRIX, client=_FakeClient(), task_name="invalid-wait")
    options: dict[str, object] = {"timeout": 1.0, "poll_interval": 0.1}
    options[field] = value

    with pytest.raises(ValueError, match=message):
        job.wait(**options)  # type: ignore[arg-type]


def test_status_rejects_non_string_provider_state() -> None:
    class MalformedStatusClient(_FakeClient):
        def query_status(  # type: ignore[override]
            self, receipt: KaiwuTaskReceipt, matrix: FrozenIsingMatrix
        ) -> object:
            del receipt, matrix
            return None

    job = submit_kaiwu_task(
        _MATRIX,
        client=MalformedStatusClient(),  # type: ignore[arg-type]
        task_name="malformed-status",
    )

    with pytest.raises(RuntimeError, match="invalid provider status"):
        job.status()


def test_save_and_restore_are_credential_free_and_do_not_submit(tmp_path: Path) -> None:
    source = _FakeClient()
    job = submit_kaiwu_task(_MATRIX, client=source, task_name="restore-test")
    receipt_path = tmp_path / "kaiwu-receipt.json"

    job.save(receipt_path)
    encoded = receipt_path.read_text()
    restored_client = _FakeClient()
    restored = restore_kaiwu_job(receipt_path, client=restored_client)

    assert restored.receipt == job.receipt
    assert restored.result().samples
    assert restored_client.submit_calls == 0
    assert "sdk_code" not in encoded.lower()
    assert "credential" not in encoded.lower()
    assert receipt_path.stat().st_mode & 0o777 == 0o600
    assert not list(tmp_path.glob(f".{receipt_path.name}.*.tmp"))


def test_save_never_overwrites_receipt(tmp_path: Path) -> None:
    job = submit_kaiwu_task(_MATRIX, client=_FakeClient(), task_name="save-test")
    receipt_path = tmp_path / "receipt.json"
    job.save(receipt_path)

    with pytest.raises(FileExistsError):
        job.save(receipt_path)


def test_save_syncs_file_and_parent_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    job = submit_kaiwu_task(_MATRIX, client=_FakeClient(), task_name="durable-save")
    synced_types: list[int] = []
    real_fsync = os.fsync

    def record_fsync(descriptor: int) -> None:
        synced_types.append(os.fstat(descriptor).st_mode)
        real_fsync(descriptor)

    monkeypatch.setattr("flagquantum.remote.kaiwu.jobs.os.fsync", record_fsync)

    job.save(tmp_path / "receipt.json")

    assert any(stat.S_ISREG(mode) for mode in synced_types)
    assert any(stat.S_ISDIR(mode) for mode in synced_types)


@pytest.mark.parametrize("unsafe_kind", ("public", "symlink"))
def test_save_rejects_unsafe_parent_directory(tmp_path: Path, unsafe_kind: str) -> None:
    job = submit_kaiwu_task(_MATRIX, client=_FakeClient(), task_name="unsafe-parent")
    private_parent = tmp_path / "private-parent"
    private_parent.mkdir(mode=0o700)
    if unsafe_kind == "public":
        private_parent.chmod(0o755)
        receipt_path = private_parent / "receipt.json"
    else:
        linked_parent = tmp_path / "linked-parent"
        linked_parent.symlink_to(private_parent, target_is_directory=True)
        receipt_path = linked_parent / "receipt.json"

    with pytest.raises(ValueError, match="private, non-symlink directory"):
        job.save(receipt_path)

    assert not receipt_path.exists()


def test_restore_rejects_public_parent_directory(tmp_path: Path) -> None:
    job = submit_kaiwu_task(_MATRIX, client=_FakeClient(), task_name="public-parent")
    private_parent = tmp_path / "private-parent"
    private_parent.mkdir(mode=0o700)
    receipt_path = private_parent / "receipt.json"
    job.save(receipt_path)
    private_parent.chmod(0o755)

    with pytest.raises(ValueError, match="private, non-symlink directory"):
        restore_kaiwu_job(receipt_path, client=_FakeClient())


def test_restore_rejects_matrix_identity_tampering(tmp_path: Path) -> None:
    job = submit_kaiwu_task(_MATRIX, client=_FakeClient(), task_name="tamper-test")
    receipt_path = tmp_path / "receipt.json"
    job.save(receipt_path)
    payload = json.loads(receipt_path.read_text())
    payload["matrix"][0][1] = 2.0
    payload["matrix"][1][0] = 2.0
    receipt_path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="matrix identity"):
        restore_kaiwu_job(receipt_path, client=_FakeClient())


@pytest.mark.parametrize(
    "encoded",
    (
        '{"matrix": [[0, 1], [1, 0]], "matrix": [[0, 2], [2, 0]], "receipt": {}}',
        '{"matrix": [[0, 1], [1, 0]], "receipt": {"task_name": "a", "task_name": "b"}}',
    ),
)
def test_restore_rejects_duplicate_json_keys(tmp_path: Path, encoded: str) -> None:
    receipt_path = tmp_path / "duplicate.json"
    receipt_path.write_text(encoded)
    receipt_path.chmod(0o600)

    with pytest.raises(ValueError, match="duplicate object keys"):
        restore_kaiwu_job(receipt_path, client=_FakeClient())


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("task_name", None, "empty task name"),
        ("matrix_size", True, "matrix size"),
        ("mode", ["sampling"], "unsupported task mode"),
        ("requested_samples", True, "must be positive"),
        ("requested_samples", 9, "between 10 and 2000"),
        ("project_no", 7, "invalid project number"),
        ("project_no", "   ", "invalid project number"),
        ("submitted_at", "2026-10-05T00:00:00", "aware UTC"),
        ("submitted_at", "2026-10-05T08:00:00+08:00", "aware UTC"),
        ("provider_task_id", 7, "invalid provider_task_id"),
        ("provider_task_id", "   ", "invalid provider_task_id"),
        ("provider_target", 7, "invalid provider_target"),
        ("provider_target", "   ", "invalid provider_target"),
    ),
)
def test_restore_rejects_malformed_receipt_fields(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    job = submit_kaiwu_task(
        _MATRIX,
        client=_FakeClient(),
        task_name="field-test",
        mode="sampling",
        requested_samples=10,
    )
    receipt_path = tmp_path / "receipt.json"
    job.save(receipt_path)
    payload = json.loads(receipt_path.read_text())
    payload["receipt"][field] = value
    receipt_path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match=message):
        restore_kaiwu_job(receipt_path, client=_FakeClient())


@pytest.mark.parametrize("unsafe_kind", ("public", "symlink"))
def test_restore_rejects_unsafe_receipt_file(tmp_path: Path, unsafe_kind: str) -> None:
    job = submit_kaiwu_task(_MATRIX, client=_FakeClient(), task_name="unsafe-restore")
    receipt_path = tmp_path / "receipt.json"
    job.save(receipt_path)
    if unsafe_kind == "public":
        receipt_path.chmod(0o644)
    else:
        target = tmp_path / "receipt-target.json"
        receipt_path.rename(target)
        receipt_path.symlink_to(target.name)

    with pytest.raises(ValueError, match="private, regular, non-symlink"):
        restore_kaiwu_job(receipt_path, client=_FakeClient())


@pytest.mark.parametrize(
    "samples,energies,error",
    (
        (((1, 0),), (0.0,), r"only -1 or \+1"),
        (((True, -1),), (2.0,), r"only -1 or \+1"),
        (((1,),), (0.0,), "width differs"),
        (((1, -1),), (999.0,), "energy failed"),
        (((1, -1),), ("2.0",), "finite real numbers"),
        (((1, -1),), (), "counts differ"),
    ),
)
def test_result_validation_fails_closed(
    samples: tuple[tuple[object, ...], ...],
    energies: tuple[object, ...],
    error: str,
) -> None:
    class MalformedClient(_FakeClient):
        def fetch_result(
            self, receipt: KaiwuTaskReceipt, matrix: FrozenIsingMatrix
        ) -> KaiwuTaskResult:
            del matrix
            return KaiwuTaskResult(
                receipt,
                samples,  # type: ignore[arg-type]
                energies,  # type: ignore[arg-type]
                "Completed",
                {"fallback_occurred": False},
            )

    job = submit_kaiwu_task(
        _MATRIX,
        client=MalformedClient(),
        task_name="malformed",
        requested_samples=10,
    )

    with pytest.raises(RuntimeError, match=error):
        job.result()


def test_result_rejects_receipt_identity_mismatch() -> None:
    class WrongReceiptClient(_FakeClient):
        def fetch_result(
            self, receipt: KaiwuTaskReceipt, matrix: FrozenIsingMatrix
        ) -> KaiwuTaskResult:
            del matrix
            wrong = replace(receipt, task_name="different")
            return KaiwuTaskResult(
                wrong,
                ((1, -1),),
                (2.0,),
                "Completed",
                {"fallback_occurred": False},
            )

    job = submit_kaiwu_task(
        _MATRIX,
        client=WrongReceiptClient(),
        task_name="expected",
    )

    with pytest.raises(RuntimeError, match="does not match"):
        job.result()


@pytest.mark.parametrize(
    ("raw_status", "metadata", "error"),
    (
        ("Failed", {"fallback_occurred": False}, "non-success"),
        (7, {"fallback_occurred": False}, "invalid provider status"),
        ("Completed", None, "metadata must be a mapping"),
        ("Completed", {}, "fallback_occurred=false"),
        ("Completed", {"fallback_occurred": True}, "fallback_occurred=false"),
    ),
)
def test_result_requires_success_and_explicit_no_fallback(
    raw_status: object,
    metadata: object,
    error: str,
) -> None:
    class EvidenceClient(_FakeClient):
        def fetch_result(
            self, receipt: KaiwuTaskReceipt, matrix: FrozenIsingMatrix
        ) -> KaiwuTaskResult:
            del matrix
            return KaiwuTaskResult(
                receipt,
                ((1, -1),),
                (2.0,),
                raw_status,  # type: ignore[arg-type]
                metadata,  # type: ignore[arg-type]
            )

    job = submit_kaiwu_task(
        _MATRIX,
        client=EvidenceClient(),
        task_name="evidence",
    )

    with pytest.raises(RuntimeError, match=error):
        job.result()


@pytest.mark.parametrize("requested_samples", (1, 9, 2001))
def test_sampling_limits_fail_before_submission(requested_samples: int) -> None:
    client = _FakeClient()

    with pytest.raises(ValueError, match="between 10 and 2000"):
        submit_kaiwu_task(
            _MATRIX,
            client=client,
            task_name="invalid-sample-count",
            mode="sampling",
            requested_samples=requested_samples,
        )

    assert client.submit_calls == 0


def test_submission_receipt_must_match_request() -> None:
    class WrongSubmissionClient(_FakeClient):
        def submit(
            self,
            matrix: FrozenIsingMatrix,
            *,
            task_name: str,
            mode: KaiwuTaskMode,
            requested_samples: int,
            project_no: str | None,
        ) -> KaiwuTaskReceipt:
            receipt = super().submit(
                matrix,
                task_name=task_name,
                mode=mode,
                requested_samples=requested_samples,
                project_no=project_no,
            )
            return replace(receipt, task_name="different")

    with pytest.raises(RuntimeError, match="does not match the request"):
        submit_kaiwu_task(
            _MATRIX,
            client=WrongSubmissionClient(),
            task_name="expected",
        )


def test_submission_rejects_non_receipt_client_response() -> None:
    class MalformedReceiptClient(_FakeClient):
        def submit(  # type: ignore[override]
            self,
            matrix: FrozenIsingMatrix,
            *,
            task_name: str,
            mode: KaiwuTaskMode,
            requested_samples: int,
            project_no: str | None,
        ) -> object:
            del matrix, task_name, mode, requested_samples, project_no
            self.submit_calls += 1
            return None

    client = MalformedReceiptClient()

    with pytest.raises(RuntimeError, match="invalid task receipt"):
        submit_kaiwu_task(_MATRIX, client=client, task_name="malformed-receipt")

    assert client.submit_calls == 1


def test_invalid_mode_type_fails_before_submission() -> None:
    client = _FakeClient()

    with pytest.raises(ValueError, match="mode must"):
        submit_kaiwu_task(
            _MATRIX,
            client=client,
            task_name="invalid-mode",
            mode=["sampling"],  # type: ignore[arg-type]
        )

    assert client.submit_calls == 0


@pytest.mark.parametrize(
    "matrix",
    (
        (),
        ((0.0, 1.0),),
        ((0.0, 1.0), (2.0, 0.0)),
        ((0.0, float("nan")), (float("nan"), 0.0)),
        ((False,),),
        (("0.0", "1.0"), ("1.0", "0.0")),
        ((np.str_("0.0"), np.str_("1.0")), (np.str_("1.0"), np.str_("0.0"))),
        (
            (np.complex64(0.0), np.complex64(1.0)),
            (np.complex64(1.0), np.complex64(0.0)),
        ),
        (
            (np.complex128(0.0), np.complex128(1.0)),
            (np.complex128(1.0), np.complex128(0.0)),
        ),
    ),
)
def test_invalid_matrix_fails_before_submission(matrix: object) -> None:
    client = _FakeClient()

    with pytest.raises(ValueError):
        submit_kaiwu_task(
            matrix,  # type: ignore[arg-type]
            client=client,
            task_name="invalid",
        )

    assert client.submit_calls == 0


@pytest.mark.parametrize("dtype", (np.int8, np.int64, np.float32, np.float64))
def test_numpy_real_matrix_scalars_remain_supported(dtype: type[np.generic]) -> None:
    client = _FakeClient()
    matrix = ((dtype(0), dtype(1)), (dtype(1), dtype(0)))

    job = submit_kaiwu_task(matrix, client=client, task_name="numpy-real")

    assert job.receipt.matrix_size == 2
    assert client.submit_calls == 1

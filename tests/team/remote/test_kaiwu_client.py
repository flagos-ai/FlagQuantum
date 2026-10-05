from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar
from unittest.mock import Mock

import numpy as np
import pytest

import flagquantum.remote.kaiwu.client as client_module
from flagquantum.remote.kaiwu import (
    KaiwuSDKClient,
    KaiwuSDKEnvironment,
    KaiwuSDKError,
    restore_kaiwu_job,
    submit_kaiwu_task,
)

pytestmark = pytest.mark.unit

_MATRIX = ((0.0, 1.0), (1.0, 0.0))


class _FakeOptimizer:
    responses: ClassVar[list[object]] = []
    created_options: ClassVar[list[dict[str, object]]] = []
    solve_calls: ClassVar[int] = 0
    result_calls: ClassVar[int] = 0

    def __init__(self, **options: object) -> None:
        self.created_options.append(options)

    def solve(self, matrix: np.ndarray) -> object:
        assert matrix.tolist() == [[0.0, 1.0], [1.0, 0.0]]
        type(self).solve_calls += 1
        if self.responses:
            return self.responses.pop(0)
        return np.array([[1, -1]] * 10, dtype=np.int8)

    def get_task_result(self, matrix: np.ndarray) -> dict[str, object]:
        assert matrix.tolist() == [[0.0, 1.0], [1.0, 0.0]]
        type(self).result_calls += 1
        return {
            "task_id": "provider-task-secret-value",
            "machine_name": "provider-target-secret-value",
            "status": "completed",
            "solutions": np.array([[1, -1]] * 10, dtype=np.int8),
        }


@pytest.fixture(autouse=True)
def _reset_fake() -> None:
    _FakeOptimizer.responses = []
    _FakeOptimizer.created_options = []
    _FakeOptimizer.solve_calls = 0
    _FakeOptimizer.result_calls = 0


def _client(
    monkeypatch: pytest.MonkeyPatch, checkpoint_dir: Path
) -> tuple[KaiwuSDKClient, object]:
    manager = SimpleNamespace(save_dir="original")
    module = SimpleNamespace(
        common=SimpleNamespace(CheckpointManager=manager),
        cim=SimpleNamespace(
            CIMOptimizer=_FakeOptimizer,
            TaskMode=SimpleNamespace(OPTIMIZATION="optimization", SAMPLING="sampling"),
        ),
    )
    monkeypatch.setattr(
        client_module,
        "initialize_kaiwu_license",
        Mock(return_value=KaiwuSDKEnvironment("1.3.1", "3.10.18")),
    )
    monkeypatch.setattr(client_module, "_load_kaiwu_module", Mock(return_value=module))
    return KaiwuSDKClient(checkpoint_dir=checkpoint_dir), manager


def test_client_rejects_unsafe_checkpoint_directory_before_license(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    initializer = Mock()
    monkeypatch.setattr(client_module, "initialize_kaiwu_license", initializer)
    public = tmp_path / "public"
    public.mkdir(mode=0o755)
    public.chmod(0o755)

    with pytest.raises(ValueError, match="group or others"):
        KaiwuSDKClient(checkpoint_dir=public)

    target = tmp_path / "private"
    target.mkdir(mode=0o700)
    link = tmp_path / "linked"
    link.symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError, match="non-symlink"):
        KaiwuSDKClient(checkpoint_dir=link)

    initializer.assert_not_called()


def test_submit_and_poll_reuse_documented_task_identity(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _FakeOptimizer.responses = [None, np.array([[1, -1]] * 10, dtype=np.int8)]
    client, manager = _client(monkeypatch, tmp_path)

    job = submit_kaiwu_task(
        _MATRIX,
        client=client,
        task_name="same-task",
        mode="sampling",
        requested_samples=10,
        project_no="CPQC-test",
    )

    assert job.status() == "succeeded"
    result = job.result()
    assert len(result.samples) == 10
    assert _FakeOptimizer.solve_calls == 2
    assert len(_FakeOptimizer.created_options) == 1
    assert _FakeOptimizer.created_options[0]["task_name"] == "same-task"
    assert _FakeOptimizer.created_options[0]["wait"] is False
    assert manager.save_dir == "original"
    recovery_path = client.recovery_receipt_path(job.receipt)
    assert recovery_path.is_file()
    assert recovery_path.stat().st_mode & 0o777 == 0o600


def test_restore_queries_same_identity_without_client_submit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    first_client, _ = _client(monkeypatch, tmp_path)
    job = submit_kaiwu_task(
        _MATRIX,
        client=first_client,
        task_name="recoverable",
        mode="sampling",
        requested_samples=10,
    )
    receipt_path = tmp_path / "receipt.json"
    job.save(receipt_path)
    original_options = dict(_FakeOptimizer.created_options[0])

    _FakeOptimizer.created_options = []
    restored_client, _ = _client(monkeypatch, tmp_path)
    restored = restore_kaiwu_job(receipt_path, client=restored_client)
    result = restored.result()

    assert result.samples
    assert len(_FakeOptimizer.created_options) == 1
    assert _FakeOptimizer.created_options[0] == original_options


def test_restore_rejects_receipt_that_differs_from_authoritative_recovery(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    first_client, _ = _client(monkeypatch, tmp_path)
    job = submit_kaiwu_task(
        _MATRIX,
        client=first_client,
        task_name="forged-explicit-receipt",
        mode="sampling",
        requested_samples=10,
    )
    receipt_path = tmp_path / "receipt.json"
    job.save(receipt_path)
    payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    payload["receipt"]["provider_task_id"] = "forged-provider-task"
    receipt_path.write_text(json.dumps(payload), encoding="utf-8")
    solve_calls = _FakeOptimizer.solve_calls

    restored_client, _ = _client(monkeypatch, tmp_path)
    restored = restore_kaiwu_job(receipt_path, client=restored_client)
    with pytest.raises(KaiwuSDKError, match="authoritative recovery bundle"):
        restored.status()

    assert _FakeOptimizer.solve_calls == solve_calls


def test_restore_requires_authoritative_sdk_recovery_bundle(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    first_client, _ = _client(monkeypatch, tmp_path)
    job = submit_kaiwu_task(
        _MATRIX,
        client=first_client,
        task_name="missing-sdk-recovery",
        mode="sampling",
        requested_samples=10,
    )
    receipt_path = tmp_path / "receipt.json"
    job.save(receipt_path)
    first_client.recovery_receipt_path(job.receipt).unlink()
    solve_calls = _FakeOptimizer.solve_calls

    restored_client, _ = _client(monkeypatch, tmp_path)
    restored = restore_kaiwu_job(receipt_path, client=restored_client)
    with pytest.raises(KaiwuSDKError, match="recovery receipt is invalid"):
        restored.status()

    assert _FakeOptimizer.solve_calls == solve_calls


def test_new_client_reuses_preexisting_recovery_receipt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    first_client, _ = _client(monkeypatch, tmp_path)
    first = submit_kaiwu_task(
        _MATRIX,
        client=first_client,
        task_name="restart",
        mode="sampling",
        requested_samples=10,
    )

    second_client, _ = _client(monkeypatch, tmp_path)
    second = submit_kaiwu_task(
        _MATRIX,
        client=second_client,
        task_name="restart",
        mode="sampling",
        requested_samples=10,
    )

    assert second.receipt == first.receipt
    assert second_client.recovery_receipt_path(second.receipt) == (
        first_client.recovery_receipt_path(first.receipt)
    )


def test_conflicting_recovery_receipt_fails_before_sdk_operation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    client, _ = _client(monkeypatch, tmp_path)
    first = submit_kaiwu_task(
        _MATRIX,
        client=client,
        task_name="conflict",
        mode="sampling",
        requested_samples=10,
    )
    path = client.recovery_receipt_path(first.receipt)
    payload = json.loads(path.read_text())
    payload["receipt"]["requested_samples"] = 11
    path.write_text(json.dumps(payload))
    solve_calls = _FakeOptimizer.solve_calls

    restarted, _ = _client(monkeypatch, tmp_path)
    with pytest.raises(KaiwuSDKError, match="conflicts"):
        submit_kaiwu_task(
            _MATRIX,
            client=restarted,
            task_name="conflict",
            mode="sampling",
            requested_samples=10,
        )

    assert _FakeOptimizer.solve_calls == solve_calls


@pytest.mark.parametrize("unsafe_kind", ("public", "symlink"))
def test_unsafe_recovery_receipt_fails_before_sdk_operation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, unsafe_kind: str
) -> None:
    client, _ = _client(monkeypatch, tmp_path)
    first = submit_kaiwu_task(
        _MATRIX,
        client=client,
        task_name=f"unsafe-{unsafe_kind}",
        mode="sampling",
        requested_samples=10,
    )
    path = client.recovery_receipt_path(first.receipt)
    if unsafe_kind == "public":
        path.chmod(0o644)
    else:
        target = path.with_name(f"{path.name}.target")
        path.rename(target)
        path.symlink_to(target.name)
    solve_calls = _FakeOptimizer.solve_calls

    restarted, _ = _client(monkeypatch, tmp_path)
    with pytest.raises(KaiwuSDKError, match="recovery receipt is invalid"):
        submit_kaiwu_task(
            _MATRIX,
            client=restarted,
            task_name=f"unsafe-{unsafe_kind}",
            mode="sampling",
            requested_samples=10,
        )

    assert _FakeOptimizer.solve_calls == solve_calls


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("provider_task_id", "forged-provider-task"),
        ("provider_target", "forged-provider-target"),
        ("submitted_at", "not-a-timestamp"),
        ("submitted_at", "2026-10-05T12:00:00+08:00"),
    ),
)
def test_recovery_receipt_rejects_forged_provider_evidence_before_sdk_operation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    field: str,
    value: str,
) -> None:
    client, _ = _client(monkeypatch, tmp_path)
    first = submit_kaiwu_task(
        _MATRIX,
        client=client,
        task_name=f"forged-{field}",
        mode="sampling",
        requested_samples=10,
    )
    path = client.recovery_receipt_path(first.receipt)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["receipt"][field] = value
    path.write_text(json.dumps(payload), encoding="utf-8")
    solve_calls = _FakeOptimizer.solve_calls

    restarted, _ = _client(monkeypatch, tmp_path)
    with pytest.raises(KaiwuSDKError, match="recovery receipt is invalid"):
        submit_kaiwu_task(
            _MATRIX,
            client=restarted,
            task_name=f"forged-{field}",
            mode="sampling",
            requested_samples=10,
        )

    assert _FakeOptimizer.solve_calls == solve_calls


def test_recovery_receipt_rejects_undeclared_top_level_fields(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    client, _ = _client(monkeypatch, tmp_path)
    first = submit_kaiwu_task(
        _MATRIX,
        client=client,
        task_name="extra-recovery-field",
        mode="sampling",
        requested_samples=10,
    )
    path = client.recovery_receipt_path(first.receipt)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["credentials"] = "must-not-be-accepted"
    path.write_text(json.dumps(payload), encoding="utf-8")
    solve_calls = _FakeOptimizer.solve_calls

    restarted, _ = _client(monkeypatch, tmp_path)
    with pytest.raises(KaiwuSDKError, match="recovery receipt is invalid"):
        submit_kaiwu_task(
            _MATRIX,
            client=restarted,
            task_name="extra-recovery-field",
            mode="sampling",
            requested_samples=10,
        )

    assert _FakeOptimizer.solve_calls == solve_calls


def test_result_is_independently_scored_and_marks_evidence_gaps(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    client, _ = _client(monkeypatch, tmp_path)
    job = submit_kaiwu_task(
        _MATRIX,
        client=client,
        task_name="evidence-gap",
        mode="sampling",
        requested_samples=10,
    )

    result = job.result()

    assert result.energies == (2.0,) * 10
    assert result.metadata["fallback_occurred"] is False
    assert result.metadata["provider_task_id_available"] is False
    assert result.metadata["provider_target_available"] is False
    assert result.receipt.provider_task_id is None
    assert result.receipt.provider_target is None
    schema = result.metadata["provider_result_schema"]
    assert schema["available"] is True
    assert schema["result"]["fields"] == [
        "machine_name",
        "solutions",
        "status",
        "task_id",
    ]
    assert schema["result"]["field_schemas"]["task_id"] == {
        "type": "builtins.str",
        "length": len("provider-task-secret-value"),
    }
    assert schema["result"]["field_schemas"]["solutions"] == {
        "type": "numpy.ndarray",
        "shape": [10, 2],
        "dtype": "int8",
    }
    encoded = json.dumps(schema)
    assert "provider-task-secret-value" not in encoded
    assert "provider-target-secret-value" not in encoded
    assert _FakeOptimizer.result_calls == 1


def test_result_schema_failure_is_redacted_and_nonfatal(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def failing_result(self: _FakeOptimizer, matrix: np.ndarray) -> object:
        del self, matrix
        raise RuntimeError("provider-task-secret sdk-code-secret")

    monkeypatch.setattr(_FakeOptimizer, "get_task_result", failing_result)
    client, _ = _client(monkeypatch, tmp_path)
    job = submit_kaiwu_task(
        _MATRIX,
        client=client,
        task_name="schema-failure",
        mode="sampling",
        requested_samples=10,
    )

    result = job.result()

    assert result.samples
    assert result.metadata["provider_result_schema"] == {
        "available": False,
        "reason": "inspection_failed",
    }
    encoded = json.dumps(result.metadata)
    assert "provider-task-secret" not in encoded
    assert "sdk-code-secret" not in encoded


def test_same_matrix_with_different_task_name_creates_distinct_sdk_identity(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    client, _ = _client(monkeypatch, tmp_path)

    submit_kaiwu_task(
        _MATRIX,
        client=client,
        task_name="first",
        mode="sampling",
        requested_samples=10,
    )
    submit_kaiwu_task(
        _MATRIX,
        client=client,
        task_name="second",
        mode="sampling",
        requested_samples=10,
    )

    assert [options["task_name"] for options in _FakeOptimizer.created_options] == [
        "first",
        "second",
    ]


@pytest.mark.parametrize(
    ("response", "message"),
    (
        (np.array([[1, 0]] * 10), "spin domain"),
        (np.array([[1, -1, 1]] * 10), "solution shape"),
    ),
)
def test_malformed_sdk_solution_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    response: np.ndarray,
    message: str,
) -> None:
    _FakeOptimizer.responses = [response]
    client, _ = _client(monkeypatch, tmp_path)

    with pytest.raises(KaiwuSDKError, match=message):
        submit_kaiwu_task(
            _MATRIX,
            client=client,
            task_name="malformed",
            mode="sampling",
            requested_samples=10,
        )


def test_vendor_operation_error_is_redacted_and_checkpoint_is_restored(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _FakeOptimizer.responses = [RuntimeError("user-secret sdk-code-secret")]

    original_solve = _FakeOptimizer.solve

    def failing_solve(self: _FakeOptimizer, matrix: np.ndarray) -> object:
        del self, matrix
        raise _FakeOptimizer.responses.pop(0)

    monkeypatch.setattr(_FakeOptimizer, "solve", failing_solve)
    client, manager = _client(monkeypatch, tmp_path)

    with pytest.raises(KaiwuSDKError) as caught:
        submit_kaiwu_task(
            _MATRIX,
            client=client,
            task_name="redacted",
            mode="sampling",
            requested_samples=10,
        )

    assert "user-secret" not in str(caught.value)
    assert "sdk-code-secret" not in str(caught.value)
    assert caught.value.__cause__ is None
    assert manager.save_dir == "original"
    recovery_paths = list(tmp_path.glob("flagquantum-kaiwu-*.json"))
    assert len(recovery_paths) == 1
    encoded = recovery_paths[0].read_text()
    assert "user-secret" not in encoded
    assert "sdk-code-secret" not in encoded
    assert json.loads(encoded)["receipt"]["task_name"] == "redacted"
    monkeypatch.setattr(_FakeOptimizer, "solve", original_solve)

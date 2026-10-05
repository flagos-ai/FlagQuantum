from __future__ import annotations

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

    def __init__(self, **options: object) -> None:
        self.created_options.append(options)

    def solve(self, matrix: np.ndarray) -> object:
        assert matrix.tolist() == [[0.0, 1.0], [1.0, 0.0]]
        type(self).solve_calls += 1
        if self.responses:
            return self.responses.pop(0)
        return np.array([[1, -1]] * 10, dtype=np.int8)


@pytest.fixture(autouse=True)
def _reset_fake() -> None:
    _FakeOptimizer.responses = []
    _FakeOptimizer.created_options = []
    _FakeOptimizer.solve_calls = 0


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
    monkeypatch.setattr(_FakeOptimizer, "solve", original_solve)

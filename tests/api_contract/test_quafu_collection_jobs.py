"""SDK-backed Quafu collection submission and restoration contracts."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import flagquantum as fq
from flagquantum.errors import ExecutionError, ValidationError
from flagquantum.remote import jobs
from flagquantum.remote.qpu import _quafu_sdk
from flagquantum.remote.qpu.quafu import QuafuProvider

pytestmark = pytest.mark.unit


class _SdkCollectionJob:
    def __init__(self) -> None:
        self.id = "sdk-collection-123"
        self.n_tasks = 3
        self.state = "Finished"
        self.result_rows: list[SimpleNamespace] | None = None

    def status(self) -> str:
        return self.state

    def results(self, timeout: float | None = None) -> list[SimpleNamespace]:
        assert timeout == 0
        if self.result_rows is not None:
            return self.result_rows
        raw = ({"001": 1024}, {"010": 1024}, {"100": 1024})
        return [
            SimpleNamespace(
                seq=index,
                status="Finished",
                task_id=f"{self.id}-{index}",
                target="Baihua",
                backend="Baihua",
                bit_order="c0_rightmost",
                n_clbits=3,
                shots_requested=1024,
                shots_returned=1024,
                counts=counts,
                measurements=None,
                executed_qasm=None,
                compiler={"name": "qiskit", "version": "2.5"},
                calibration_id="cal-1",
                error=None,
            )
            for index, counts in enumerate(raw)
        ]

    def cancel(self) -> dict[str, object]:
        return {"status": "Cancelled", "cancelled_now": 3}


class _SdkCollectionClient:
    def __init__(self) -> None:
        self.job_handle = _SdkCollectionJob()
        self.submissions: list[tuple[object, dict[str, object]]] = []
        self.restores: list[str] = []

    def metadata(self, target: str) -> dict[str, object]:
        assert target == "Baihua"
        return {
            "n_qubits": 156,
            "basis_gates": ["h", "rx", "ry", "rz", "cz"],
            "coupling_map": [[17, 18]],
            "limits": {"max_tasks_per_job": 500},
            "calibration_id": "cal-1",
        }

    def submit(self, circuits: object, **kwargs: object) -> _SdkCollectionJob:
        self.submissions.append((circuits, kwargs))
        return self.job_handle

    def job(self, job_id: str) -> _SdkCollectionJob:
        self.restores.append(job_id)
        assert job_id == self.job_handle.id
        return self.job_handle


def _install(monkeypatch: pytest.MonkeyPatch, sdk_client: _SdkCollectionClient) -> None:
    class Client(QuafuProvider):
        def __init__(self, **kwargs: object) -> None:
            super().__init__(
                _sdk_adapter=_quafu_sdk._QuafuSdkAdapter.for_submission(
                    client=sdk_client
                ),
                **kwargs,
            )

    monkeypatch.setattr(jobs, "QuafuProvider", Client)


def test_collection_preserves_public_submit_restore_and_result_contract(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    sdk_client = _SdkCollectionClient()
    _install(monkeypatch, sdk_client)
    circuits = [fq.Circuit(3).x(qubit) for qubit in range(3)]

    job = fq.submit(
        circuits,
        target="quafu:Baihua",
        shots=1024,
        outputs=fq.counts(name="readout"),
        name="sdk asymmetric collection",
    )

    programs, options = sdk_client.submissions[0]
    assert isinstance(programs, list) and len(programs) == 3
    assert options == {
        "target": "Baihua",
        "shots": 1024,
        "compiler": "qiskit",
        "name": "sdk asymmetric collection",
    }
    receipt = tmp_path / "sdk-collection.receipt.json"
    job.save(receipt)
    saved = json.loads(receipt.read_text())
    assert saved["submission_identity"]["quafu_protocol"] == "quafu_sdk_v1"
    restored = fq.restore_job(receipt)
    assert restored.result().counts == [
        {"100": 1024},
        {"010": 1024},
        {"001": 1024},
    ]
    assert len(sdk_client.submissions) == 1
    assert sdk_client.restores == ["sdk-collection-123"]


def test_collection_partial_failure_never_becomes_partial_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sdk_client = _SdkCollectionClient()
    _install(monkeypatch, sdk_client)
    job = fq.submit(
        [fq.Circuit(3).x(qubit) for qubit in range(3)],
        target="quafu:Baihua",
        shots=1024,
    )
    sdk_client.job_handle.state = "PartiallyFinished"

    with pytest.raises(ExecutionError, match="failed.*PartiallyFinished"):
        job.result()

    assert len(sdk_client.submissions) == 1


def test_collection_rejects_duplicate_child_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sdk_client = _SdkCollectionClient()
    _install(monkeypatch, sdk_client)
    job = fq.submit(
        [fq.Circuit(3).x(qubit) for qubit in range(3)],
        target="quafu:Baihua",
        shots=1024,
    )
    rows = sdk_client.job_handle.results(timeout=0)
    rows[1].seq = 0
    sdk_client.job_handle.result_rows = rows

    with pytest.raises(ExecutionError, match="Duplicate or invalid"):
        job.result()


def test_collection_does_not_retry_unexpected_accepted_task_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sdk_client = _SdkCollectionClient()
    sdk_client.job_handle.n_tasks = 2
    _install(monkeypatch, sdk_client)

    with pytest.raises(RuntimeError, match="unexpected task count"):
        fq.submit(
            [fq.Circuit(3).x(qubit) for qubit in range(3)],
            target="quafu:Baihua",
            shots=1024,
        )

    assert len(sdk_client.submissions) == 1


def test_retired_collection_receipt_is_not_restored_or_resubmitted(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    sdk_client = _SdkCollectionClient()
    _install(monkeypatch, sdk_client)
    job = fq.submit(
        [fq.Circuit(3).x(qubit) for qubit in range(3)],
        target="quafu:Baihua",
        shots=1024,
    )
    receipt = tmp_path / "retired.receipt.json"
    job.save(receipt)
    raw = json.loads(receipt.read_text())
    raw["submission_identity"]["quafu_protocol"] = "task_api_v1"
    receipt.write_text(json.dumps(raw))

    with pytest.raises((ValueError, ValidationError), match="Invalid|retired|identity"):
        fq.restore_job(receipt)

    assert len(sdk_client.submissions) == 1

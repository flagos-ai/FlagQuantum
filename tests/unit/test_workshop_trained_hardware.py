from __future__ import annotations

import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

pytestmark = pytest.mark.unit
NOTEBOOK = (
    Path(__file__).resolve().parents[2]
    / "workshops/flagos2026/reference_notebooks/cloud/13_trained_hardware.ipynb"
)


class FakeExecutionError(RuntimeError):
    pass


class FakeJob:
    id = "existing-task"
    raw_status = "Queued"

    def __init__(self) -> None:
        self.outcome = "pending"

    def save(self, path: Path) -> None:
        with path.open("x") as stream:
            json.dump({"id": self.id}, stream)

    def wait(self, *, timeout: float, poll_interval: float) -> SimpleNamespace:
        assert timeout == 30 and poll_interval == 3
        if self.outcome == "pending":
            raise TimeoutError("Still queued")
        if self.outcome == "failed":
            self.raw_status = "Failed"
            raise FakeExecutionError("Backend failed")
        return SimpleNamespace(counts=[{"0": 640, "1": 384}])


def _cell(marker: str) -> str:
    notebook = json.loads(NOTEBOOK.read_text())
    return next(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code" and marker in "".join(cell["source"])
    )


@pytest.fixture
def scope(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    monkeypatch.setenv("QUAFU_API_KEY", "test-only-not-a-credential")
    errors = ModuleType("flagquantum.errors")
    errors.__dict__["ExecutionError"] = FakeExecutionError
    monkeypatch.setitem(sys.modules, "flagquantum.errors", errors)
    job = FakeJob()
    submissions: list[dict[str, object]] = []

    def submit(*args: object, **kwargs: object) -> FakeJob:
        submissions.append(kwargs)
        return job

    def restore(path: Path) -> FakeJob:
        assert json.loads(path.read_text())["id"] == job.id
        return job

    return {
        "json": json,
        "OUTPUTS": tmp_path,
        "fq": SimpleNamespace(submit=submit, restore_job=restore),
        "trained_circuit": object(),
        "settings": {"circuit_hash": "frozen-hash", "shots": 1024},
        "circuit_hash": "frozen-hash",
        "shots": 1024,
        "submissions": submissions,
        "fake_job": job,
    }


def _submit(scope: dict[str, Any]) -> None:
    source = _cell("submit_trained_circuit = False")
    exec(
        source.replace(
            "submit_trained_circuit = False", "submit_trained_circuit = True"
        ),
        scope,
    )


def test_pending_task_is_saved_and_restored_without_resubmission(
    scope: dict[str, Any],
) -> None:
    _submit(scope)
    path = scope["OUTPUTS"] / "trained_hardware.json"
    assert json.loads(path.read_text())["task_id"] == "existing-task"
    exec(_cell("hardware_job.wait"), scope)
    record = json.loads(path.read_text())
    assert record["status"] == "pending"
    assert record["provider_status"] == "Queued"
    assert scope["hardware_counts"] is None
    _submit(scope)
    exec(_cell("hardware_job.wait"), scope)
    assert len(scope["submissions"]) == 1


def test_saved_task_can_finish_on_a_later_collection(scope: dict[str, Any]) -> None:
    _submit(scope)
    exec(_cell("hardware_job.wait"), scope)
    scope["fake_job"].outcome = "completed"
    exec(_cell("hardware_job.wait"), scope)
    assert scope["hardware_counts"] == {"0": 640, "1": 384}
    assert scope["hardware_record"]["status"] == "completed"
    assert len(scope["submissions"]) == 1


def test_backend_failure_is_recorded_and_raised(scope: dict[str, Any]) -> None:
    _submit(scope)
    scope["fake_job"].outcome = "failed"
    with pytest.raises(FakeExecutionError, match="Backend failed"):
        exec(_cell("hardware_job.wait"), scope)
    record = json.loads((scope["OUTPUTS"] / "trained_hardware.json").read_text())
    assert record["status"] == "execution_error"
    assert record["task_id"] == "existing-task"


@pytest.mark.parametrize("changed", ("circuit_hash", "shots", "task_id"))
def test_mismatched_saved_task_does_not_submit(
    scope: dict[str, Any], changed: str
) -> None:
    _submit(scope)
    path = scope["OUTPUTS"] / "trained_hardware.json"
    record = json.loads(path.read_text())
    record[changed] = "different"
    path.write_text(json.dumps(record))
    with pytest.raises(RuntimeError):
        _submit(scope)
    assert len(scope["submissions"]) == 1


def test_uncertain_submission_guard_prevents_retry(scope: dict[str, Any]) -> None:
    path = scope["OUTPUTS"] / "trained_hardware.json"
    record = dict(
        scope["settings"], status="submission_started", target="quafu:Dongling"
    )
    path.write_text(json.dumps(record))
    with pytest.raises(RuntimeError, match="do not resubmit"):
        _submit(scope)
    assert scope["submissions"] == []
    assert json.loads(path.read_text()) == record


def test_reconciled_completed_result_does_not_submit(scope: dict[str, Any]) -> None:
    path = scope["OUTPUTS"] / "trained_hardware.json"
    record = dict(
        scope["settings"],
        status="completed",
        target="quafu:Dongling",
        task_id="reconciled-task",
        counts={"0": 640, "1": 384},
    )
    path.write_text(json.dumps(record))
    _submit(scope)
    exec(_cell("hardware_job.wait"), scope)
    assert scope["hardware_counts"] == record["counts"]
    assert scope["submissions"] == []


@pytest.mark.parametrize("counts", (None, {}, {"0": 1}, {"0": 1024.0}, {"00": 1024}))
def test_invalid_reconciled_counts_do_not_submit(
    scope: dict[str, Any], counts: object
) -> None:
    path = scope["OUTPUTS"] / "trained_hardware.json"
    record = dict(
        scope["settings"],
        status="completed",
        target="quafu:Dongling",
        task_id="reconciled-task",
        counts=counts,
    )
    path.write_text(json.dumps(record))
    with pytest.raises(RuntimeError, match="invalid counts"):
        _submit(scope)
    assert scope["submissions"] == []


@pytest.mark.parametrize(
    "record",
    (
        None,
        {"status": "submission_started"},
        {"task_id": "existing-task", "status": "pending"},
    ),
)
def test_report_handles_absent_or_pending_task(
    scope: dict[str, Any], record: dict[str, str] | None
) -> None:
    source = _cell("report = dict(").split("report = dict(", 1)[1]
    scope.update(hardware_record=record, comparisons=[{"source": "local_sampling"}])
    exec("report = dict(" + source, scope)
    report = json.loads(
        (scope["OUTPUTS"] / "trained_measurement_report.json").read_text()
    )
    assert report["hardware_task_id"] == (record or {}).get("task_id")
    assert report["hardware_status"] == (
        "not_submitted" if record is None else record["status"]
    )

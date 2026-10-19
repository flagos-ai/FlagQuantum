"""Provider-neutral receipt recovery and conservative local submission exclusion."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event
from types import SimpleNamespace
from typing import Any

import pytest

import flagquantum as fq
from flagquantum.remote import _job_receipts, jobs
from flagquantum.remote.compute import JiudingCredentials
from flagquantum.remote.compute._native_job import NativeJiudingJobClient
from flagquantum.remote.qpu.quafu import QuafuProvider

pytestmark = pytest.mark.unit


@pytest.fixture(params=["quafu", "jiuding"])
def session(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> SimpleNamespace:
    calls: list[object] = []
    clients: list[object] = []
    errors: list[Exception] = []
    credentials = JiudingCredentials(access_key="private-ak", secret_key="private-sk")

    class Transport:
        def post_json(
            self, url: str, payload: dict, headers: dict, timeout: float
        ) -> dict:
            calls.append(payload)
            if errors:
                raise errors[0]
            return {"job_id": "job-123"}

        def get_json(self, url: str, headers: dict, timeout: float) -> dict:
            if url.endswith("/healthz"):
                return {"healthy": True}
            raise AssertionError("Submission/restoration must not poll")

    class Quafu(QuafuProvider):
        def __init__(self, **kwargs: Any) -> None:
            clients.append(self)
            super().__init__(
                api_key="private-api-key",
                task_server_url="https://quafu.test/api/v1",
                transport=Transport(),
                **kwargs,
            )

    class Jiuding(NativeJiudingJobClient):
        def __init__(
            self,
            *,
            project: str,
            queue: str | None = None,
            credentials: JiudingCredentials | None = None,
        ) -> None:
            clients.append(credentials)
            self.project = project
            self.queue = queue

        def submit_program(self, program: object, **kwargs: Any) -> dict:
            calls.append(kwargs)
            if errors:
                raise errors[0]
            return {
                "jobId": "job-123",
                "project": self.project,
                "queue": self.queue or "default-queue",
                "artifact_transport": "job_logs",
            }

    monkeypatch.setattr(jobs, "QuafuProvider", Quafu)
    monkeypatch.setattr(jobs, "NativeJiudingJobClient", Jiuding)
    options: dict[str, Any] = {
        "target": (
            f"{request.param}:cpu" if request.param == "jiuding" else "quafu:Baihua"
        ),
        "shots": 1024,
        "outputs": fq.counts(),
    }
    if request.param == "jiuding":
        options.update(
            image="private-image",
            project="test.project",
            queue="test-queue",
            credentials=credentials,
        )
    return SimpleNamespace(
        provider=request.param,
        calls=calls,
        clients=clients,
        errors=errors,
        options=options,
        circuit=fq.Circuit(3).x(0),
    )


def _submit(session: SimpleNamespace, path: Path, **changes: Any) -> jobs.RemoteJob:
    options = session.options | changes
    program = options.pop("program", session.circuit)
    return fq.submit(program, **options, receipt=path)


def test_matching_request_restores_without_submit_or_poll(
    session: SimpleNamespace, tmp_path: Path
) -> None:
    path = tmp_path / "job.json"
    job = _submit(session, path)
    before = path.read_bytes()
    restored = _submit(session, path, program=session.circuit.to_ir())
    assert restored.id == job.id == "job-123"
    assert restored.target == job.target
    assert path.read_bytes() == before
    assert len(session.calls) == 1
    journal = Path(str(path) + ".request.json")
    assert path.stat().st_mode & 0o777 == journal.stat().st_mode & 0o777 == 0o600
    saved = json.loads(journal.read_text())
    assert set(saved) == {"schema", "request_sha256", "receipt_sha256"}
    assert len(saved["request_sha256"]) == len(saved["receipt_sha256"]) == 64
    for secret in (
        "private-ak",
        "private-sk",
        "private-api-key",
        "private-image",
        "test.project",
    ):
        assert secret not in journal.read_text()
    for secret in ("private-ak", "private-sk", "private-api-key"):
        assert secret not in path.read_text()


@pytest.mark.parametrize(
    "change",
    [
        "program",
        "target",
        "shots",
        "outputs",
        "compiler",
        "target_qubits",
        "name",
        "image",
        "devices",
        "workspace",
        "project",
        "queue",
    ],
)
def test_changed_configuration_never_submits(
    session: SimpleNamespace, tmp_path: Path, change: str
) -> None:
    path = tmp_path / "job.json"
    _submit(session, path)
    changes = {
        "program": fq.Circuit(3).x(1),
        "target": session.options["target"] + "-other",
        "shots": 2048,
        "outputs": fq.counts(name="different"),
        "compiler": "qsteed",
        "target_qubits": [2, 1, 0],
        "name": "changed",
        "image": "changed-image",
        "devices": 2,
        "workspace": "changed-workspace",
        "project": "another.project",
        "queue": "other-queue",
    }
    with pytest.raises(ValueError, match="configuration changed"):
        _submit(session, path, **{change: changes[change]})
    assert len(session.calls) == 1


@pytest.mark.parametrize(
    "change",
    [
        "missing_receipt",
        "missing_journal",
        "corrupt_receipt",
        "corrupt_journal",
        "incomplete",
        "schema",
        "oversized",
        "digest",
        "request_digest",
        "receipt_digest_type",
        "extra",
        "array",
    ],
)
def test_invalid_records_never_fall_back(
    session: SimpleNamespace, tmp_path: Path, change: str
) -> None:
    path = tmp_path / "job.json"
    _submit(session, path)
    journal = Path(str(path) + ".request.json")
    if change == "missing_receipt":
        path.unlink()
    elif change == "missing_journal":
        journal.unlink()
    elif change == "corrupt_receipt":
        path.write_text("{broken")
    elif change == "corrupt_journal":
        journal.write_text("{broken")
    elif change == "oversized":
        journal.write_text(" " * 4097)
    else:
        raw = json.loads(journal.read_text())
        if change == "incomplete":
            raw["receipt_sha256"] = None
        elif change == "schema":
            raw["schema"] = "future"
        elif change == "request_digest":
            raw["request_sha256"] = 123
        elif change == "receipt_digest_type":
            raw["receipt_sha256"] = 123
        elif change == "extra":
            raw["extra"] = 1
        elif change == "array":
            raw = []
        else:
            raw["receipt_sha256"] = "broken"
        journal.write_text(json.dumps(raw))
    with pytest.raises((ValueError, RuntimeError)):
        _submit(session, path)
    assert len(session.calls) == 1


def test_lost_submission_response_is_not_retried(
    session: SimpleNamespace, tmp_path: Path
) -> None:
    path = tmp_path / "uncertain.json"
    session.errors.append(TimeoutError("response lost"))
    with pytest.raises(TimeoutError, match="response lost"):
        _submit(session, path)
    session.errors.clear()
    with pytest.raises(RuntimeError, match="reconcile"):
        _submit(session, path)
    assert len(session.calls) == 1 and not path.exists()


def test_restore_error_never_creates_a_new_job(
    session: SimpleNamespace, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "existing.json"
    _submit(session, path)

    def fail_restore(path: Path, *, credentials: object = None) -> jobs.RemoteJob:
        raise RuntimeError("credentials unavailable")

    monkeypatch.setattr(jobs, "restore_job", fail_restore)
    with pytest.raises(RuntimeError, match="credentials unavailable"):
        _submit(session, path)
    assert len(session.calls) == 1


def test_reservation_race_never_submits(
    session: SimpleNamespace, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = _job_receipts._reserve

    def reserve_twice(path: Path, journal: _job_receipts._Journal) -> None:
        original(path, journal)
        original(path, journal)

    monkeypatch.setattr(_job_receipts, "_reserve", reserve_twice)
    with pytest.raises(RuntimeError, match="Another attempt"):
        _submit(session, tmp_path / "raced.json")
    assert session.calls == []


@pytest.mark.parametrize("path", ["", 123])
def test_invalid_receipt_path_never_submits(
    session: SimpleNamespace, path: object
) -> None:
    with pytest.raises((TypeError, ValueError), match="receipt"):
        fq.submit(session.circuit, **session.options, receipt=path)
    assert session.calls == []


def test_symlink_receipt_is_not_followed(
    session: SimpleNamespace, tmp_path: Path
) -> None:
    path = tmp_path / "link.json"
    path.symlink_to(tmp_path / "missing.json")
    with pytest.raises(ValueError, match="symbolic links"):
        _submit(session, path)
    assert session.calls == []


def test_standalone_old_receipt_remains_readable(
    session: SimpleNamespace, tmp_path: Path
) -> None:
    path = tmp_path / "old.json"
    job = fq.submit(session.circuit, **session.options)
    job.save(path)
    with pytest.raises(ValueError, match="use fq.restore_job"):
        _submit(session, path)
    assert (
        fq.restore_job(path, credentials=session.options.get("credentials")).id
        == job.id
    )
    assert len(session.calls) == 1


def test_omitted_receipt_remains_a_new_submission(
    session: SimpleNamespace, tmp_path: Path
) -> None:
    fq.submit(session.circuit, **session.options)
    fq.submit(session.circuit, **session.options)
    assert len(session.calls) == 2 and not list(tmp_path.iterdir())


def test_failed_completion_keeps_the_guard(
    session: SimpleNamespace, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail_complete(path: Path, journal: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(_job_receipts, "_complete", fail_complete)
    path = tmp_path / "job.json"
    with pytest.raises(OSError, match="job-123.*do not resubmit"):
        _submit(session, path)
    assert path.exists()
    with pytest.raises(RuntimeError, match="reconcile"):
        _submit(session, path)
    assert len(session.calls) == 1


def test_overlapping_calls_cannot_submit_twice(
    session: SimpleNamespace, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    entered, release = Event(), Event()
    original = _job_receipts._reserve

    def reserve(path: Path, journal: _job_receipts._Journal) -> None:
        original(path, journal)
        entered.set()
        assert release.wait(timeout=10)

    monkeypatch.setattr(_job_receipts, "_reserve", reserve)
    path = tmp_path / "job.json"
    with ThreadPoolExecutor(max_workers=1) as executor:
        pending = executor.submit(_submit, session, path)
        try:
            assert entered.wait(timeout=10)
            with pytest.raises(RuntimeError, match="reconcile"):
                _submit(session, path)
            assert session.calls == []
        finally:
            release.set()
        assert pending.result(timeout=10).id == "job-123"
    assert len(session.calls) == 1


@pytest.mark.parametrize("session", ["jiuding"], indirect=True)
def test_jiuding_resolves_project_and_default_devices_without_storing_credentials(
    session: SimpleNamespace, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("JIUDING_PROJECT", "test.project")
    session.options.pop("project")
    path = tmp_path / "compute.json"
    first = _submit(session, path)
    replacement = JiudingCredentials(access_key="rotated-ak", secret_key="rotated-sk")
    restored = _submit(
        session, path, project="test.project", devices=1, credentials=replacement
    )
    assert restored.id == first.id and session.clients[-1] is replacement
    monkeypatch.setenv("JIUDING_PROJECT", "different.project")
    with pytest.raises(ValueError, match="configuration changed"):
        _submit(session, path)
    assert len(session.calls) == 1

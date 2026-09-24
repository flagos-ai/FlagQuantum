"""Detached jobs remain resumable and never resubmit during observation."""

import json
from unittest.mock import Mock

import pytest

import flagquantum as fq
from flagquantum.errors import CapabilityError, ExecutionError
from flagquantum.remote import jobs
from flagquantum.remote.compute import JiudingCredentials
from flagquantum.remote.qpu.quafu import QuafuProvider

pytestmark = pytest.mark.unit


class Transport:
    def __init__(self):
        self.posts = []
        self.gets = []
        self.state = "Transpiled"

    def post_json(self, url, payload, headers, timeout):
        self.posts.append(payload)
        return {"task_id": "job-123"}

    def get_json(self, url, headers, timeout):
        self.gets.append(url)
        if "/status/" in url:
            return self.state
        if "/result/" in url:
            return {"count": {"01": 1000, "00": 24}, "transpiled": "physical circuit"}
        if "/cancel/" in url:
            self.state = "Cancelled"
            return {}
        raise AssertionError(url)


@pytest.fixture
def quafu(monkeypatch):
    transport = Transport()

    class Client(QuafuProvider):
        def __init__(self, **kwargs):
            super().__init__(token="test-private-token", transport=transport, **kwargs)

    monkeypatch.setattr(jobs, "QuafuProvider", Client)
    return transport


def test_quafu_submit_result_and_restore(quafu, tmp_path):
    job = fq.submit(
        fq.Circuit(2).x(0),
        target="quafu:Baihua",
        shots=1024,
        outputs=fq.counts(name="readout"),
    )
    assert job.id == "job-123"
    assert job.target == "quafu:Baihua"
    assert quafu.gets == []
    assert job.status() == "queued"
    assert job.raw_status == "Transpiled"
    with pytest.raises(ExecutionError, match="queued"):
        job.result()
    assert not any("/result/" in url for url in quafu.gets)
    receipt = tmp_path / "job.json"
    job.save(receipt)
    assert "test-private-token" not in receipt.read_text()
    assert receipt.stat().st_mode & 0o777 == 0o600
    with pytest.raises(FileExistsError):
        job.save(receipt)
    restored = fq.restore_job(receipt)
    assert len(quafu.posts) == 1
    quafu.state = "Finished"
    result = restored.result()
    assert result.counts == [{"10": 1000, "00": 24}]
    assert result.measurements[0].metadata["name"] == "readout"
    assert result.provenance["service_compiler"] == "quarkcircuit"
    assert len(quafu.posts) == 1


def test_new_quafu_task_api_protocol_survives_receipt_restore(monkeypatch, tmp_path):
    class TaskApiTransport:
        def __init__(self):
            self.posts = []

        def post_json(self, url, payload, headers, timeout):
            self.posts.append(url)
            return {"job_id": "u-job-123", "status": "Queued"}

        def get_json(self, url, headers, timeout):
            if url.endswith("/healthz"):
                return {"healthy": True}
            if url.endswith("/status"):
                return {"status": "Finished"}
            if url.endswith("/results"):
                return {
                    "counts": [{"01": 1024}],
                    "shots_returned": [1024],
                    "bit_order": "c0_rightmost",
                }
            raise AssertionError(url)

    transport = TaskApiTransport()

    class Client(QuafuProvider):
        def __init__(self, **kwargs):
            super().__init__(
                api_key="qf_test",
                task_server_url="https://quafu.test/api/v1",
                transport=transport,
                **kwargs,
            )

    monkeypatch.setattr(jobs, "QuafuProvider", Client)
    job = fq.submit(fq.Circuit(2).x(0), target="quafu:Baihua", shots=1024)
    path = tmp_path / "task-api-job.json"
    job.save(path)

    receipt = json.loads(path.read_text())
    assert receipt["submission_identity"]["quafu_protocol"] == "task_api_v1"
    restored = fq.restore_job(path)
    assert restored.result().counts == [{"10": 1024}]
    assert transport.posts == ["https://quafu.test/api/v1/jobs"]


def test_wait_cancel_and_unknown_state(quafu):
    job = fq.submit(fq.Circuit(2), target="quafu:Baihua", shots=1024)
    quafu.state = "UnrecognizedProviderState"
    assert job.status() == "unknown"
    with pytest.raises(TimeoutError, match="no cancellation"):
        job.wait(timeout=0)
    assert len(quafu.posts) == 1
    assert not any("/cancel/" in url for url in quafu.gets)
    job.cancel()
    assert job.status() == "cancelled"
    with pytest.raises(ExecutionError, match="cancelled"):
        job.wait(timeout=1)
    quafu.state = "Failed"
    with pytest.raises(ExecutionError, match="failed"):
        job.result()
    quafu.state = "Finished"
    assert job.wait(timeout=0).counts == [{"10": 1000, "00": 24}]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"timeout": -1},
        {"timeout": float("nan")},
        {"timeout": float("inf")},
        {"poll_interval": 0},
    ],
)
def test_wait_validates_before_query(quafu, kwargs):
    job = fq.submit(fq.Circuit(2), target="quafu:Baihua", shots=1024)
    with pytest.raises(ValueError):
        job.wait(**kwargs)
    assert quafu.gets == []


def test_rejected_outputs_and_shots_never_submit(quafu):
    with pytest.raises(CapabilityError):
        fq.submit(
            fq.Circuit(2),
            target="quafu:Baihua",
            shots=1024,
            outputs=fq.expectation(fq.Z(0)),
        )
    with pytest.raises(ValueError):
        fq.submit(fq.Circuit(2), target="quafu:Baihua", shots=12)
    assert quafu.posts == []


def test_no_retry_after_uncertain_submission(quafu, monkeypatch):
    post = Mock(side_effect=TimeoutError("response lost"))
    monkeypatch.setattr(quafu, "post_json", post)
    with pytest.raises(TimeoutError):
        fq.submit(fq.Circuit(2), target="quafu:Baihua", shots=1024)
    assert post.call_count == 1


def test_receipt_validation_precedes_network(quafu, tmp_path):
    job = fq.submit(fq.Circuit(2), target="quafu:Baihua", shots=1024)
    path = tmp_path / "receipt.json"
    job.save(path)
    valid = json.loads(path.read_text())
    for field, value in [
        ("schema", "future"),
        ("target", "other:host"),
        ("n_wires", True),
        ("target_qubits", [1, 1]),
        ("id", ""),
    ]:
        path.write_text(json.dumps({**valid, field: value}))
        with pytest.raises(ValueError):
            fq.restore_job(path)
    assert quafu.gets == []
    assert len(quafu.posts) == 1


def test_jiuding_native_jobs_restore_without_resubmission(monkeypatch, tmp_path):
    from flagquantum.remote.compute._native_job import NativeJiudingJobClient

    calls = []
    state = ["Pending"]
    expected = fq.run(fq.Circuit(1))
    credentials = JiudingCredentials(
        access_key="session-ak", secret_key="session-secret"
    )
    clients = []

    class Client(NativeJiudingJobClient):
        def __init__(self, *, project, queue=None, credentials=None):
            self.project = project
            self.queue = queue
            self.credentials = credentials
            clients.append(self)

        def submit_program(self, *args, **kwargs):
            calls.append("submit")
            return {
                "jobId": "compute-123",
                "project": self.project,
                "queue": self.queue,
                "artifact_transport": "job_logs",
            }

        def read_result(self, receipt):
            return expected

        def status(self, receipt):
            return [
                {"id": "compute-123", "status": state[0]},
                {"id": "previous-attempt", "status": "Failed"},
            ]

        def cancel(self, receipt):
            calls.append("cancel")
            state[0] = "Cancelled"

    monkeypatch.setattr(jobs, "NativeJiudingJobClient", Client)
    job = fq.submit(
        fq.Circuit(1),
        target="jiuding:cpu",
        image="test-image",
        project="test.project",
        queue="test-queue",
        credentials=credentials,
    )
    assert job.status() == "queued"
    with pytest.raises(ExecutionError):
        job.result()
    path = tmp_path / "compute.json"
    job.save(path)
    assert "session-ak" not in path.read_text()
    assert "session-secret" not in path.read_text()
    restored = fq.restore_job(path, credentials=credentials)
    state[0] = "Succeed"
    assert restored.result() is expected
    assert calls.count("submit") == 1
    assert calls == ["submit"]
    assert len(clients) == 2
    assert all(client.credentials is credentials for client in clients)
    restored.cancel()
    assert restored.status() == "cancelled"


def test_jiuding_missing_image_fails_before_client(monkeypatch):
    client = Mock(side_effect=AssertionError("must not connect"))
    monkeypatch.setattr(jobs, "NativeJiudingJobClient", client)
    with pytest.raises(ValueError, match="requires image"):
        fq.submit(fq.Circuit(1), target="jiuding:cpu")
    client.assert_not_called()


def test_explicit_jiuding_credentials_are_rejected_for_quafu(quafu):
    credentials = JiudingCredentials(access_key="ak", secret_key="sk")
    with pytest.raises(TypeError, match="only to Jiuding"):
        fq.submit(
            fq.Circuit(2),
            target="quafu:Baihua",
            shots=1024,
            credentials=credentials,
        )
    assert quafu.posts == []


def test_wait_polls_without_resubmitting(quafu, monkeypatch):
    job = fq.submit(fq.Circuit(2), target="quafu:Baihua", shots=1024)
    sleeps = []

    def finish(delay):
        sleeps.append(delay)
        quafu.state = "Finished"

    monkeypatch.setattr(jobs.time, "sleep", finish)
    assert job.wait(timeout=10, poll_interval=0.1).counts == [{"10": 1000, "00": 24}]
    assert sleeps == [0.1]
    assert len(quafu.posts) == 1


def test_approved_job_state_and_receipt_contract(quafu, tmp_path):
    from pathlib import Path
    from typing import get_args

    contract = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "contracts/remote-jobs-v1-candidate.json"
        ).read_text()
    )
    assert list(get_args(jobs.JobStatus)) == contract["statuses"]
    job = fq.submit(fq.Circuit(2), target="quafu:Baihua", shots=1024)
    path = tmp_path / "job.json"
    job.save(path)
    receipt = json.loads(path.read_text())
    assert receipt["schema"] == contract["receipt_schema"]
    assert receipt["submission_identity"]["deployment_artifact_sha256"]


def test_wait_handles_delayed_quafu_result(quafu, monkeypatch):
    job = fq.submit(fq.Circuit(2), target="quafu:Baihua", shots=1024)
    quafu.state = "Finished"
    original = quafu.get_json
    result_reads = []

    def delayed(url, headers, timeout):
        if "/result/" in url:
            result_reads.append(url)
            if len(result_reads) == 1:
                return {}
        return original(url, headers, timeout)

    monkeypatch.setattr(quafu, "get_json", delayed)
    monkeypatch.setattr(jobs.time, "sleep", lambda delay: None)
    assert job.wait(timeout=5).counts == [{"10": 1000, "00": 24}]
    assert len(result_reads) == 2
    assert len(quafu.posts) == 1


@pytest.mark.parametrize(
    "options, error, message",
    [
        ({}, ValueError, "requires project"),
        ({"workspace": "old-workspace"}, TypeError, "project and queue"),
        ({"project": "invalid"}, ValueError, "project-set.project"),
    ],
)
def test_native_job_configuration_fails_before_network(
    monkeypatch, options, error, message
):
    from flagquantum.remote.compute._native_job import NativeJiudingJobClient

    monkeypatch.delenv("JIUDING_PROJECT", raising=False)
    monkeypatch.setattr(
        NativeJiudingJobClient,
        "_request",
        Mock(side_effect=AssertionError("must not connect")),
    )
    with pytest.raises(error, match=message):
        fq.submit(fq.Circuit(1), target="jiuding:cpu", image="test-image", **options)

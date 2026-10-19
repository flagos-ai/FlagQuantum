from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from flagquantum.remote._job_receipts import resume_or_submit

pytestmark = pytest.mark.unit
WORKSHOP = Path(__file__).resolve().parents[2] / "workshops/flagos2026"


class State:
    def abs(self) -> State:
        return self

    def square(self) -> State:
        return self

    def tolist(self) -> list[float]:
        return [0.75, 0, 0, 0, 0.25, 0, 0, 0]

    def reshape(self, size: int) -> State:
        assert size == -1
        return self


class Circuit:
    def __init__(self, qubits: int) -> None:
        assert qubits == 3

    def h(self, wire: int) -> Circuit:
        return self

    def cx(self, control: int, target: int) -> Circuit:
        return self

    def x(self, wire: int) -> Circuit:
        assert wire in range(3)
        return self

    def ry(self, wire: int, angle: float) -> Circuit:
        assert wire == 0
        return self

    def draw(self, *, format: str, show_all_qubits: bool) -> tuple[None, None]:
        assert format == "mpl" and show_all_qubits
        return None, None


class FakeJob:
    id = "saved-task"
    raw_status = "Queued"

    def __init__(self) -> None:
        self.outcome = "completed"

    def status(self) -> str:
        return "succeeded" if self.outcome == "completed" else "queued"

    def save(self, path: Path) -> None:
        with path.open("x") as stream:
            json.dump({"id": self.id}, stream)

    def wait(self, *, timeout: int) -> SimpleNamespace:
        assert timeout == 30
        if self.outcome == "pending":
            raise TimeoutError("Queued")
        if self.outcome == "failed":
            raise RuntimeError("Backend failed")
        return SimpleNamespace(state=State(), counts=[{"000": 768, "100": 256}])


@pytest.fixture
def setup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    start = tmp_path / "flagos2026/notebooks"
    start.mkdir(parents=True)
    monkeypatch.chdir(start)
    monkeypatch.syspath_prepend(str(WORKSHOP / "scripts"))
    monkeypatch.delenv("JIUDING_AK", raising=False)
    monkeypatch.delenv("QUAFU_TARGET", raising=False)
    exists = Path.exists
    monkeypatch.setattr(
        Path, "exists", lambda p: False if p == Path("/etc/accesskey") else exists(p)
    )
    for key, value in {
        "QUAFU_API_KEY": "test-only",
        "JIUDING_PROJECT": "test.project",
        "JIUDING_IMAGE": "verified:image",
        "JIUDING_QUEUE": "test-queue",
    }.items():
        monkeypatch.setenv(key, value)
    job = FakeJob()
    calls: list[dict[str, Any]] = []

    def submit(circuit: object, **kwargs: Any) -> FakeJob:
        path = Path(kwargs["receipt"])
        settings = {key: value for key, value in kwargs.items() if key != "receipt"}
        identity = hashlib.sha256(
            json.dumps(settings, sort_keys=True).encode()
        ).hexdigest()

        def create() -> FakeJob:
            calls.append(kwargs)
            job.target = kwargs["target"]
            return job

        return resume_or_submit(
            path,
            identity=identity,
            target=kwargs["target"],
            submitter=create,
            restore=restore,
        )

    def restore(path: Path) -> FakeJob:
        assert json.loads(path.read_text())["id"] == job.id
        return job

    fq = ModuleType("flagquantum")
    fq.__dict__.update(
        Circuit=Circuit,
        run=lambda circuit: SimpleNamespace(
            state=State(), to_statevector=lambda: State()
        ),
        submit=submit,
        restore_job=restore,
        counts=lambda: "counts-request",
    )
    monkeypatch.setitem(sys.modules, "flagquantum", fq)
    torch = ModuleType("torch")

    def assert_close(a: State, b: State) -> None:
        assert a.tolist() == b.tolist()

    torch.__dict__["testing"] = SimpleNamespace(assert_close=assert_close)
    monkeypatch.setitem(sys.modules, "torch", torch)
    matplotlib = ModuleType("matplotlib")
    pyplot = ModuleType("matplotlib.pyplot")
    for name in ("show", "bar", "xticks", "xlabel", "ylabel", "legend"):
        pyplot.__dict__[name] = lambda *args, **kwargs: None
    matplotlib.__dict__["pyplot"] = pyplot
    monkeypatch.setitem(sys.modules, "matplotlib", matplotlib)
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", pyplot)
    return SimpleNamespace(job=job, calls=calls, outputs=start.parent / "outputs")


def run_cells(
    filename: str, *, enabled: bool, language: str = "notebooks"
) -> dict[str, Any]:
    notebook = json.loads((WORKSHOP / language / filename).read_text())
    scope: dict[str, Any] = {}
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            source = "".join(cell["source"])
            if enabled:
                source = source.replace(
                    "submit_to_cloud = False", "submit_to_cloud = True"
                )
                source = source.replace(
                    "submit_to_hardware = False", "submit_to_hardware = True"
                )
            # Rendering is checked in real kernels by verify_notebooks.py;
            # this provider-free harness executes the submission/recovery cells.
            source = source.replace("%matplotlib inline", "")
            exec(source, scope)
    return scope


@pytest.mark.parametrize(
    "filename", ("05_jiuding_jobs.ipynb", "06_quafu_hardware.ipynb")
)
@pytest.mark.parametrize("enabled", (False, True))
@pytest.mark.parametrize("language", ("notebooks", "notebooks_cn"))
def test_all_standard_api_cells_run_and_reruns_do_not_submit(
    setup: SimpleNamespace, filename: str, enabled: bool, language: str
) -> None:
    for _ in range(2):
        scope = run_cells(filename, enabled=enabled, language=language)
    assert len(setup.calls) == int(enabled)
    if enabled:
        assert scope["job"] is setup.job
        assert scope["receipt_path"].exists()
        request = setup.calls[0]
        assert request["receipt"] == scope["receipt_path"]
        if filename.startswith("05"):
            assert (
                request["queue"] == "test-queue"
                and request["image"] == "verified:image"
            )
        else:
            assert (
                request["outputs"] == "counts-request" and request["compiler"] is None
            )
            record = json.loads(
                (setup.outputs / "quafu-ry-q0-three-qubits.json").read_text()
            )
            assert (
                record["counts"] == {"000": 768, "100": 256}
                and record["source"] == "live"
            )


@pytest.mark.parametrize(
    "filename", ("05_jiuding_jobs.ipynb", "06_quafu_hardware.ipynb")
)
def test_pending_result_is_not_failure_or_duplicate(
    setup: SimpleNamespace, filename: str
) -> None:
    setup.job.outcome = "pending"
    assert run_cells(filename, enabled=True)["result"] is None
    assert run_cells(filename, enabled=True)["result"] is None
    assert len(setup.calls) == 1


@pytest.mark.parametrize(
    "filename", ("05_jiuding_jobs.ipynb", "06_quafu_hardware.ipynb")
)
def test_switching_language_restores_the_same_receipt(
    setup: SimpleNamespace, filename: str
) -> None:
    english = run_cells(filename, enabled=True)
    receipt = english["receipt_path"].read_bytes()
    chinese = run_cells(filename, enabled=True, language="notebooks_cn")
    assert english["job"].id == chinese["job"].id
    assert chinese["receipt_path"] == english["receipt_path"]
    assert chinese["receipt_path"].read_bytes() == receipt
    assert len(setup.calls) == 1


def test_changed_request_is_rejected(setup: SimpleNamespace) -> None:
    scope = run_cells("06_quafu_hardware.ipynb", enabled=True)
    journal = Path(str(scope["receipt_path"]) + ".request.json")
    record = json.loads(journal.read_text())
    record["request_sha256"] = "0" * 64
    journal.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="configuration changed"):
        run_cells("06_quafu_hardware.ipynb", enabled=True)
    assert len(setup.calls) == 1


def test_hardware_default_and_explicit_target_override(
    setup: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_cells("06_quafu_hardware.ipynb", enabled=True)
    assert setup.calls[0]["target"] == "quafu:Baihua"
    monkeypatch.setenv("QUAFU_TARGET", "quafu:Dongling")
    with pytest.raises(ValueError, match="configuration changed"):
        run_cells("06_quafu_hardware.ipynb", enabled=True)
    assert len(setup.calls) == 1


def test_configured_hardware_target_is_used_for_a_fresh_experiment(
    setup: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("QUAFU_TARGET", "quafu:Dongling")
    run_cells("06_quafu_hardware.ipynb", enabled=True)
    assert setup.calls[0]["target"] == "quafu:Dongling"


def test_uncertain_attempt_cannot_be_retried(setup: SimpleNamespace) -> None:
    setup.outputs.mkdir()
    (setup.outputs / "quafu-ry-q0-three-qubits.submission.json").write_text("{}")
    with pytest.raises(RuntimeError, match="do not resubmit"):
        run_cells("06_quafu_hardware.ipynb", enabled=True)
    assert setup.calls == []


def test_asymmetric_lesson_keeps_old_bell_files(setup: SimpleNamespace) -> None:
    setup.outputs.mkdir()
    old_files = {
        "quafu-bell.receipt.json": '{"id": "old-bell-task"}',
        "quafu-bell.submission.json": '{"circuit_hash": "old-bell-hash"}',
        "quafu-bell.json": '{"counts": {"00": 512, "11": 512}}',
    }
    for name, content in old_files.items():
        (setup.outputs / name).write_text(content)
    for _ in range(2):
        scope = run_cells("06_quafu_hardware.ipynb", enabled=True)
    assert len(setup.calls) == 1
    assert scope["receipt_path"].name == "quafu-ry-q0-three-qubits.receipt.json"
    for name, content in old_files.items():
        assert (setup.outputs / name).read_text() == content


def test_failed_job_is_not_hidden(setup: SimpleNamespace) -> None:
    setup.job.outcome = "failed"
    with pytest.raises(RuntimeError, match="Backend failed"):
        run_cells("06_quafu_hardware.ipynb", enabled=True)
    assert len(setup.calls) == 1


def test_save_failure_keeps_guard_and_reports_id(
    setup: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def reject(path: Path) -> None:
        raise OSError("Storage unavailable")

    monkeypatch.setattr(setup.job, "save", reject)
    with pytest.raises(OSError, match="saved-task.*do not resubmit"):
        run_cells("06_quafu_hardware.ipynb", enabled=True)
    with pytest.raises(RuntimeError, match="reconcile"):
        run_cells("06_quafu_hardware.ipynb", enabled=True)
    assert len(setup.calls) == 1


@pytest.mark.parametrize(
    "filename,stem",
    [
        ("05_jiuding_jobs.ipynb", "jiuding-x0-three-qubits"),
        ("06_quafu_hardware.ipynb", "quafu-ry-q0-three-qubits"),
    ],
)
@pytest.mark.parametrize("language", ("notebooks", "notebooks_cn"))
def test_legacy_receipts_are_preserved_without_automatic_adoption(
    setup: SimpleNamespace, filename: str, stem: str, language: str
) -> None:
    setup.outputs.mkdir()
    receipt = setup.outputs / f"{stem}.receipt.json"
    old_content = '{"id": "previous-task"}'
    receipt.write_text(old_content)
    guard = setup.outputs / f"{stem}.submission.json"
    guard.write_text('{"status": "previous-attempt"}')
    with pytest.raises(ValueError, match="fq.restore_job"):
        run_cells(filename, enabled=True, language=language)
    assert receipt.read_text() == old_content
    assert guard.read_text() == '{"status": "previous-attempt"}'
    assert not Path(str(receipt) + ".request.json").exists()
    assert setup.calls == []


@pytest.mark.parametrize(
    "filename,record",
    [
        ("05_jiuding_jobs.ipynb", "jiuding-x0-three-qubits.submission.json"),
        ("06_quafu_hardware.ipynb", "quafu-ry-q0-three-qubits.json"),
    ],
)
def test_legacy_records_without_receipts_do_not_start_new_tasks(
    setup: SimpleNamespace, filename: str, record: str
) -> None:
    setup.outputs.mkdir()
    old = setup.outputs / record
    old.write_text('{"task_id": "previous-task"}')
    with pytest.raises(RuntimeError, match="do not resubmit"):
        run_cells(filename, enabled=True)
    assert old.read_text() == '{"task_id": "previous-task"}'
    assert setup.calls == []


@pytest.mark.parametrize(
    "filename", ("05_jiuding_jobs.ipynb", "06_quafu_hardware.ipynb")
)
def test_submission_error_does_not_leave_a_previous_job_in_the_kernel(
    setup: SimpleNamespace, filename: str
) -> None:
    scope = run_cells(filename, enabled=True)
    scope["target"] += "-changed"
    notebook = json.loads((WORKSHOP / "notebooks" / filename).read_text())
    source = "".join(notebook["cells"][10]["source"])
    with pytest.raises(ValueError, match="configuration changed"):
        exec(source, scope)
    assert scope["job"] is None and len(setup.calls) == 1

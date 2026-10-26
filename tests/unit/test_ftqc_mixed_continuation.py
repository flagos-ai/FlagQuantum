"""Controller boundaries use fake reports/processes, never study circuit execution."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

from tools import collect_ftqc_mixed_continuation as controller

pytestmark = pytest.mark.unit


def reports(path: Path, *, reason: str = "completed", returncode: int = 0) -> None:
    (path / "supervisor").mkdir(parents=True)
    (path / "supervisor/watchdog.json").write_text(
        json.dumps(
            {
                "reason": reason,
                "detail": "retained primary detail",
                "command_returncode": returncode,
                "rss_sample_count": 5,
                "sampled_group_rss_max_bytes": 200,
                "rss_sample_gap_max_seconds": 0.03,
                "termination_elapsed_seconds": 0.01,
            }
        )
    )


def producer(path: Path, *, status: str = "completed", error: float = 0.0) -> None:
    (path / "producer.json").write_text(
        json.dumps(
            {
                "n": 8,
                "authored_t_count": 4,
                "topology": "ring",
                "requested_mode": "tensor_network",
                "status": status,
                "diagnostic": "retained capability diagnostic",
                "errors": dict.fromkeys(
                    ("probability", "normalization", "density", "reference"), error
                ),
                "performance_targets_enabled": False,
            }
        )
    )


@pytest.mark.parametrize("available", [False, True])
def test_monitor_failure_is_primary_even_with_absent_or_broken_producer(
    tmp_path: Path,
    available: bool,
) -> None:
    reports(tmp_path, reason="rss_monitor_unavailable", returncode=-15)
    if available:
        (tmp_path / "producer.json").write_text("not json")
    result = controller.assess(tmp_path, 124, controller.matrix()[0])
    assert result["failure_domain"] == "supervision"
    assert result["producer_state"] == ("available" if available else "absent")
    assert result["watchdog"]["detail"] == "retained primary detail"
    assert result["watchdog"]["command_returncode"] == -15
    assert result["watchdog_sha256"] == controller.digest(
        tmp_path / "supervisor/watchdog.json"
    )


def test_missing_watchdog_never_accepts_producer(tmp_path: Path) -> None:
    producer(tmp_path)
    assert (
        controller.assess(tmp_path, 0, controller.matrix()[0])["failure_domain"]
        == "supervision"
    )


@pytest.mark.parametrize("broken", ["missing", "malformed", "array"])
def test_successful_supervision_reports_producer_failure(
    tmp_path: Path, broken: str
) -> None:
    reports(tmp_path)
    if broken != "missing":
        (tmp_path / "producer.json").write_text("{" if broken == "malformed" else "[]")
    result = controller.assess(tmp_path, 0, controller.matrix()[0])
    assert result["status"] == "failed" and result["failure_domain"] == "producer"
    assert "Producer report failed" in result["diagnostic"]


@pytest.mark.parametrize("error", [1e-11, float("nan"), float("inf"), -1.0, True])
def test_numeric_failure_never_passes(tmp_path: Path, error: float) -> None:
    reports(tmp_path, returncode=1)
    producer(tmp_path, status="failed", error=error)
    result = controller.assess(tmp_path, 1, controller.matrix()[0])
    assert result["status"] == "failed" and result["failure_domain"] == "producer"


def test_capability_refusal_keeps_diagnostic(tmp_path: Path) -> None:
    reports(tmp_path)
    producer(tmp_path, status="refused")
    result = controller.assess(tmp_path, 0, controller.matrix()[0])
    assert result["status"] == "refused"
    assert result["diagnostic"] == "retained capability diagnostic"


def test_valid_report_and_probe_admission(tmp_path: Path) -> None:
    reports(tmp_path)
    producer(tmp_path)
    assert (
        controller.assess(tmp_path, 0, controller.matrix()[0])["status"] == "completed"
    )
    assert controller.assess(tmp_path, 0, None)["status"] == "completed"


def test_continuation_links_only_original_failed_and_missing_slots() -> None:
    attempts = controller.matrix()
    assert len(attempts) == 28
    assert [item.original_slot for item in attempts] == list(range(164, 192))
    assert len({item.label for item in attempts}) == 28
    assert sum(item.repetition == 0 for item in attempts) == 7
    assert attempts[0].label == "n8-t4-ring-tensor_network-0"
    assert {(item.n, item.t) for item in attempts} == {(8, 4), (8, 8)}


def fake_admission(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    original = tmp_path / "fts10-cpu-observations-20261025"
    original.mkdir()
    (original / "manifest.json").write_text("immutable")
    monkeypatch.setattr(
        controller, "ORIGINAL_MANIFEST", controller.digest(original / "manifest.json")
    )
    monkeypatch.setattr(controller, "admission", lambda *args: {"fake": True})
    return tmp_path / "new"


@pytest.mark.parametrize("refused", [False, True])
def test_first_workload_failure_retains_remaining_slots_and_never_retries(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    refused: bool,
) -> None:
    output = fake_admission(monkeypatch, tmp_path)
    run = Mock(return_value=(0, None))
    monkeypatch.setattr(controller, "execute", run)
    monkeypatch.setattr(
        controller,
        "assess",
        Mock(
            side_effect=[
                {"status": "completed"},
                {"status": "refused" if refused else "failed"},
            ]
        ),
    )
    result = controller.campaign(tmp_path, output, tmp_path / "watchdog")
    assert run.call_count == 2  # One probe, one workload, no retry.
    assert result["status"] == "failed"
    assert result["attempts"][0]["status"] == ("refused" if refused else "failed")
    assert all(item["status"] == "not_attempted" for item in result["attempts"][1:])
    assert (
        tmp_path / "fts10-cpu-observations-20261025/manifest.json"
    ).read_text() == "immutable"


def test_probe_failure_starts_no_workload(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    output = fake_admission(monkeypatch, tmp_path)
    run = Mock(return_value=(124, None))
    monkeypatch.setattr(controller, "execute", run)
    monkeypatch.setattr(controller, "assess", lambda *args: {"status": "failed"})
    result = controller.campaign(tmp_path, output, tmp_path / "watchdog")
    assert run.call_count == 1
    assert all(item["status"] == "not_attempted" for item in result["attempts"])


@pytest.mark.parametrize("kind", ["storage", "time", "disk"])
def test_resource_preflight_launches_nothing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, kind: str
) -> None:
    output = fake_admission(monkeypatch, tmp_path)
    run = Mock()
    monkeypatch.setattr(controller, "execute", run)
    if kind == "storage":
        monkeypatch.setattr(
            controller, "retained_bytes", lambda folder: controller.STORAGE_BYTES
        )
    elif kind == "time":
        monkeypatch.setattr(controller, "TOTAL_SECONDS", 1)
    else:
        monkeypatch.setattr(
            controller.shutil,
            "disk_usage",
            lambda folder: type("Disk", (), {"free": 0})(),
        )
    if kind == "disk":
        with pytest.raises(OSError):
            controller.campaign(tmp_path, output, tmp_path / "watchdog")
    else:
        assert (
            controller.campaign(tmp_path, output, tmp_path / "watchdog")["status"]
            == "failed"
        )
    run.assert_not_called()


def test_output_never_overwritten(tmp_path: Path) -> None:
    marker = tmp_path / "marker"
    marker.write_text("retain")
    with pytest.raises(FileExistsError):
        controller.campaign(tmp_path, tmp_path, tmp_path / "watchdog")
    assert marker.read_text() == "retain"


def test_cancellation_terminates_watchdog_before_returning(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    process = Mock(spec=subprocess.Popen)
    process.poll.return_value = None
    process.returncode = -15
    monkeypatch.setattr(controller.subprocess, "Popen", Mock(return_value=process))
    monkeypatch.setattr(
        controller, "check_budget", Mock(side_effect=InterruptedError("cancel"))
    )
    result = controller.execute(
        [sys.executable, "-c", "pass"], tmp_path, tmp_path, tmp_path, 100
    )
    process.terminate.assert_called_once()
    process.wait.assert_called_once_with(timeout=15)
    assert result == (-15, "InterruptedError: cancel")


def test_full_fake_campaign_has_one_probe_and_exactly_28_workloads(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    output = fake_admission(monkeypatch, tmp_path)
    run = Mock(return_value=(0, None))
    monkeypatch.setattr(controller, "execute", run)
    monkeypatch.setattr(controller, "assess", lambda *args: {"status": "completed"})
    result = controller.campaign(tmp_path, output, tmp_path / "watchdog")
    assert result["status"] == "completed" and run.call_count == 29
    assert len(result["attempts"]) == 28


@pytest.mark.parametrize("gap", [0.100001, float("nan"), -0.01])
def test_monitor_gap_cannot_be_reclassified_as_numeric_success(
    tmp_path: Path, gap: float
) -> None:
    reports(tmp_path)
    path = tmp_path / "supervisor/watchdog.json"
    data = json.loads(path.read_text())
    data["rss_sample_gap_max_seconds"] = gap
    path.write_text(json.dumps(data))
    producer(tmp_path)
    assert (
        controller.assess(tmp_path, 0, controller.matrix()[0])["failure_domain"]
        == "supervision"
    )


def test_corrupt_watchdog_is_preserved_without_reading_producer(tmp_path: Path) -> None:
    (tmp_path / "supervisor").mkdir()
    path = tmp_path / "supervisor/watchdog.json"
    path.write_text("{")
    result = controller.assess(tmp_path, 124, controller.matrix()[0])
    assert result["failure_domain"] == "supervision"
    assert result["watchdog_sha256"] == controller.digest(path)


def test_numeric_error_is_rejected_even_if_command_claims_success(
    tmp_path: Path,
) -> None:
    reports(tmp_path)
    producer(tmp_path, error=1e-11)
    assert controller.assess(tmp_path, 0, controller.matrix()[0])["status"] == "failed"


def test_missing_retained_admission_artifact_starts_no_probe(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    original = tmp_path / "fts10-cpu-observations-20261025"
    original.mkdir()
    (original / "manifest.json").write_text("altered")
    run = Mock()
    monkeypatch.setattr(controller, "execute", run)
    result = controller.campaign(tmp_path, tmp_path / "new", tmp_path / "watchdog")
    assert result["status"] == "failed"
    assert "manifest identity changed" in result["controller_error"]
    run.assert_not_called()


def test_watchdog_unresponsive_cleanup_is_an_explicit_failure() -> None:
    process = Mock(spec=subprocess.Popen)
    process.poll.return_value = None
    process.wait.side_effect = [subprocess.TimeoutExpired("fake-watchdog", 15), 0]
    with pytest.raises(RuntimeError, match="cleanup"):
        controller.stop(process)
    process.terminate.assert_called_once()
    process.kill.assert_called_once()


@pytest.mark.parametrize(
    "changed", [None, "watchdog", "controller", "wheel", "installed", "runtime", "cpu"]
)
def test_retained_producer_admission_verifies_artifact_and_profile_identity(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    changed: str | None,
) -> None:
    import zipfile

    original = tmp_path / "fts10-cpu-observations-20261025"
    original.mkdir()
    watchdog = tmp_path / "watchdog.py"
    watchdog.write_text("unchanged watchdog")
    old_controller = tmp_path / "collect_fts10_cpu.py"
    old_controller.write_text("unchanged collector")
    package = tmp_path / "site-packages/flagquantum"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("original producer")
    wheel = tmp_path / "original.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("flagquantum/__init__.py", "original producer")
    wheel_record = {"path": str(wheel), "sha256": controller.digest(wheel)}
    cases: list[dict[str, object]] = [{} for _ in range(192)]
    for item in controller.matrix():
        cases[item.original_slot] = dict(
            n=item.n,
            t=item.t,
            topology=item.topology,
            mode=item.mode,
            repetition=item.repetition,
            status="failed" if item.original_slot == 164 else "not_attempted",
        )
    manifest = original / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "source_revision": controller.SOURCE,
                "watchdog_sha256": controller.digest(watchdog),
                "controller_sha256": controller.digest(old_controller),
                "cases": cases,
                "installed_wheel": wheel_record,
                "cpu_model": "Apple M5",
            }
        )
    )
    monkeypatch.setattr(controller, "ORIGINAL_MANIFEST", controller.digest(manifest))
    (tmp_path / "pr848-main-install-acceptance.json").write_text(
        json.dumps(
            {
                "source_commit": controller.SOURCE,
                "wheel": wheel_record,
            }
        )
    )
    if changed in ("watchdog", "controller", "wheel", "installed"):
        target = {
            "watchdog": watchdog,
            "controller": old_controller,
            "wheel": wheel,
            "installed": package / "__init__.py",
        }[changed]
        target.write_text("changed")
    runtime = [[3, 12, 14], "2.14.1", str(package / "__init__.py")]
    if changed == "runtime":
        runtime[1] = "different"
    monkeypatch.setattr(
        controller.subprocess,
        "check_output",
        Mock(
            side_effect=[
                json.dumps(runtime),
                "different CPU" if changed == "cpu" else "Apple M5",
            ]
        ),
    )
    if changed is not None:
        with pytest.raises(ValueError, match="changed|differs"):
            controller.admission(tmp_path, watchdog, controller.time.monotonic() + 100)
    else:
        admitted = controller.admission(
            tmp_path, watchdog, controller.time.monotonic() + 100
        )
        assert admitted["source_revision"] == controller.SOURCE
        assert admitted["installed_wheel"] == wheel_record

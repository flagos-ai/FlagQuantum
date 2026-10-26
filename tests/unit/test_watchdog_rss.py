"""Owned-command cleanup and fail-closed RSS supervision, without optional devices."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from tools import run_with_watchdog as watchdog

pytestmark = pytest.mark.unit


def arguments(path: Path, code: str, limit: int = 1024) -> list[str]:
    return [
        "--phase",
        "rss-test",
        "--stall-seconds",
        "2",
        "--timeout-seconds",
        "3",
        "--diagnostics",
        str(path),
        "--rss-limit-bytes",
        str(limit),
        "--",
        sys.executable,
        "-c",
        code,
    ]


def test_ps_counts_only_owned_group_and_converts_kib(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def snapshot(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess([], 0, "0 0\n12 100\n99 999999\n12 200\n")

    monkeypatch.setattr(subprocess, "run", snapshot)
    assert watchdog._process_group_rss(12) == 300 * 1024


@pytest.mark.parametrize("text", ["", "12\n", "12 -1\n", "bad 100\n"])
def test_ps_missing_or_malformed_accounting_is_not_zero(
    monkeypatch: pytest.MonkeyPatch, text: str
) -> None:
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess([], 0, text),
    )
    with pytest.raises(ValueError):
        watchdog._process_group_rss(12)


def test_unavailable_monitor_prevents_command_launch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    marker = tmp_path / "launched"

    def unavailable(group: int | None) -> int:
        raise PermissionError("process accounting denied")

    monkeypatch.setattr(watchdog, "_process_group_rss", unavailable)
    assert (
        watchdog.run(arguments(tmp_path, f"open({str(marker)!r}, 'w').close()")) == 125
    )
    assert not marker.exists()
    assert (
        json.loads((tmp_path / "watchdog.json").read_text())["command_started"] is False
    )


def test_rss_excess_terminates_before_work_can_finish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    marker = tmp_path / "finished"
    monkeypatch.setattr(
        watchdog, "_process_group_rss", lambda group: 0 if group is None else 2048
    )
    code = f"import time; time.sleep(3); open({str(marker)!r}, 'w').close()"
    assert watchdog.run(arguments(tmp_path, code)) == 124
    report = json.loads((tmp_path / "watchdog.json").read_text())
    assert report["reason"] == "rss_limit_exceeded"
    assert report["sampled_group_rss_max_bytes"] == 2048
    assert report["hard_allocation_limit"] is False
    assert not marker.exists()


def test_lost_accounting_stops_live_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def snapshot(group: int | None) -> int:
        if group is not None:
            raise subprocess.TimeoutExpired("ps", 0.05)
        return 0

    monkeypatch.setattr(watchdog, "_process_group_rss", snapshot)
    assert watchdog.run(arguments(tmp_path, "import time; time.sleep(3)")) == 124
    assert (
        json.loads((tmp_path / "watchdog.json").read_text())["reason"]
        == "rss_monitor_unavailable"
    )


def test_delayed_sampling_is_not_certified_as_within_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def snapshot(group: int | None) -> int:
        if group is not None:
            time.sleep(0.12)
        return 0 if group is None else 512

    monkeypatch.setattr(watchdog, "_process_group_rss", snapshot)
    assert watchdog.run(arguments(tmp_path, "import time; time.sleep(3)")) == 124
    assert (
        json.loads((tmp_path / "watchdog.json").read_text())["reason"]
        == "rss_sampling_gap_exceeded"
    )


def test_success_records_observations_and_preserves_exit_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def snapshot(group: int | None) -> int:
        if group is None:
            return 0
        try:
            os.kill(group, 0)
        except ProcessLookupError:
            return 0
        return 512

    monkeypatch.setattr(watchdog, "_process_group_rss", snapshot)
    assert (
        watchdog.run(
            arguments(tmp_path, "import time; time.sleep(0.15); raise SystemExit(7)")
        )
        == 7
    )
    report = json.loads((tmp_path / "watchdog.json").read_text())
    assert report["reason"] == "completed"
    assert report["command_returncode"] == 7
    assert report["sampled_group_rss_max_bytes"] == 512
    assert report["rss_sample_count"] > 0
    assert report["rss_sample_gap_max_seconds"] <= 0.1


def test_children_outliving_launcher_are_terminated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    marker = tmp_path / "orphan-finished"
    child = f"import time; time.sleep(0.5); open({str(marker)!r}, 'w').close()"
    command = f"import subprocess,sys,time; subprocess.Popen([sys.executable,'-c',{child!r}]); time.sleep(0.1)"
    monkeypatch.setattr(
        watchdog, "_process_group_rss", lambda group: 0 if group is None else 512
    )
    assert watchdog.run(arguments(tmp_path, command)) == 124
    report = json.loads((tmp_path / "watchdog.json").read_text())
    assert report["reason"] == "rss_group_survived_launcher"
    time.sleep(0.6)
    assert not marker.exists()


def test_no_rss_flag_preserves_existing_command_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden(group: int | None) -> int:
        raise AssertionError("default command must not query RSS")

    monkeypatch.setattr(watchdog, "_process_group_rss", forbidden)
    assert (
        watchdog.run(
            [
                "--phase",
                "default",
                "--stall-seconds",
                "2",
                "--timeout-seconds",
                "3",
                "--diagnostics",
                str(tmp_path),
                "--",
                sys.executable,
                "-c",
                "pass",
            ]
        )
        == 0
    )
    assert not (tmp_path / "watchdog.json").exists()


def test_python_interruption_is_preserved_after_owned_group_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    marker = tmp_path / "finished"

    def interrupt(group: int | None) -> int:
        if group is not None:
            raise KeyboardInterrupt()
        return 0

    monkeypatch.setattr(watchdog, "_process_group_rss", interrupt)
    with pytest.raises(KeyboardInterrupt):
        watchdog.run(
            arguments(
                tmp_path, f"import time;time.sleep(3);open({str(marker)!r},'w').close()"
            )
        )
    record = json.loads((tmp_path / "watchdog.json").read_text())
    assert record["reason"] == "supervisor_interrupted"
    assert record["exception_type"] == "KeyboardInterrupt"
    assert record["command_returncode"] is not None
    assert not marker.exists()


@pytest.mark.parametrize("listing", ["0\n42\n", "  0\n  42\n"])
def test_denied_signal_accepts_only_a_confirmed_absent_group(
    monkeypatch: pytest.MonkeyPatch, listing: str
) -> None:
    import signal

    def denied(group: int, signum: int) -> None:
        raise PermissionError("signal denied")

    def snapshot(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        assert kwargs["check"] is True
        timeout = kwargs["timeout"]
        assert isinstance(timeout, (int, float)) and 0 < timeout <= 0.05
        return subprocess.CompletedProcess([], 0, listing)

    monkeypatch.setattr(os, "killpg", denied)
    monkeypatch.setattr(subprocess, "run", snapshot)
    watchdog._signal_group(123, signal.SIGKILL)


@pytest.mark.parametrize(
    "listing", ["123\n", "42\n123\n", "", "bad\n", "-1\n", "123 1\n"]
)
def test_denied_signal_keeps_live_or_unknown_groups_failed(
    monkeypatch: pytest.MonkeyPatch, listing: str
) -> None:
    import signal

    def denied(group: int, signum: int) -> None:
        raise PermissionError("signal denied")

    monkeypatch.setattr(os, "killpg", denied)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess([], 0, listing),
    )
    with pytest.raises(PermissionError, match="signal denied"):
        watchdog._signal_group(123, signal.SIGTERM)


def test_denied_signal_does_not_hide_unavailable_accounting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import signal

    def denied(group: int, signum: int) -> None:
        raise PermissionError("signal denied")

    def unavailable(
        *args: object, **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired("ps", 0.05)

    monkeypatch.setattr(os, "killpg", denied)
    monkeypatch.setattr(subprocess, "run", unavailable)
    with pytest.raises(PermissionError, match="signal denied"):
        watchdog._signal_group(123, signal.SIGKILL)

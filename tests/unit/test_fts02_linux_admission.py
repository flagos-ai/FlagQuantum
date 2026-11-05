"""Pure admission substitutes: no syscalls, mounts or real children are invoked."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

from tools import check_fts02_linux_admission as admission
from tools import launch_fts02_cpu as launcher
from tools import run_with_watchdog as watchdog
from tools.fts02_process_budget import (
    CreationDeniedError,
    Reservation,
    isolation_identity,
)

pytestmark = pytest.mark.unit


def mounts(
    *, root: str = "ro", capacity_fs: str = "tmpfs", writable: str = "rw"
) -> str:
    return (
        f"1 0 0:1 / / {root} - ext4 disk {root}\n"
        f"2 1 0:2 / /evidence {writable} - {capacity_fs} tmpfs {writable}\n"
    )


def test_readonly_namespace_and_capped_evidence_are_required() -> None:
    admission.check_mounts(mounts(), Path("/evidence"), 32 * 1024 * 1024)


@pytest.mark.parametrize(
    "text,capacity",
    [
        (mounts(root="rw"), 32),
        (mounts(capacity_fs="ext4"), 32),
        (mounts(writable="ro"), 32),
        (mounts(), 0),
        (mounts(), 32 * 1024 * 1024 + 1),
        ("", 32),
        (mounts() + mounts(), 32),
    ],
)
def test_bad_mounts_refuse_before_any_process(text: str, capacity: int) -> None:
    with pytest.raises(RuntimeError):
        admission.check_mounts(text, Path("/evidence"), capacity)


def test_dependency_mismatch_stops_before_watchdog(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reservation = Reservation.create(tmp_path / "r", "a" * 40, {"python": "approved"})
    reservation.claim("admission")
    reservation.update(flag="admission_passed")
    reservation.precharge_campaign()
    monkeypatch.setattr(launcher, "runtime_identity", lambda: {"python": "different"})
    forbidden = Mock(side_effect=AssertionError("must not create a process"))
    monkeypatch.setattr(watchdog, "run", forbidden)
    with pytest.raises(RuntimeError, match="dependency"):
        launcher.run_campaign(reservation.directory / "run", "a" * 40, reservation)
    forbidden.assert_not_called()
    assert reservation.summary()["creation_attempts"] == 2


def test_campaign_requires_external_precharge(tmp_path: Path) -> None:
    reservation = Reservation.create(tmp_path / "r", "a" * 40, {"python": "fake"})
    reservation.claim("admission")
    reservation.update(flag="admission_passed")
    with pytest.raises(CreationDeniedError, match="charged"):
        reservation.claim("campaign")


def test_native_denial_slots_are_charged_and_bounded(tmp_path: Path) -> None:
    reservation = Reservation.create(tmp_path / "r", "a" * 40, {"python": "fake"})
    reservation.claim("admission")
    for _ in range(3):
        reservation.charge("admission", "native_denial")
    with pytest.raises(CreationDeniedError):
        reservation.charge("admission", "native_denial")
    assert reservation.summary()["creation_attempts"] == 4


def test_success_report_uses_bounded_writer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A fake process is already exited; the final RSS check refuses zero live samples.
    process = Mock(pid=123, returncode=0)
    process.poll.return_value = 0
    process.stdout.read.return_value = b""
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: process)
    monkeypatch.setattr(os, "set_blocking", lambda *a: None)
    monkeypatch.setattr(os, "killpg", lambda *a: None)
    monkeypatch.setattr(watchdog, "_process_group_rss", lambda *a, **k: 0)
    writes: list[Path] = []
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: Mock(stdout="", stderr=""))
    result = watchdog.run(
        [
            "--phase",
            "fake",
            "--stall-seconds",
            "1",
            "--timeout-seconds",
            "1",
            "--rss-limit-bytes",
            "100",
            "--diagnostics",
            str(tmp_path),
            "--",
            "fake",
        ],
        charge=lambda role: None,
        write=lambda path, data: writes.append(path),
    )
    assert result == 124
    assert tmp_path / "watchdog.json" in writes


@pytest.mark.parametrize(
    "stdout,stderr",
    [
        (b"partial stdout", b"partial stderr"),
        ("partial stdout", "partial stderr"),
        (None, None),
        (b"partial stdout", None),
    ],
)
def test_timeout_retains_partial_output_and_original_exception(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stdout: bytes | str | None,
    stderr: bytes | str | None,
) -> None:
    reservation = Reservation.create(tmp_path / "r", "a" * 40, {"python": "fake"})
    reservation.claim("admission")
    failure = subprocess.TimeoutExpired(["fake"], 15, output=stdout, stderr=stderr)
    seen: list[int] = []

    def timeout(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        assert kwargs["timeout"] == 15
        attempts, _, _ = reservation._read()
        seen.append(attempts["admission/native"])
        raise failure

    monkeypatch.setattr(subprocess, "run", timeout)
    with pytest.raises(subprocess.TimeoutExpired) as captured:
        admission.launch_child(
            reservation, launcher.Evidence(reservation), "native", ["fake"]
        )
    assert captured.value is failure
    assert seen == [1]
    for channel, expected in (("stdout", stdout), ("stderr", stderr)):
        expected_bytes = (
            expected.encode() if isinstance(expected, str) else expected or b""
        )
        assert (
            reservation.directory / f"admission-native-timeout.{channel}"
        ).read_bytes() == expected_bytes
    with pytest.raises(CreationDeniedError):
        reservation.charge("admission", "native")


def test_timeout_log_failure_preserves_original_cause(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reservation = Reservation.create(tmp_path / "r", "a" * 40, {"python": "fake"})
    reservation.claim("admission")
    failure = subprocess.TimeoutExpired(["fake"], 15, output=b"partial")
    run = Mock(side_effect=failure)
    monkeypatch.setattr(subprocess, "run", run)
    output = launcher.Evidence(reservation)

    def unavailable(path: Path, data: bytes, *, diagnostic: bool = False) -> None:
        raise OSError("fake evidence volume failure")

    monkeypatch.setattr(output, "write", unavailable)
    with pytest.raises(OSError, match="volume") as captured:
        admission.launch_child(reservation, output, "native", ["fake"])
    assert captured.value.__cause__ is failure
    assert run.call_count == 1


@pytest.mark.parametrize("change", ["namespace", "writable", "capacity", "missing"])
def test_changed_isolation_refuses_before_campaign_child(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    change: str,
) -> None:
    reservation = Reservation.create(tmp_path / "r", "a" * 40, {"python": "fake"})
    reservation.claim("admission")
    if change != "missing":
        reservation.stamp_isolation({"namespace": "approved"})
    reservation.update(flag="admission_passed")
    reservation.precharge_campaign()
    monkeypatch.setattr(launcher, "runtime_identity", lambda: {"python": "fake"})

    def identity(path: Path) -> dict[str, str]:
        if change == "writable":
            admission.check_mounts(mounts(root="rw"), Path("/evidence"), 32)
        if change == "capacity":
            admission.check_mounts(mounts(), Path("/evidence"), 32 * 1024 * 1024 + 1)
        return {"namespace": "changed" if change == "namespace" else "approved"}

    monkeypatch.setattr(launcher, "isolation_identity", identity)
    child = Mock(side_effect=AssertionError("must not launch"))
    monkeypatch.setattr(watchdog, "run", child)
    monkeypatch.setattr(launcher, "git_provenance", child)
    with pytest.raises((ValueError, RuntimeError, OSError)):
        launcher.run_campaign(reservation.directory / "run", "a" * 40, reservation)
    child.assert_not_called()
    assert reservation.summary()["creation_attempts"] == 2
    flags = reservation.summary()["flags"]
    assert isinstance(flags, list)
    assert "closed" in flags


def test_matching_isolation_allows_only_the_mocked_campaign(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reservation = Reservation.create(tmp_path / "r", "a" * 40, {"python": "fake"})
    reservation.claim("admission")
    reservation.stamp_isolation({"namespace": "approved"})
    reservation.update(flag="admission_passed")
    reservation.precharge_campaign()
    monkeypatch.setattr(launcher, "runtime_identity", lambda: {"python": "fake"})
    monkeypatch.setattr(
        launcher, "isolation_identity", lambda path: {"namespace": "approved"}
    )
    monkeypatch.setattr(launcher, "git_provenance", lambda *a: None)
    run = Mock(return_value=0)
    monkeypatch.setattr(watchdog, "run", run)
    assert (
        launcher.run_campaign(reservation.directory / "run", "a" * 40, reservation) == 0
    )
    assert run.call_count == 6
    assert reservation.summary()["creation_attempts"] == 2
    flags = reservation.summary()["flags"]
    assert isinstance(flags, list)
    assert "closed" in flags


def test_isolation_identity_binds_validated_namespace_and_mounts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence = tmp_path.resolve()
    text = mounts().replace("/evidence", str(evidence))
    namespace = {"/proc/self/ns/mnt": "mnt:[123]", "/proc/self/ns/net": "net:[456]"}
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(os, "geteuid", lambda: 1000)
    monkeypatch.setattr(os, "statvfs", lambda path: Mock(f_frsize=4096, f_blocks=8192))
    monkeypatch.setattr(os, "readlink", lambda path: namespace[str(path)])

    def read_mountinfo(path: Path) -> str:
        assert path == Path("/proc/self/mountinfo")
        return text

    monkeypatch.setattr(Path, "read_text", read_mountinfo)
    first = isolation_identity(evidence)
    assert first["mount_namespace"] == "mnt:[123]"
    assert first["capacity_bytes"] == str(32 * 1024 * 1024)
    namespace["/proc/self/ns/mnt"] = "mnt:[789]"
    assert isolation_identity(evidence) != first
    namespace["/proc/self/ns/mnt"] = "mnt:[123]"
    text = text.replace("1 0 0:1", "7 0 0:1")
    assert isolation_identity(evidence) != first
    text = mounts(root="rw").replace("/evidence", str(evidence))
    with pytest.raises(RuntimeError, match="writable"):
        isolation_identity(evidence)

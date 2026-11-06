"""Exercise creation accounting with substitutes for every native creation."""

from __future__ import annotations

import os
import signal
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tools import run_with_watchdog as watchdog
from tools.fts02_process_budget import (
    POINTS,
    CreationBudget,
    CreationDeniedError,
    TargetCreationGuard,
    point_limits,
    run_points,
)

pytestmark = pytest.mark.unit


def test_exact_campaign_capacity() -> None:
    assert 1 + sum(sum(point_limits(point).values()) for point in POINTS) == 90_097
    coordinator = CreationBudget("campaign")
    coordinator.charge("coordinator")
    with pytest.raises(CreationDeniedError):
        coordinator.charge("coordinator")


@pytest.mark.parametrize("point", POINTS)
def test_all_role_edges(point: str) -> None:
    budget = CreationBudget(point)
    limits = point_limits(point)
    for role, limit in limits.items():
        for _ in range(limit):
            budget.charge(role)
        with pytest.raises(CreationDeniedError):
            budget.charge(role)
    assert budget.snapshot() == {role: cap for role, cap in limits.items() if cap}


def test_git_requests_and_unapproved_target_creations() -> None:
    guard = TargetCreationGuard("w3", "/repo")
    for args in (
        ["git", "rev-parse", "HEAD"],
        ["git", "status", "--porcelain", "--untracked-files=no"],
    ):
        guard("subprocess.Popen", ("git", args, "/repo", None))
    with pytest.raises(CreationDeniedError):
        guard("subprocess.Popen", ("git", args, "/repo", None))
    with pytest.raises(CreationDeniedError):
        guard("subprocess.Popen", ("sh", ["sh"], "/repo", None))
    with pytest.raises(CreationDeniedError):
        guard("os.fork", ())
    assert guard.budget.snapshot() == {"provenance": 2}


def test_stops_at_first_failed_point() -> None:
    def fake_launch(point: str, budget: CreationBudget) -> int:
        budget.charge("target")
        return 124 if point == "w3" else 0

    result, attempts = run_points(fake_launch)
    assert result == 124
    assert tuple(attempts) == POINTS[:3]
    assert all(row == {"watchdog": 1, "target": 1} for row in attempts.values())


def test_failed_rss_creation_stays_charged(monkeypatch: pytest.MonkeyPatch) -> None:
    budget = CreationBudget("w2-x")

    def fail(*args: object, **kwargs: object) -> None:
        assert budget.snapshot() == {"admission": 1}
        raise OSError("fake creation failure")

    monkeypatch.setattr(subprocess, "run", fail)
    with pytest.raises(OSError, match="fake creation failure"):
        watchdog._process_group_rss(None, "admission", budget.charge)
    assert budget.snapshot() == {"admission": 1}
    with pytest.raises(CreationDeniedError):
        watchdog._process_group_rss(None, "admission", budget.charge)


def test_cleanup_checks_are_independent_of_exhausted_regular_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    budget = CreationBudget("w2-x")
    for _ in range(point_limits("w2-x")["regular"]):
        budget.charge("regular")
    with pytest.raises(CreationDeniedError):
        budget.charge("regular")

    def denied(group: int, signum: int) -> None:
        assert signum in (signal.SIGTERM, signal.SIGKILL)
        raise PermissionError

    monkeypatch.setattr(os, "killpg", denied)
    monkeypatch.setattr(
        subprocess, "run", lambda *a, **k: subprocess.CompletedProcess([], 0, "99\n")
    )
    process = Mock(pid=12)
    for _ in range(3):
        watchdog._stop_group(process, budget.charge)
    assert budget.snapshot()["cleanup"] == 6
    with pytest.raises(CreationDeniedError):
        watchdog._stop_group(process, budget.charge)


def test_diagnostics_charge_before_creation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    budget = CreationBudget("w2-x")
    observed = []

    def fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        observed.append(budget.snapshot()["diagnostic"])
        return subprocess.CompletedProcess([], 0, "fake\n", "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    watchdog._diagnostics(tmp_path, {}, budget.charge)
    watchdog._diagnostics(tmp_path, {}, budget.charge)
    with pytest.raises(CreationDeniedError):
        watchdog._diagnostics(tmp_path, {}, budget.charge)
    assert observed == [1, 2, 3, 4]


def test_watchdog_denial_enters_cleanup_and_retains_diagnostics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    budget = CreationBudget("w2-x")
    for _ in range(point_limits("w2-x")["regular"]):
        budget.charge("regular")
    signals = []

    def fake_run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        if argv == ["ps", "-axo", "pgid=,rss="]:
            assert budget.snapshot()["admission"] == 1
            return subprocess.CompletedProcess(argv, 0, "99 1\n")
        assert budget.snapshot()["diagnostic"] in (1, 2)
        return subprocess.CompletedProcess(argv, 0, "", "")

    def fake_popen(*args: object, **kwargs: object) -> SimpleNamespace:
        assert budget.snapshot()["target"] == 1
        return SimpleNamespace(
            pid=12,
            returncode=-15,
            poll=lambda: None,
            wait=lambda timeout: None,
            stdout=SimpleNamespace(fileno=lambda: 0, close=lambda: None),
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    monkeypatch.setattr(os, "set_blocking", lambda *a: None)
    monkeypatch.setattr(os, "killpg", lambda *a: signals.append(a))
    monkeypatch.setattr(signal, "signal", lambda *a: None)
    with pytest.raises(CreationDeniedError, match="regular"):
        watchdog.run(
            [
                "--phase",
                "fake",
                "--stall-seconds",
                "1",
                "--timeout-seconds",
                "1",
                "--diagnostics",
                str(tmp_path),
                "--rss-limit-bytes",
                "1024",
                "--",
                "fake-target",
            ],
            charge=budget.charge,
        )
    assert signals == [(12, signal.SIGTERM), (12, signal.SIGKILL)] * 2
    assert budget.snapshot()["diagnostic"] == 2
    assert "CreationDeniedError" in (tmp_path / "watchdog.json").read_text()


def test_persistent_reservation_survives_reload_and_refuses_replay(
    tmp_path: Path,
) -> None:
    from tools.fts02_process_budget import CAPACITY, Reservation, reservation_limits

    assert sum(reservation_limits().values()) == CAPACITY == 90_097
    path = tmp_path / "reservation"
    reservation = Reservation.create(path, "a" * 40, {"python": "fake"})
    assert reservation.summary()["creation_attempts"] == 1
    reservation.claim("admission")
    with pytest.raises(CreationDeniedError):
        reservation.claim("admission")
    reservation.charge("admission", "native")
    restored = Reservation(path, "a" * 40, {"python": "fake"})
    with pytest.raises(CreationDeniedError):
        restored.charge("admission", "native")
    with pytest.raises(CreationDeniedError):
        restored.precharge_campaign()
    restored.update(flag="admission_passed")
    restored.precharge_campaign()
    restored.claim("campaign")
    restored.charge("w2-x", "target")
    assert restored.summary()["creation_attempts"] == 4
    restored.update(flag="closed")
    with pytest.raises(CreationDeniedError):
        restored.charge("w2-z", "target")


def test_stale_lock_identity_and_byte_capacity_fail_closed(tmp_path: Path) -> None:
    from tools.fts02_process_budget import Reservation

    path = tmp_path / "reservation"
    reservation = Reservation.create(path, "b" * 40, {"python": "fake"})
    with pytest.raises(ValueError, match="identity"):
        Reservation(path, "c" * 40, {"python": "fake"}).summary()
    reservation.update(bytes_count=10, byte_limit=10)
    with pytest.raises(CreationDeniedError, match="evidence"):
        reservation.update(bytes_count=1, byte_limit=10)
    (path / "active.lock").touch()
    with pytest.raises(FileExistsError):
        reservation.claim("admission")
    assert reservation.summary()["charged_evidence_bytes"] == 10


def test_diagnostic_bytes_do_not_block_reserved_cleanup_or_close(
    tmp_path: Path,
) -> None:
    from tools.fts02_process_budget import (
        EVIDENCE_CAPACITY,
        Reservation,
        reservation_limits,
    )

    reservation = Reservation.create(tmp_path / "r", "a" * 40, {"python": "fake"})
    reservation.claim("admission")
    reservation.update(flag="admission_passed")
    reservation.precharge_campaign()
    reservation.claim("campaign")
    normal_limit = EVIDENCE_CAPACITY - 1024 * 1024
    reservation.update(bytes_count=normal_limit)
    with pytest.raises(CreationDeniedError, match="evidence"):
        reservation.update(bytes_count=1)
    diagnostic_limit = EVIDENCE_CAPACITY - 128 * 1024
    reservation.update(
        bytes_count=diagnostic_limit - normal_limit, byte_limit=diagnostic_limit
    )
    attempts, flags, evidence = reservation._read()
    attempts["w2-x/regular"] = reservation_limits()["w2-x/regular"]
    reservation._save(attempts, flags, evidence)
    with pytest.raises(CreationDeniedError, match="regular"):
        reservation.charge("w2-x", "regular")
    for _ in range(6):
        reservation.charge("w2-x", "cleanup")
    with pytest.raises(CreationDeniedError, match="cleanup"):
        reservation.charge("w2-x", "cleanup")
    with pytest.raises(CreationDeniedError, match="evidence"):
        reservation.update(bytes_count=1, byte_limit=diagnostic_limit)
    receipt_limit = EVIDENCE_CAPACITY - 64 * 1024
    reservation.update(
        bytes_count=receipt_limit - diagnostic_limit, byte_limit=receipt_limit
    )
    reservation.update(flag="closed")
    restored = Reservation(reservation.directory, "a" * 40, {"python": "fake"})
    assert restored.summary()["charged_evidence_bytes"] == receipt_limit
    with pytest.raises(CreationDeniedError, match="phase"):
        restored.charge("w2-z", "target")


def test_write_tiers_cannot_exceed_global_evidence_capacity(tmp_path: Path) -> None:
    from tools.fts02_process_budget import EVIDENCE_CAPACITY, Reservation

    reservation = Reservation.create(tmp_path / "r", "a" * 40, {"python": "fake"})
    with pytest.raises(CreationDeniedError, match="evidence"):
        reservation.update(
            bytes_count=EVIDENCE_CAPACITY + 1, byte_limit=EVIDENCE_CAPACITY + 100
        )
    with pytest.raises(CreationDeniedError, match="evidence"):
        reservation.update(bytes_count=-1)
    assert reservation.summary()["charged_evidence_bytes"] == 0

"""Review the launcher without creating processes or loading a native filter."""

from __future__ import annotations

import ctypes
import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

from tools import fts02_native_boundary as boundary
from tools import launch_fts02_cpu as launcher
from tools import run_with_watchdog
from tools.fts02_process_budget import POINTS, CreationBudget, Reservation

pytestmark = pytest.mark.unit


def test_fixed_manifest_uses_only_stdout() -> None:
    for point in POINTS:
        _, args = launcher.command(point)
        assert "--output" not in args
        assert "--json-output" not in args
    assert launcher.command("w3")[1] == ["--distance", "9", "--rounds", "5"]
    with pytest.raises(ValueError):
        launcher.command("other")


def test_evidence_reserves_final_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(launcher, "CAPACITY", 16)
    monkeypatch.setattr(launcher, "RESERVE", 6)
    monkeypatch.setattr(launcher, "RECEIPT_RESERVE", 2)
    evidence = launcher.Evidence()
    log = tmp_path / "target.log"
    evidence.write(log, b"0123456789")
    with pytest.raises(RuntimeError, match="capacity"):
        evidence.write(log, b"x")
    diagnostics = tmp_path / "diagnostics"
    evidence.write(diagnostics, b"abcdef", diagnostic=True)
    assert diagnostics.read_bytes() == b"abcd"
    assert evidence.written == 14
    assert evidence.diagnostics_truncated


def test_launcher_binds_counts_and_stops(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen = []

    def git(budget: CreationBudget, revision: str, reservation: Reservation) -> None:
        assert revision == "a" * 40
        budget.charge("provenance")
        budget.charge("provenance")

    def watchdog(args: list[str], **kwargs: object) -> int:
        point = args[1]
        seen.append(point)
        assert kwargs["setup"] is boundary.install
        charge = kwargs["charge"]
        emit = kwargs["emit"]
        assert callable(charge) and callable(emit)
        charge("target")
        emit(b"fake output")
        return 124 if point == "w6-16" else 0

    monkeypatch.setattr(launcher, "git_provenance", git)
    monkeypatch.setattr(run_with_watchdog, "run", watchdog)
    reservation = Reservation.create(tmp_path / "reservation", "a" * 40, {"fake": "1"})
    reservation.claim("admission")
    reservation.stamp_isolation({"namespace": "fake"})
    reservation.update(flag="admission_passed")
    reservation.precharge_campaign()
    monkeypatch.setattr(launcher, "runtime_identity", lambda: {"fake": "1"})
    monkeypatch.setattr(
        launcher, "isolation_identity", lambda path: {"namespace": "fake"}
    )
    output = reservation.directory / "run"
    assert launcher.run_campaign(output, "a" * 40, reservation) == 124
    assert tuple(seen) == POINTS[:4]
    receipt = json.loads((output / "receipt.json").read_text())
    assert receipt["point_attempts"]["w3"] == {"provenance": 2, "target": 1}
    assert receipt["point_attempts"]["w6-32"] == {}
    assert receipt["status"] == "failed"
    with pytest.raises(RuntimeError):
        launcher.run_campaign(output, "a" * 40, reservation)
    assert len(seen) == 4


def test_git_attempts_are_charged_before_contained_launch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    budget = CreationBudget("w3")
    seen = []

    def fake_git(args: list[str], **kwargs: object) -> str:
        seen.append(args)
        assert budget.snapshot()["provenance"] == len(seen)
        assert kwargs["preexec_fn"] is boundary.install
        return "a" * 40 if len(seen) == 1 else ""

    monkeypatch.setattr(subprocess, "check_output", fake_git)
    launcher.git_provenance(budget, "a" * 40)
    assert len(seen) == 2


def test_native_filter_rules_and_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    lib = Mock()
    lib.seccomp_init.return_value = 123
    lib.seccomp_syscall_resolve_name.side_effect = range(6)
    lib.seccomp_rule_add_array.return_value = 0
    lib.seccomp_load.return_value = -1
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(os, "uname", lambda: Mock(machine="aarch64"))
    monkeypatch.setattr(ctypes, "CDLL", lambda _: lib)
    with pytest.raises(RuntimeError, match="cannot load"):
        boundary.install()
    assert [
        call.args[0] for call in lib.seccomp_syscall_resolve_name.call_args_list
    ] == [
        b"fork",
        b"vfork",
        b"clone3",
        b"clone",
        b"setsid",
        b"setpgid",
    ]
    clone = lib.seccomp_rule_add_array.call_args_list[3].args
    assert clone[3] == 1
    assert clone[4]._obj.mask == boundary.CLONE_THREAD
    assert clone[4]._obj.value == 0
    lib.seccomp_release.assert_called_once_with(123)

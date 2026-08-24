from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from tools.pre_push import Check, checks, environment_executable, run_checks

pytestmark = pytest.mark.unit


def test_pre_push_gate_reuses_checked_in_ci_tiers() -> None:
    commands = {check.command for check in checks("python")}
    assert (
        "pre-commit",
        "run",
        "--all-files",
        "--hook-stage",
        "pre-commit",
        "--show-diff-on-failure",
    ) in commands
    assert ("python", "tools/check_dependency_policy.py") in commands
    assert ("python", "tools/check_qiskit_interop_contract.py") in commands
    assert ("python", "tools/check_pennylane_interop_contract.py") in commands
    assert ("python", "tools/check_double_single_contract.py") in commands
    assert ("python", "tools/check_split_real_imag_contract.py") in commands
    assert ("python", "tools/check_split_real_imag_p1_contract.py") in commands
    assert (
        "python",
        "tools/validate_split_real_imag_training_evidence.py",
    ) in commands
    assert (
        "python",
        "tools/check_split_real_imag_p2_precision_contract.py",
    ) in commands
    assert (
        "python",
        "tools/validate_split_real_imag_precision_evidence.py",
    ) in commands
    assert (
        "python",
        "tools/check_split_real_imag_p3_double_single_contract.py",
    ) in commands
    assert (
        "python",
        "tools/validate_split_real_imag_double_single_evidence.py",
    ) in commands
    assert any(
        check.name == "strict typed interoperability contract"
        and "flagquantum/interop/contracts.py" in check.command
        and "flagquantum/interop/conformance.py" in check.command
        and "flagquantum/interop/qiskit/conformance.py" in check.command
        and "flagquantum/interop/pennylane/adapter.py" in check.command
        for check in checks("python")
    )
    for tier in ("pr-default", "pr-runtime", "pr-distributed"):
        assert ("python", "tools/ci_tier.py", tier) in commands


def test_pre_push_gate_resolves_tools_next_to_active_python(tmp_path: Path) -> None:
    python = tmp_path / "bin" / "python"
    mypy = tmp_path / "bin" / "mypy"
    python.parent.mkdir()
    python.touch()
    mypy.touch()

    assert environment_executable(str(python), "mypy") == str(mypy)
    assert environment_executable(str(python), "missing") == "missing"


def test_pre_push_gate_stops_at_first_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    invoked: list[tuple[str, ...]] = []

    monkeypatch.setattr(
        "tools.pre_push.shutil.which",
        lambda executable, *, path: executable,
    )

    def fake_run(
        command: tuple[str, ...],
        *,
        cwd: Path,
        check: bool,
        env: dict[str, str],
    ) -> subprocess.CompletedProcess[bytes]:
        assert cwd == tmp_path
        assert check is False
        assert "PATH" in env
        invoked.append(command)
        return subprocess.CompletedProcess(command, 1)

    monkeypatch.setattr("tools.pre_push.subprocess.run", fake_run)
    selected = (
        Check("fails", ("python", "first.py")),
        Check("must not run", ("python", "second.py")),
    )

    assert run_checks(selected, root=tmp_path, dry_run=False) == 1
    assert invoked == [("python", "first.py")]


def test_pre_push_gate_dry_run_does_not_spawn_processes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        "tools.pre_push.shutil.which",
        lambda executable, *, path: executable,
    )

    def unexpected_run(*args: object, **kwargs: object) -> None:
        raise AssertionError("dry run must not spawn a subprocess")

    monkeypatch.setattr("tools.pre_push.subprocess.run", unexpected_run)
    assert (
        run_checks(
            (Check("dry", ("python", "check.py")),),
            root=tmp_path,
            dry_run=True,
        )
        == 0
    )

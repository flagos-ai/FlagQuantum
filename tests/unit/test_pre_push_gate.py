from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest
import yaml

from tools.pre_push import Check, checks, environment_executable, run_checks

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]


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
    assert ("python", "tools/check_braket_interop_contract.py") in commands
    assert ("python", "tools/check_qiskit_interop_contract.py") in commands
    assert ("python", "tools/check_cudaq_export_contract.py") in commands
    assert ("python", "tools/check_pennylane_interop_contract.py") in commands
    assert ("python", "tools/check_interop_capability_gap_matrix.py") in commands
    assert ("python", "tools/check_openqasm_import_contract.py") in commands
    assert ("python", "tools/check_double_single_contract.py") in commands
    assert ("python", "tools/check_circuit_composition_contract.py") in commands
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
    assert (
        "python",
        "tools/check_split_real_imag_p4_device_double_single_contract.py",
    ) in commands
    assert (
        "python",
        "tools/validate_split_real_imag_device_double_single_evidence.py",
    ) in commands
    assert (
        "python",
        "tools/check_split_real_imag_p5_autograd_optimizer_contract.py",
    ) in commands
    assert any(
        check.name == "strict type check of the whole package"
        and check.command[-1] == "flagquantum"
        and "--strict" in check.command
        for check in checks("python")
    )
    for tier in ("pr-default", "pr-runtime", "pr-distributed"):
        assert ("python", "tools/ci_tier.py", tier) in commands


def test_pre_push_gate_checks_ownership_of_the_changed_paths() -> None:
    """The gate must classify changes, not only validate the policy file.

    `--validate` reads `team-ownership.toml` and never resolves a path, so a
    branch that adds an unowned file passes it. The CI quality job runs the
    classification step as well; the local gate has to run both, or the failure
    is only discovered after the push.
    """

    commands = {check.command for check in checks("python")}
    assert ("python", "tools/check_team_scope.py", "--validate") in commands
    assert (
        "python",
        "tools/check_team_scope.py",
        "--require-classified",
        "--base",
        "origin/main",
    ) in commands


def test_pre_push_gate_checks_new_evidence_retention() -> None:
    commands = {check.command for check in checks("python")}
    assert (
        "python",
        "tools/check_evidence_retention.py",
        "--base",
        "origin/main",
    ) in commands


def test_pre_push_gate_resolves_tools_next_to_active_python(tmp_path: Path) -> None:
    python = tmp_path / "bin" / "python"
    mypy = tmp_path / "bin" / "mypy"
    python.parent.mkdir()
    python.touch()
    mypy.touch()

    assert environment_executable(str(python), "mypy") == str(mypy)
    assert environment_executable(str(python), "missing") == "missing"


def test_pre_push_gate_keeps_the_active_environment_first_on_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A virtual environment interpreter is a symlink to its base interpreter.

    The gate must prepend the directory holding the interpreter as invoked. If
    it resolved that symlink first, the base interpreter's ``bin`` would come
    first on ``PATH`` and shadow the environment's console scripts, so the
    checked-in tools would run without the project's dependencies.
    """

    base_bin = tmp_path / "base" / "bin"
    environment_bin = tmp_path / "env" / "bin"
    base_bin.mkdir(parents=True)
    environment_bin.mkdir(parents=True)
    base_python = base_bin / "python"
    base_python.touch()
    environment_python = environment_bin / "python"
    environment_python.symlink_to(base_python)

    captured: list[dict[str, str]] = []
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
        captured.append(env)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr("tools.pre_push.subprocess.run", fake_run)

    assert (
        run_checks(
            (Check("ok", ("python", "check.py")),),
            root=tmp_path,
            dry_run=False,
            python_executable=str(environment_python),
        )
        == 0
    )
    assert captured[0]["PATH"].split(os.pathsep)[0] == str(environment_bin)


def test_pre_push_gate_leaves_bare_command_names_to_the_ambient_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A bare command name carries no directory and must not prepend the cwd."""

    captured: list[dict[str, str]] = []
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
        captured.append(env)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr("tools.pre_push.subprocess.run", fake_run)

    assert (
        run_checks(
            (Check("ok", ("python", "check.py")),),
            root=tmp_path,
            dry_run=False,
            python_executable="python",
        )
        == 0
    )
    assert captured[0]["PATH"] == os.environ.get("PATH", "")


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


def test_pre_push_lint_is_the_environments_ruff_and_not_a_pinned_second_copy() -> None:
    """The pre-push gate must not be red on a checkout that CI accepts.

    `run_checks` starts with `pre-commit run --all-files`, which installs
    whatever revision each remote hook names. A `ruff-pre-commit` revision is a
    second ruff that nothing keeps in step with the one CI's quality job
    installs from `.[dev]`, and the two disagree about which findings exist as
    soon as either moves. The retired `v0.12.7` pin reported RUF012 on
    `flagquantum/compute/pytorch.py`, which the installed ruff does not report,
    so the gate blocked every push from the day that file landed while
    `main` stayed green. Lint therefore runs the installed tool, exactly as the
    black hook beside it does.
    """

    config = yaml.safe_load((ROOT / ".pre-commit-config.yaml").read_text("utf-8"))
    assert "https://github.com/astral-sh/ruff-pre-commit" not in {
        repo["repo"] for repo in config["repos"]
    }

    hooks = {
        hook["id"]: hook
        for repo in config["repos"]
        if repo["repo"] == "local"
        for hook in repo["hooks"]
    }
    ruff = hooks["ruff"]
    assert ruff["language"] == "system"
    assert ruff["entry"] == "ruff check --force-exclude"
    assert ruff["args"] == ["--fix"]
    # The same three trees the CI quality job lints. `--force-exclude` keeps
    # that true if a ruff exclusion is ever configured: without it, an explicit
    # path argument overrides the exclusion and the hook would lint more than
    # CI does.
    assert ruff["files"] == r"^(flagquantum|tests|tools)/.*\.py$"

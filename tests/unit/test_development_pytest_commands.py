"""Complete original pytest wrappers and native checkout invocation paths."""

import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from tests.unit.test_development_tools import REFERENCE
from tools.development import process, pytest_command
from tools.development.changed_tests import changed_test_paths, comparison_revision
from tools.ftqc_source_inputs import FTQC_SOURCE_INVENTORY

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def git(root: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ["git", *arguments], cwd=root, text=True, stderr=subprocess.STDOUT, timeout=15
    ).strip()


@pytest.fixture
def checkout(tmp_path: Path) -> Path:
    git(tmp_path, "init", "-b", "main")
    git(tmp_path, "config", "user.name", "FlagQuantum")
    git(tmp_path, "config", "user.email", "liuwei.chem.phys@gmail.com")
    for relative in ("lib/a.py", "lib/a_test.py", "lib/b_test.py", "lib/unrelated.txt"):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# Initial source\n")
    checks = tmp_path / "check"
    checks.mkdir()
    for name in ("pytest", "pytest-dev-tools", "pytest-changed-files"):
        shutil.copyfile(REFERENCE / "check" / name, checks / name)
    git(tmp_path, "add", ".")
    git(tmp_path, "commit", "-m", "Baseline")
    return tmp_path


def test_complete_frozen_pytest_source_group() -> None:
    selected = {"Q-0030", "Q-0032", "Q-0034"}
    verified = set()
    with (FTQC_SOURCE_INVENTORY).open(encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            if row["id"] not in selected:
                continue
            data = (REFERENCE / row["path"]).read_bytes()
            assert (
                hashlib.sha1(
                    b"blob " + str(len(data)).encode() + b"\0" + data
                ).hexdigest()
                == row["git_blob"]
            )
            verified.add(row["id"])
    assert verified == selected


def run_original(
    root: Path, name: str, arguments: list[str], exit_code: int
) -> tuple[subprocess.CompletedProcess[str], list[str]]:
    binaries = root / "binaries"
    binaries.mkdir(exist_ok=True)
    executable = binaries / "pytest"
    executable.write_text(
        f"#!{sys.executable}\nimport json,os,sys\n"
        "from pathlib import Path\n"
        "Path(os.environ['COMMAND_LOG']).write_text(json.dumps(sys.argv[1:]))\n"
        "raise SystemExit(int(os.environ['COMMAND_EXIT']))\n"
    )
    executable.chmod(0o755)
    log = root / "command.json"
    log.unlink(missing_ok=True)
    env = dict(
        os.environ,
        PATH=str(binaries) + ":" + os.environ["PATH"],
        COMMAND_LOG=str(log),
        COMMAND_EXIT=str(exit_code),
    )
    result = subprocess.run(
        ["bash", str(root / "check" / name), *arguments],
        cwd=root / "lib",
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return result, json.loads(log.read_text()) if log.exists() else []


@pytest.mark.parametrize("exit_code", [0, 1, 2, 5])
@pytest.mark.parametrize(
    "mode,source,paths",
    [
        ("package", "pytest", ["tests/"]),
        ("development", "pytest-dev-tools", ["tools/development/", "tests/unit/"]),
    ],
)
def test_original_default_parallel_flags_forwarding_and_status(
    checkout: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: pytest_command.Mode,
    source: str,
    paths: list[str],
    exit_code: int,
) -> None:
    flags = ["-n0", "-q", "-k", "some expression"]
    result, expected = run_original(checkout, source, flags, exit_code)
    calls = []

    def run(arguments: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append((arguments, kwargs))
        return subprocess.CompletedProcess(arguments, exit_code)

    monkeypatch.setattr(process, "run", run)
    assert (
        pytest_command.run_tests(checkout, mode, flags)
        == result.returncode
        == exit_code
    )
    assert expected == [
        "-n=auto",
        *flags,
        "qualtran/" if mode == "package" else "dev_tools/",
    ]
    arguments, keywords = calls[0]
    assert arguments == [sys.executable, "-m", "pytest", *expected[:-1], *paths]
    assert keywords["cwd"] == checkout and keywords["timeout"] == 1200


def test_revision_defaults_order_missing_invalid_and_noncommit(checkout: Path) -> None:
    assert comparison_revision(checkout, None) == "main"
    git(checkout, "branch", "origin/main")
    assert comparison_revision(checkout, None) == "origin/main"
    git(checkout, "branch", "upstream/main")
    assert comparison_revision(checkout, None) == "upstream/main"
    assert comparison_revision(checkout, "HEAD") == "HEAD"
    for revision in ("does-not-exist", "HEAD:lib/a.py"):
        with pytest.raises(ValueError, match="No revision"):
            comparison_revision(checkout, revision)
        result, calls = run_original(checkout, "pytest-changed-files", [revision], 0)
        assert result.returncode == 1 and not calls
    git(checkout, "checkout", "--detach")
    for revision in ("main", "origin/main", "upstream/main"):
        git(checkout, "branch", "-D", revision)
    with pytest.raises(ValueError, match="No default revision"):
        comparison_revision(checkout, None)
    result, calls = run_original(checkout, "pytest-changed-files", [], 0)
    assert result.returncode == 1 and not calls


@pytest.mark.parametrize("flags", [[], ["-q", "-k", "one or two"]])
def test_original_changed_selection_direct_revision_and_failure(
    checkout: Path, monkeypatch: pytest.MonkeyPatch, flags: list[str]
) -> None:
    (checkout / "lib/a.py").write_text("# Changed implementation\n")
    (checkout / "lib/b_test.py").write_text("# Changed test\n")
    (checkout / "lib/unrelated.txt").write_text("Changed text\n")
    selected = changed_test_paths(checkout, "HEAD")
    assert selected == ["lib/a_test.py", "lib/b_test.py"]
    result, expected = run_original(
        checkout, "pytest-changed-files", ["HEAD", *flags], 7
    )
    real_run = process.run
    captured = []

    def run(arguments: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        if arguments[:3] == [sys.executable, "-m", "pytest"]:
            captured.append(arguments[3:])
            return subprocess.CompletedProcess(arguments, 7)
        return real_run(arguments, **kwargs)

    monkeypatch.setattr(process, "run", run)
    assert (
        pytest_command.run_tests(checkout, "changed", ["HEAD", *flags])
        == result.returncode
        == 7
    )
    assert captured == [expected] == [[*flags, *selected]]
    (checkout / "lib/a_test.py").unlink()
    (checkout / "lib/b_test.py").unlink()
    assert pytest_command.run_tests(checkout, "changed", ["HEAD"]) == 0
    assert len(captured) == 1


def test_native_layout_renames_deletions_staged_and_untracked(checkout: Path) -> None:
    paths = {
        "flagquantum/new.py": "# Module\n",
        "tests/unit/test_new.py": "# Native test\n",
        "lib/test_native.py": "# Local native test\n",
        "lib/native.py": "# Local module\n",
        "lib/a_test.py": "# Existing source test\n",
    }
    for name, text in paths.items():
        path = checkout / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    # Untracked implementations have the original Git diff exclusion.
    assert changed_test_paths(checkout, "HEAD") == ["lib/a_test.py"]
    git(checkout, "add", ".")
    assert changed_test_paths(checkout, "HEAD") == [
        "lib/a_test.py",
        "lib/test_native.py",
        "tests/unit/test_new.py",
    ]
    git(checkout, "commit", "-m", "Native paths")
    git(checkout, "mv", "lib/a.py", "lib/renamed.py")
    (checkout / "lib/renamed_test.py").write_text("# Rename target test\n")
    (checkout / "lib/b_test.py").unlink()
    assert changed_test_paths(checkout, "HEAD") == ["lib/renamed_test.py"]


def test_original_direct_revision_does_not_substitute_merge_base(
    checkout: Path,
) -> None:
    git(checkout, "checkout", "-b", "comparison")
    (checkout / "lib/a.py").write_text("# Comparison branch implementation\n")
    git(checkout, "add", ".")
    git(checkout, "commit", "-m", "Comparison branch change")
    git(checkout, "checkout", "main")
    (checkout / "lib/b_test.py").write_text("# Current branch test\n")
    git(checkout, "add", ".")
    git(checkout, "commit", "-m", "Current branch change")
    assert changed_test_paths(checkout, "comparison") == [
        "lib/a_test.py",
        "lib/b_test.py",
    ]
    result, calls = run_original(checkout, "pytest-changed-files", ["comparison"], 0)
    assert result.returncode == 0
    assert calls == ["lib/a_test.py", "lib/b_test.py"]
    base = git(checkout, "merge-base", "comparison", "HEAD")
    assert changed_test_paths(checkout, base) == ["lib/b_test.py"]


def test_child_timeout_and_launch_failure_are_visible(
    checkout: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def timeout(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired("pytest", 1200)

    monkeypatch.setattr(process, "run", timeout)
    with pytest.raises(subprocess.TimeoutExpired):
        pytest_command.run_tests(checkout, "package", [])

    def missing(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        raise FileNotFoundError("Python executable missing")

    monkeypatch.setattr(process, "run", missing)
    with pytest.raises(FileNotFoundError, match="Python executable missing"):
        pytest_command.run_tests(checkout, "development", [])


def test_actual_native_module_commands(tmp_path: Path) -> None:
    root = tmp_path
    for folder in ("tests/unit", "tools/development", "lib"):
        (root / folder).mkdir(parents=True)
    passing = root / "tests/unit/test_case.py"
    passing.write_text("def test_case():\n    assert True\n")
    git(root, "init", "-b", "main")
    git(root, "config", "user.name", "FlagQuantum")
    git(root, "config", "user.email", "liuwei.chem.phys@gmail.com")
    git(root, "add", ".")
    git(root, "commit", "-m", "Initial native tests")
    env = dict(
        os.environ,
        PYTHONPATH=os.pathsep.join(sys.path),
        PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
    )

    def run(arguments: list[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, "-m", "tools.development.pytest_command", *arguments],
            cwd=root,
            env=env,
            capture_output=True,
            text=True,
            timeout=45,
        )

    # Load xdist explicitly while keeping unrelated environment plugins out.
    for mode in ("package", "development"):
        result = run([mode, "-p", "xdist.plugin", "-n0", "-q"])
        assert result.returncode == 0, result.stdout + result.stderr
        assert "1 passed" in result.stdout
    passing.write_text("def test_case():\n    assert False\n")
    result = run(["changed", "HEAD", "-q"])
    assert result.returncode == 1 and "1 failed" in result.stdout
    assert run(["changed", "invalid-revision"]).returncode == 1
    assert run([]).returncode != 0

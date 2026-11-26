"""Complete check composer source and native corrections to false success."""

import csv
import hashlib
import itertools
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from tests.unit.test_development_pytest_commands import git
from tests.unit.test_development_tools import REFERENCE
from tools.development import (
    check_command,
    coverage_command,
    format_command,
    lint_command,
    typecheck_command,
)
from tools.ftqc_source_inputs import FTQC_SOURCE_INVENTORY

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def checkout(tmp_path: Path) -> Path:
    git(tmp_path, "init", "-b", "main")
    git(tmp_path, "config", "user.name", "FlagQuantum")
    git(tmp_path, "config", "user.email", "liuwei.chem.phys@gmail.com")
    (tmp_path / "check").mkdir()
    shutil.copyfile(REFERENCE / "check/all", tmp_path / "check/all")
    (tmp_path / "a.py").write_text("value = 1\n")
    git(tmp_path, "add", ".")
    git(tmp_path, "commit", "-m", "Baseline")
    return tmp_path


def test_complete_frozen_composer() -> None:
    with (FTQC_SOURCE_INVENTORY).open(encoding="utf-8-sig") as stream:
        row = next(row for row in csv.DictReader(stream) if row["id"] == "Q-0023")
    data = (REFERENCE / row["path"]).read_bytes()
    assert (
        hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
        == row["git_blob"]
    )


def original(
    root: Path,
    arguments: list[str],
    statuses: tuple[int, int, int, int],
    *,
    real_format: bool = False,
) -> tuple[subprocess.CompletedProcess[str], list[list[Any]]]:
    log = root / "events.jsonl"
    log.unlink(missing_ok=True)
    commands = {
        "pylint": statuses[0],
        "pylint-changed-files": statuses[0],
        "mypy": statuses[1],
        "format-incremental": statuses[2],
        "pytest-and-incremental-coverage": statuses[3],
        "pytest-changed-files-and-incremental-coverage": statuses[3],
    }
    for name, status in commands.items():
        path = root / "check" / name
        path.write_text(
            f"#!{sys.executable}\nimport json,sys\nfrom pathlib import Path\n"
            f"with Path({str(log)!r}).open('a') as out:\n"
            f" out.write(json.dumps([{name!r}, sys.argv[1:]]) + '\\n')\n"
            f"raise SystemExit({status})\n"
        )
        path.chmod(0o755)
    if real_format:
        shutil.copyfile(
            REFERENCE / "check/format-incremental", root / "check/format-incremental"
        )
    result = subprocess.run(
        ["bash", str(root / "check/all"), *arguments],
        cwd=root,
        text=True,
        capture_output=True,
        timeout=30,
    )
    return result, (
        [json.loads(line) for line in log.read_text().splitlines()]
        if log.exists()
        else []
    )


def native_recorded(
    root: Path,
    arguments: list[str],
    statuses: tuple[int, int, int, int],
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[int, list[list[Any]]]:
    calls: list[list[Any]] = []

    def lint(directory: Path, mode: lint_command.Mode, rest: list[str]) -> int:
        assert directory == root
        calls.append(["lint", mode, rest])
        return statuses[0]

    def typing(directory: Path, mode: typecheck_command.Mode, rest: list[str]) -> int:
        assert directory == root
        calls.append(["typing", mode, rest])
        return statuses[1]

    def formatting(directory: Path, rest: list[str]) -> int:
        assert directory == root
        calls.append(["format", rest])
        return statuses[2]

    def coverage(
        directory: Path, rest: list[str], *, changed_only: bool = False
    ) -> int:
        assert directory == root
        calls.append(["coverage", changed_only, rest])
        return statuses[3]

    with monkeypatch.context() as patched:
        patched.setattr(lint_command, "run_lint", lint)
        patched.setattr(typecheck_command, "run_typecheck", typing)
        patched.setattr(format_command, "format_files", formatting)
        patched.setattr(coverage_command, "run_coverage", coverage)
        result = check_command.run_checks(root, arguments)
    return result, calls


@pytest.mark.parametrize("statuses", list(itertools.product((0, 7), repeat=4)))
@pytest.mark.parametrize(
    "changed,apply", list(itertools.product((False, True), repeat=2))
)
def test_complete_source_order_and_all_failure_combinations(
    checkout: Path,
    monkeypatch: pytest.MonkeyPatch,
    statuses: tuple[int, int, int, int],
    changed: bool,
    apply: bool,
) -> None:
    arguments = ["HEAD"]
    if changed:
        arguments.append("--only-changed-files")
    if apply:
        arguments.append("--apply-format-changes")
    source, events = original(checkout, arguments, statuses)
    result, calls = native_recorded(checkout, arguments, statuses, monkeypatch)
    assert source.returncode == statuses[-1]
    assert result == int(any(statuses))
    assert events == [
        ["pylint-changed-files" if changed else "pylint", []],
        ["mypy", []],
        ["format-incremental", ["HEAD", "--apply" if apply else ""]],
        [
            (
                "pytest-changed-files-and-incremental-coverage"
                if changed
                else "pytest-and-incremental-coverage"
            ),
            ["HEAD"],
        ],
    ]
    assert calls == [
        ["lint", "changed" if changed else "package", ["HEAD"] if changed else []],
        ["typing", "package", []],
        ["format", ["HEAD", "--apply"] if apply else ["HEAD"]],
        ["coverage", changed, ["HEAD"]],
    ]


@pytest.mark.parametrize(
    "arguments",
    [
        [],
        [
            "--only-changed-files",
            "--only-changed-files",
            "--apply-format-changes",
            "--apply-format-changes",
        ],
    ],
)
def test_defaults_empty_optionals_and_repeated_flags(
    checkout: Path, monkeypatch: pytest.MonkeyPatch, arguments: list[str]
) -> None:
    source, events = original(checkout, arguments, (0, 0, 0, 0))
    result, calls = native_recorded(checkout, arguments, (0, 0, 0, 0), monkeypatch)
    assert result == source.returncode == 0
    assert events[2][1][0] == "" and events[3][1] == [""]
    assert calls[0][2] == [] and calls[-1][2] == []
    assert calls[2][1] == (["--apply"] if arguments else [])


@pytest.mark.parametrize(
    "arguments", [["missing"], ["HEAD", "HEAD"], [""], ["--unknown"]]
)
def test_invalid_input_starts_no_stage(
    checkout: Path, monkeypatch: pytest.MonkeyPatch, arguments: list[str]
) -> None:
    source, events = original(checkout, arguments, (0, 0, 0, 0))
    result, calls = native_recorded(checkout, arguments, (0, 0, 0, 0), monkeypatch)
    assert result == source.returncode == 1 and not calls and not events


def test_original_empty_formatter_argument_is_a_hidden_failure(checkout: Path) -> None:
    source, events = original(checkout, ["HEAD"], (0, 0, 0, 0), real_format=True)
    assert source.returncode == 0
    assert "Too many arguments" in source.stderr
    assert len(events) == 3
    assert format_command.format_files(checkout, ["HEAD"]) == 0


def test_actual_native_pipeline_and_earlier_failure_propagation(checkout: Path) -> None:
    for name, content in {
        "flagquantum/__init__.py": "",
        "flagquantum/values.py": "def value() -> int:\n    return 1\n",
        "tests/unit/test_values.py": "from flagquantum.values import value\n\n\ndef test_value() -> None:\n    assert value() >= 1\n",
    }.items():
        path = checkout / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    config = checkout / "tools/development/coverage.ini"
    config.parent.mkdir(parents=True)
    shutil.copyfile(ROOT / "tools/development/coverage.ini", config)
    shutil.copyfile(ROOT / "pyproject.toml", checkout / "pyproject.toml")
    git(checkout, "add", ".")
    git(checkout, "commit", "-m", "Real native command fixture")
    values = checkout / "flagquantum/values.py"
    values.write_text("def value() -> int:\n    return 2\n")
    env = dict(
        os.environ,
        PYTHONPATH=os.pathsep.join(
            [
                str(checkout),
                str(REFERENCE / "format-command-deps"),
                str(ROOT),
                *sys.path,
            ]
        ),
        PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
        PYTEST_PLUGINS="pytest_cov.plugin,xdist.plugin",
        PYTEST_XDIST_AUTO_NUM_WORKERS="1",
    )

    def invoke(*arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                sys.executable,
                "-m",
                "tools.development.check_command",
                *arguments,
            ],
            cwd=checkout,
            env=env,
            capture_output=True,
            text=True,
            timeout=90,
        )

    for arguments in ((), ("HEAD",), ("HEAD", "--only-changed-files")):
        result = invoke(*arguments)
        assert result.returncode == 0, result.stdout + result.stderr
        assert "All touched lines covered" in result.stdout
        assert not list(checkout.rglob("*.py,cover"))
    content = "import os\n\n\ndef value() -> int:\n    return 2\n"
    values.write_text(content)
    result = invoke("HEAD", "--only-changed-files")
    assert result.returncode == 1, result.stdout + result.stderr
    assert "F401" in result.stdout and "All touched lines covered" in result.stdout
    assert values.read_text() == content
    assert not list(checkout.rglob("*.py,cover"))

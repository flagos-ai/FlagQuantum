"""Frozen typing wrappers and real native strict command execution."""

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

from tests.unit.test_development_pytest_commands import git
from tests.unit.test_development_tools import REFERENCE
from tools.development import process, typecheck_command
from tools.ftqc_source_inputs import FTQC_SOURCE_INVENTORY

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def test_frozen_complete_typing_sources() -> None:
    selected = {"Q-0026", "Q-0027", "Q-0049"}
    verified = set()
    with (FTQC_SOURCE_INVENTORY).open(encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            if row["id"] in selected:
                data = (REFERENCE / row["path"]).read_bytes()
                assert (
                    hashlib.sha1(
                        b"blob " + str(len(data)).encode() + b"\0" + data
                    ).hexdigest()
                    == row["git_blob"]
                )
                verified.add(row["id"])
    assert verified == selected


@pytest.mark.parametrize("mode", ["package", "development"])
@pytest.mark.parametrize("status", [0, 1, 2, 17])
def test_complete_original_wrapper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    mode: typecheck_command.Mode,
    status: int,
) -> None:
    git(tmp_path, "init", "-b", "main")
    checks = tmp_path / "check"
    checks.mkdir()
    source = "mypy" if mode == "package" else "mypy-dev-tools"
    shutil.copyfile(REFERENCE / "check" / source, checks / source)
    binary = tmp_path / "bin"
    binary.mkdir()
    command = binary / "mypy"
    command.write_text(
        f"#!{sys.executable}\nimport json,sys\nfrom pathlib import Path\n"
        f"Path({str(tmp_path / 'arguments.json')!r}).write_text(json.dumps(sys.argv[1:]))\n"
        f"raise SystemExit({status})\n"
    )
    command.chmod(0o755)
    flags = ["--show-error-context", "--cache-dir", "a directory with spaces"]
    original = subprocess.run(
        ["bash", str(checks / source), *flags],
        cwd=binary,
        env=dict(os.environ, PATH=str(binary) + ":" + os.environ["PATH"]),
        capture_output=True,
        text=True,
        timeout=30,
    )
    expected = ["--config-file=dev_tools/conf/mypy.ini"]
    if mode == "development":
        expected.append("--no-namespace-packages")
    expected.extend([*flags, "qualtran/" if mode == "package" else "dev_tools/"])
    assert json.loads((tmp_path / "arguments.json").read_text()) == expected
    calls = []

    def run(arguments: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append((arguments, kwargs))
        return subprocess.CompletedProcess(arguments, status)

    monkeypatch.setattr(process, "run", run)
    assert typecheck_command.run_typecheck(tmp_path, mode, flags) == original.returncode
    assert capsys.readouterr().out == original.stdout == "\033[31m\033[0m"
    arguments, keywords = calls[0]
    native_options = [
        "--config-file=pyproject.toml",
        "--strict",
        "--python-version=3.12",
        "--ignore-missing-imports",
    ]
    if mode == "development":
        native_options.append("--explicit-package-bases")
    assert arguments == [
        sys.executable,
        "-m",
        "mypy",
        *native_options,
        *flags,
        "flagquantum/" if mode == "package" else "tools/development/",
    ]
    assert keywords == dict(
        cwd=tmp_path, check=False, timeout=1200, log_run_to_stderr=False
    )


@pytest.mark.parametrize("mode", ["package", "development"])
@pytest.mark.parametrize("bad", [False, True])
def test_real_cli_strict_namespace_and_error_codes(
    tmp_path: Path, mode: typecheck_command.Mode, bad: bool
) -> None:
    git(tmp_path, "init", "-b", "main")
    (tmp_path / "pyproject.toml").write_text(
        '[tool.mypy]\npython_version = "3.10"\nignore_missing_imports = true\n'
    )
    package = tmp_path / ("flagquantum" if mode == "package" else "tools/development")
    package.mkdir(parents=True)
    (package / "__init__.py").touch()
    if mode == "development":
        for name in ("typecheck_command.py", "process.py", "repository.py"):
            shutil.copyfile(
                Path(typecheck_command.__file__).parent / name, package / name
            )
    (package / "values.py").write_text(
        'def value() -> int:\n    return "wrong"\n'
        if bad
        else "def value() -> int:\n    return 3\n"
    )
    (package / "consumer.py").write_text(
        f"from {package.relative_to(tmp_path).as_posix().replace('/', '.')} import values\n"
        "def twice() -> int:\n    return 2 * values.value()\n"
    )
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[2]))
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "tools.development.typecheck_command",
            mode,
            "--cache-dir=/dev/null",
        ],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        timeout=90,
    )
    assert result.returncode == int(bad), result.stdout + result.stderr
    assert result.stdout.startswith("\033[31m") and result.stdout.endswith("\033[0m")
    assert ("[return-value]" in result.stdout) == bad
    assert "Duplicate module" not in result.stdout


def test_timeout_resets_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def timeout(
        arguments: list[str], **kwargs: Any
    ) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired(arguments, kwargs["timeout"])

    monkeypatch.setattr(process, "run", timeout)
    with pytest.raises(subprocess.TimeoutExpired):
        typecheck_command.run_typecheck(tmp_path, "package", [])
    assert capsys.readouterr().out == "\033[31m\033[0m"


@pytest.mark.parametrize("arguments", [[], ["unknown"]])
def test_invalid_mode(arguments: list[str]) -> None:
    with pytest.raises(
        SystemExit, match="Expected package, development or retained-source"
    ):
        typecheck_command.main(arguments)

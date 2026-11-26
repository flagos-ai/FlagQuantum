"""Complete frozen touched-line coverage CLI and real Git-checkout execution."""

import csv
import hashlib
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests.unit.test_development_tools import REFERENCE
from tests.unit.test_documentation_commands import original_command
from tools.development import incremental_coverage as native
from tools.development import process
from tools.development.environment import PreparedEnvironment
from tools.ftqc_source_inputs import FTQC_SOURCE_INVENTORY

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def test_source_command_arguments_output_and_errors(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    remaining = {"Q-0044"}
    with (FTQC_SOURCE_INVENTORY).open(encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            if row["id"] in remaining:
                data = (REFERENCE / row["path"]).read_bytes()
                assert (
                    hashlib.sha1(
                        b"blob " + str(len(data)).encode() + b"\0" + data
                    ).hexdigest()
                    == row["git_blob"]
                )
                remaining.remove(row["id"])
    assert not remaining
    monkeypatch.chdir(tmp_path)
    calls: list[tuple[str | None, str, str | None]] = []
    uncovered = 0

    def check(environment: PreparedEnvironment) -> int:
        calls.append(
            (
                environment.actual_commit_id,
                environment.compare_commit_id,
                environment.destination_directory,
            )
        )
        return uncovered

    monkeypatch.setattr(native, "check_for_uncovered_lines", check)
    modules: dict[str, dict[str, object]] = {
        "shell_tools": {"highlight": process.highlight, "RED": process.RED},
        "incremental_coverage": {"check_for_uncovered_lines": check},
        "prepared_env": {"PreparedEnv": PreparedEnvironment},
    }
    for arguments, count in (
        ([], 0),
        (["HEAD"], 0),
        (["HEAD~1"], 1),
        (["branch", "ignored-extra"], 7),
        (["HEAD"], -2),
    ):
        uncovered = count
        calls.clear()
        expected = 0
        try:
            original_command(
                monkeypatch,
                "check-incremental-coverage-annotations.py",
                modules,
                arguments,
            )
        except SystemExit as error:
            assert isinstance(error.code, int)
            expected = error.code
        expected_output = capsys.readouterr().out
        expected_calls = list(calls)
        calls.clear()
        assert native.main(arguments) == expected
        assert capsys.readouterr().out == expected_output
        assert calls == expected_calls
        assert calls == ([(None, arguments[0], str(tmp_path))] if arguments else [])
    monkeypatch.setattr(sys, "argv", ["coverage-command", "HEAD"])
    uncovered = 0
    assert native.main() == 0

    def fail(environment: PreparedEnvironment) -> int:
        raise OSError("coverage owner failure")

    monkeypatch.setattr(native, "check_for_uncovered_lines", fail)
    with pytest.raises(OSError, match="coverage owner failure"):
        native.main(["HEAD"])
    modules["incremental_coverage"] = {"check_for_uncovered_lines": fail}
    with pytest.raises(OSError, match="coverage owner failure"):
        original_command(
            monkeypatch, "check-incremental-coverage-annotations.py", modules, ["HEAD"]
        )


def test_real_coverage_command_in_git_checkout(tmp_path: Path) -> None:
    environment = dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path))

    def git(*arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", *arguments],
            cwd=tmp_path,
            env=environment,
            capture_output=True,
            text=True,
            check=True,
            timeout=15,
        )

    def command(*arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                sys.executable,
                "-m",
                "tools.development.incremental_coverage",
                *arguments,
            ],
            cwd=tmp_path,
            env=environment,
            capture_output=True,
            text=True,
            timeout=30,
        )

    git("init")
    git("config", "user.name", "FlagQuantum")
    git("config", "user.email", "liuwei.chem.phys@gmail.com")
    source = tmp_path / "case.py"
    source.write_text("value = 1\n")
    git("add", "case.py")
    git("commit", "-m", "Coverage command fixture")
    result = command("HEAD")
    assert result.returncode == 0 and "All touched lines covered" in result.stdout
    source.write_text("value = 2\n")
    result = command("HEAD")
    assert result.returncode == 1, result.stdout + result.stderr
    assert "uncovered touched lines" in result.stdout
    annotation = tmp_path / "case.py,cover"
    annotation.write_text("! value = 2\n")
    assert command("HEAD").returncode == 1
    annotation.write_text("> value = 2\n")
    result = command("HEAD")
    assert result.returncode == 0 and "All touched lines covered" in result.stdout
    assert command().returncode == 1
    result = command("missing-revision")
    assert result.returncode != 0 and "CalledProcessError" in result.stderr

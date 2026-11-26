"""Complete frozen clean-tree wrapper and real native publication workflow."""

import csv
import hashlib
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

from tests.unit.test_development_tools import REFERENCE
from tools.development import notebook_publication as publication
from tools.development import notebook_publication_check as native
from tools.development.notebook_catalog import notebook_sections
from tools.ftqc_source_inputs import FTQC_SOURCE_INVENTORY

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def git(root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=root, text=True, stderr=subprocess.STDOUT, timeout=15
    )


def test_frozen_shell_source_blob() -> None:
    with (FTQC_SOURCE_INVENTORY).open(encoding="utf-8-sig") as stream:
        row = next(row for row in csv.DictReader(stream) if row["id"] == "Q-0024")
    data = (REFERENCE / row["path"]).read_bytes()
    assert (
        hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
        == row["git_blob"]
    )


@pytest.mark.parametrize("exit_code", [3, 7, 37])
def test_generator_failure_bypasses_git_status(
    exit_code: int, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = Mock(return_value=subprocess.CompletedProcess([], exit_code))
    monkeypatch.setattr(subprocess, "run", runner)
    assert native.check_notebooks(tmp_path) == exit_code
    runner.assert_called_once_with(
        [sys.executable, "-m", "tools.development.notebook_publication"],
        cwd=tmp_path,
        check=False,
        timeout=120,
    )
    generator = tmp_path / "dev_tools/autogenerate-bloqs-notebooks-v2.py"
    generator.parent.mkdir()
    generator.write_text(f"raise SystemExit({exit_code})\n")
    # Use the real subprocess after the native failure path has been observed.
    monkeypatch.undo()
    result = subprocess.run(
        ["bash", str(REFERENCE / "check/autogenerate-notebooks")],
        cwd=tmp_path,
        env=dict(
            os.environ, PATH=str(Path(sys.executable).parent) + ":" + os.environ["PATH"]
        ),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == exit_code
    assert result.stdout == result.stderr == ""


def test_status_failure_and_literal_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    runner = Mock(
        side_effect=[
            subprocess.CompletedProcess([], 0),
            subprocess.CompletedProcess([], 128, stdout=""),
        ]
    )
    monkeypatch.setattr(subprocess, "run", runner)
    assert native.check_notebooks(tmp_path) == 1
    assert capsys.readouterr().out == "\n"
    assert runner.call_args.kwargs["timeout"] == 15
    runner.side_effect = [
        subprocess.CompletedProcess([], 0),
        subprocess.CompletedProcess([], 0, stdout=' M "path with spaces"\n?? *.txt\n'),
    ]
    assert native.check_notebooks(tmp_path) == 1
    assert capsys.readouterr().out == 'M "path with spaces" ?? *.txt\n'


def test_actual_original_and_native_full_publication_check(tmp_path: Path) -> None:
    env = dict(
        os.environ,
        PATH=str(Path(sys.executable).parent) + ":" + os.environ["PATH"],
        PYTHONPATH=os.pathsep.join(sys.path),
    )
    roots = [tmp_path / "original", tmp_path / "native"]
    for root in roots:
        source = root / "examples/notebooks"
        source.mkdir(parents=True)
        names = {path.name for _, paths in notebook_sections(source) for path in paths}
        names.update(publication.CONCEPTS + publication.ROOT_COMPONENTS)
        for name in names:
            (source / name).write_text(
                '{"cells": [], "metadata": {}, "nbformat": 4, "nbformat_minor": 5}\n'
            )
        generator = root / "dev_tools/autogenerate-bloqs-notebooks-v2.py"
        generator.parent.mkdir()
        generator.write_text(
            "from tools.development.notebook_publication import main\nmain()\n"
        )
        (root / ".gitignore").write_text("ignored.txt\n")
        git(root, "init")
        git(root, "config", "user.name", "FlagQuantum")
        git(root, "config", "user.email", "liuwei.chem.phys@gmail.com")
        git(root, "add", ".")
        git(root, "commit", "-m", "Publication input")

    def compare(expected: int) -> None:
        commands = [
            ["bash", str(REFERENCE / "check/autogenerate-notebooks")],
            [sys.executable, "-m", "tools.development.notebook_publication_check"],
        ]
        results = [
            subprocess.run(
                command, cwd=root, env=env, capture_output=True, text=True, timeout=45
            )
            for root, command in zip(roots, commands, strict=True)
        ]
        assert [result.returncode for result in results] == [expected, expected], [
            result.stdout + result.stderr for result in results
        ]
        for result in results:
            assert result.stderr == ""
        assert git(roots[0], "status", "--porcelain") == git(
            roots[1], "status", "--porcelain"
        )
        if expected:
            assert (
                results[0].stdout.splitlines()[-1] == results[1].stdout.splitlines()[-1]
            )

    # New generated outputs are changes, even though publication itself succeeds.
    compare(1)
    for root in roots:
        assert len(list((root / "docs/components").glob("*.ipynb"))) == 120
        assert (root / "docs/components/index.rst").exists()
        git(root, "add", ".")
        git(root, "commit", "-m", "Publication baseline")
        (root / "ignored.txt").write_text("Ignored local file\n")
    compare(0)
    for root in roots:
        (root / "examples/notebooks/unary_iteration.ipynb").write_text(
            "Changed asset\n"
        )
    compare(1)
    for root in roots:
        git(root, "restore", ".")
        (root / "staged.txt").write_text("Unrelated staged change\n")
        git(root, "add", "staged.txt")
    compare(1)
    for root in roots:
        git(root, "reset", "--", "staged.txt")
    compare(1)
    for root in roots:
        (root / "staged.txt").unlink()
    compare(0)

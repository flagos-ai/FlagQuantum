"""Complete locked development helpers and their failure contracts."""

import contextlib
import csv
import hashlib
import importlib.util
import io
import subprocess
import sys
import types
from pathlib import Path
from typing import Any

import pytest

from tools.development import files, process, repository
from tools.ftqc_source_inputs import FTQC_SOURCE_INVENTORY

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
REFERENCE = Path(__file__).resolve().parents[3] / "ftqc-reference-env"


def load_source(name: str) -> types.ModuleType:
    path = REFERENCE / "dev_tools" / "qualtran_dev_tools" / (name + ".py")
    spec = importlib.util.spec_from_file_location(
        "_development_reference." + name, path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_locked_complete_source_blobs() -> None:
    ledger = FTQC_SOURCE_INVENTORY
    selected = {"Q-0054", "Q-0060", "Q-0087", "Q-0088", "Q-0090"}
    verified = set()
    with ledger.open(encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            if row["id"] not in selected:
                continue
            data = (REFERENCE / row["path"]).read_bytes()
            digest = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data)
            assert digest.hexdigest() == row["git_blob"]
            verified.add(row["id"])
    assert verified == selected


@pytest.mark.parametrize(
    "name",
    [
        "test_run_raises_on_failure",
        "test_run_returns_string_output",
        "test_run_with_command_logging",
        "test_output_of",
    ],
)
def test_unmodified_original_shell_tests(
    name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = types.ModuleType("_development_reference")
    setattr(package, "shell_tools", process)
    monkeypatch.setitem(sys.modules, package.__name__, package)
    original = load_source("shell_tools_test")
    subprocess_run = subprocess.run

    def bounded_run(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[Any]:
        kwargs.setdefault("timeout", 5)
        return subprocess_run(*args, **kwargs)

    monkeypatch.setattr(subprocess, "run", bounded_run)
    getattr(original, name)()


@pytest.mark.parametrize(
    "command",
    [
        ("git",),
        ("git", "-n", "--all", "path", "more"),
        ("git", "-", "--"),
        ("git", "", "-n"),
        (),
    ],
)
def test_command_abbreviation(command: tuple[str, ...]) -> None:
    source = load_source("shell_tools")
    if not command:
        for owner in (source, process):
            with pytest.raises(IndexError):
                owner.abbreviate_command_arguments_after_switches(command)
    else:
        assert process.abbreviate_command_arguments_after_switches(
            command
        ) == source.abbreviate_command_arguments_after_switches(command)


@pytest.mark.parametrize("text", ["", "hello", "unicode λ\n\033[0m"])
@pytest.mark.parametrize("bold", [False, True])
def test_highlight(text: str, bold: bool) -> None:
    source = load_source("shell_tools")
    for color in (
        process.BOLD,
        process.DIM,
        process.RED,
        process.GREEN,
        process.YELLOW,
    ):
        assert process.highlight(text, color, bold) == source.highlight(
            text, color, bold
        )


@pytest.mark.parametrize("module", ["shell_tools", "native"])
def test_process_execution_contract(module: str, tmp_path: Path) -> None:
    owner = process if module == "native" else load_source(module)
    command = [sys.executable, "-c", "import sys; sys.stdout.write('λ\\n\\n')"]
    assert owner.output_of(command, timeout=5) == "λ\n"
    assert (
        owner.run(command, capture_output=True, text=False, timeout=5).stdout
        == "λ\n\n".encode()
    )
    assert owner.output_of(
        [sys.executable, "-c", "import os; print(os.getcwd())"], cwd=tmp_path, timeout=5
    ) == str(tmp_path.resolve())
    with pytest.raises(subprocess.CalledProcessError) as error:
        owner.run([sys.executable, "-c", "raise SystemExit(7)"], timeout=5)
    assert error.value.returncode == 7
    assert (
        owner.run(
            [sys.executable, "-c", "raise SystemExit(7)"], check=False, timeout=5
        ).returncode
        == 7
    )
    with pytest.raises(subprocess.TimeoutExpired):
        owner.run([sys.executable, "-c", "import time; time.sleep(2)"], timeout=0.05)
    with pytest.raises(FileNotFoundError):
        owner.run([str(tmp_path / "missing-command")], timeout=5)
    with pytest.raises(TypeError):
        owner.output_of(command, text=False, timeout=5)


@pytest.mark.parametrize("module", ["write_if_different", "native"])
def test_file_lifecycle(module: str, tmp_path: Path) -> None:
    owner = files if module == "native" else load_source(module)
    path = tmp_path / "generated.txt"
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        with owner.WriteIfDifferent(path) as stream:
            assert not stream.closed and stream.writable()
            assert not stream.readable() and not stream.seekable()
            assert stream.write("λ\n") == 2
            stream.writelines(["abc", "def"])
            assert stream.tell() == 8
            assert stream.truncate(5) == 5
            stream.flush()
        assert stream.closed
        stamp = path.stat().st_mtime_ns
        with owner.WriteIfDifferent(path) as unchanged:
            unchanged.write("λ\nabc")
        assert path.stat().st_mtime_ns == stamp
        with (
            pytest.raises(RuntimeError, match="body"),
            owner.WriteIfDifferent(path) as failed,
        ):
            failed.write("partial")
            raise RuntimeError("body")
    assert path.read_text() == "partial"
    assert (
        output.getvalue()
        == f"{path}\t writing.\n{path}\t unchanged.\n{path}\t writing.\n"
    )
    for action in (stream.close, stream.tell, stream.writable):
        with pytest.raises(ValueError):
            action()
    stream.flush()
    assert not stream.readable() and not stream.seekable()
    with pytest.raises(ValueError):
        stream.write("closed")
    absent = owner.WriteIfDifferent(tmp_path / "absent" / "file")
    with pytest.raises(FileNotFoundError):
        absent.close()
    assert absent.closed


def test_repository_root_and_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = load_source("git_tools")
    subprocess.run(
        ["git", "init", "-q", str(tmp_path / "checkout")], check=True, timeout=5
    )
    nested = tmp_path / "checkout" / "nested"
    nested.mkdir()
    monkeypatch.chdir(nested)
    assert repository.get_git_root() == source.get_git_root() == nested.parent.resolve()
    monkeypatch.chdir(tmp_path)
    for owner in (repository, source):
        with pytest.raises(subprocess.CalledProcessError):
            owner.get_git_root()


def test_repository_timeout_boundary(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def bounded(args: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        assert args == ["git", "rev-parse", "--show-toplevel"]
        assert kwargs == dict(
            capture_output=True, universal_newlines=True, check=True, timeout=15
        )
        return subprocess.CompletedProcess(args, 0, stdout=str(tmp_path))

    monkeypatch.setattr(subprocess, "run", bounded)
    assert repository.get_git_root() == tmp_path

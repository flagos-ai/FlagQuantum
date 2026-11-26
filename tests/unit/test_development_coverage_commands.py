"""Complete frozen coverage wrappers, failure order and real native CLI."""

import configparser
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
from importlib import import_module
from itertools import product
from pathlib import Path
from typing import Any

import pytest

from tests.unit.test_development_pytest_commands import git
from tests.unit.test_development_tools import REFERENCE
from tools.development import coverage_command as native
from tools.development import process
from tools.development.repository import merge_base_revision as coverage_revision
from tools.ftqc_source_inputs import FTQC_SOURCE_INVENTORY

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def checkout(tmp_path: Path) -> Path:
    git(tmp_path, "init", "-b", "main")
    git(tmp_path, "config", "user.name", "FlagQuantum")
    git(tmp_path, "config", "user.email", "liuwei.chem.phys@gmail.com")
    for name, content in {
        "lib/a.py": "x = 1\n",
        "lib/a_test.py": "# Existing test\n",
        "lib/old.py,cover": "Tracked annotation\n",
        ".gitignore": "*.py,cover\n",
    }.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    for name in (
        "pytest",
        "pytest-and-incremental-coverage",
        "pytest-changed-files-and-incremental-coverage",
    ):
        destination = tmp_path / "check" / name
        destination.parent.mkdir(exist_ok=True)
        shutil.copyfile(REFERENCE / "check" / name, destination)
        destination.chmod(0o755)
    git(tmp_path, "add", "--force", ".")
    git(tmp_path, "commit", "-m", "Coverage baseline")
    (tmp_path / "lib/a.py").write_text("x = 2\n")
    return tmp_path


def test_frozen_complete_source_and_profile() -> None:
    selected = {"Q-0031", "Q-0033", "Q-0047"}
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
            selected.remove(row["id"])
    assert not selected
    source = configparser.ConfigParser()
    target = configparser.ConfigParser()
    source.read(REFERENCE / "dev_tools/conf/.coveragerc")
    target.read(ROOT / "tools/development/coverage.ini")
    assert dict(source["run"]) == dict(target["run"]) == {"include": "./*"}
    assert source.sections() == target.sections() == ["run"]


def test_profile_runtime_excludes_external_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    coverage = import_module("coverage")
    root = tmp_path / "checkout"
    root.mkdir()
    local = root / "owned.py"
    external = tmp_path / "external.py"
    for path in (local, external):
        path.write_text("value = 23\n")
    monkeypatch.chdir(root)
    observed = []
    for config in (
        REFERENCE / "dev_tools/conf/.coveragerc",
        ROOT / "tools/development/coverage.ini",
    ):
        recorder = coverage.Coverage(config_file=str(config), data_file=None)
        recorder.start()
        try:
            for path in (local, external):
                exec(compile(path.read_text(), str(path), "exec"), {})
        finally:
            recorder.stop()
        observed.append(set(recorder.get_data().measured_files()))
    assert observed == [{str(local)}, {str(local)}]


def run_original(
    root: Path, arguments: list[str], changed: bool, statuses: tuple[int, int, int]
) -> tuple[int, list[list[Any]], str]:
    binaries = root / "binaries"
    binaries.mkdir(exist_ok=True)
    real_git = shutil.which("git")
    assert real_git is not None
    code = f"""#!{sys.executable}
import json,os,subprocess,sys
from pathlib import Path
name=Path(sys.argv[0]).name
args=sys.argv[1:]
roles={{'pytest':'pytest','coverage':'annotate','python':'analyze','git':'cleanup'}}
role=roles[name]
if name!='git' or (args and args[0]=='clean'):
 with Path(os.environ['COMMAND_LOG']).open('a') as out:
  out.write(json.dumps([role,args])+'\\n')
if name=='git':
 raise SystemExit(subprocess.run([{real_git!r},*args]).returncode)
if name=='pytest':
 Path('lib/a.py,cover').write_text('Generated annotation\\n')
raise SystemExit(int(os.environ[role.upper()+'_EXIT']))
"""
    for name in ("pytest", "coverage", "python", "git"):
        path = binaries / name
        path.write_text(code)
        path.chmod(0o755)
    log = root / "command.jsonl"
    log.unlink(missing_ok=True)
    env = dict(
        os.environ,
        PATH=str(binaries) + ":" + os.environ["PATH"],
        COMMAND_LOG=str(log),
        PYTEST_EXIT=str(statuses[0]),
        ANNOTATE_EXIT=str(statuses[1]),
        ANALYZE_EXIT=str(statuses[2]),
    )
    name = (
        "pytest-changed-files-and-incremental-coverage"
        if changed
        else "pytest-and-incremental-coverage"
    )
    result = subprocess.run(
        ["bash", str(root / "check" / name), *arguments],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        timeout=45,
    )
    events = (
        [json.loads(line) for line in log.read_text().splitlines()]
        if log.exists()
        else []
    )
    return result.returncode, events, result.stderr


def run_native_recorded(
    root: Path,
    arguments: list[str],
    changed: bool,
    statuses: tuple[int, int, int],
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[int, list[list[Any]]]:
    original = process.run
    events: list[list[Any]] = []

    def run(args: list[str], **kwargs: Any) -> subprocess.CompletedProcess[Any]:
        if args[:3] == [sys.executable, "-m", "pytest"]:
            events.append(["pytest", args[3:]])
            (root / "lib/a.py,cover").write_text("Generated annotation\n")
            return subprocess.CompletedProcess(args, statuses[0])
        if args[:4] == [sys.executable, "-m", "coverage", "annotate"]:
            events.append(["annotate", args[3:]])
            return subprocess.CompletedProcess(args, statuses[1])
        if args[:3] == [sys.executable, "-m", "tools.development.incremental_coverage"]:
            events.append(["analyze", args[3:]])
            return subprocess.CompletedProcess(args, statuses[2])
        if args[:2] == ["git", "clean"]:
            events.append(["cleanup", args[1:]])
        return original(args, **kwargs)

    with monkeypatch.context() as context:
        context.setattr(process, "run", run)
        result = native.run_coverage(root, arguments, changed_only=changed)
    return result, events


def normalize_original(events: list[list[Any]]) -> list[list[Any]]:
    normalized = []
    for role, args in events:
        if role == "analyze":
            args = args[1:]
        elif role == "pytest":
            args = [
                (
                    native.CONFIG_ARGUMENT
                    if value == "--cov-config=dev_tools/conf/.coveragerc"
                    else "tests/" if value == "qualtran/" else value
                )
                for value in args
            ]
        normalized.append([role, args])
    return normalized


@pytest.mark.parametrize("changed", [False, True])
@pytest.mark.parametrize("statuses", list(product((0, 3), (0, 5), (0, 7))))
def test_complete_original_failure_aggregation_order_and_cleanup(
    checkout: Path,
    monkeypatch: pytest.MonkeyPatch,
    changed: bool,
    statuses: tuple[int, int, int],
) -> None:
    expected, events, _ = run_original(checkout, ["HEAD"], changed, statuses)
    assert not (checkout / "lib/a.py,cover").exists()
    actual, native_events = run_native_recorded(
        checkout, ["HEAD"], changed, statuses, monkeypatch
    )
    assert actual == expected == int(bool(statuses[0] or statuses[2]))
    assert native_events == normalize_original(events)
    assert not (checkout / "lib/a.py,cover").exists()
    assert (checkout / "lib/old.py,cover").read_text() == "Tracked annotation\n"


@pytest.mark.parametrize(
    "arguments",
    [
        ["HEAD", "--no-analyze", "-n0", "-q"],
        ["--no-analyze", "--no-analyze", "-n0"],
        ["HEAD", "-k", "one or two", "-n0"],
    ],
)
def test_full_flags_no_analyze_and_default_revision(
    checkout: Path, monkeypatch: pytest.MonkeyPatch, arguments: list[str]
) -> None:
    expected, events, _ = run_original(checkout, arguments, False, (0, 3, 0))
    actual, native_events = run_native_recorded(
        checkout, arguments, False, (0, 3, 0), monkeypatch
    )
    assert actual == expected == 0
    assert native_events == normalize_original(events)
    assert (checkout / "lib/a.py,cover").exists() == ("--no-analyze" in arguments)


def test_changed_no_tests_ignored_extra_arguments_and_protobuf_dirs(
    checkout: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (checkout / "generated").mkdir()
    (checkout / "generated/messages_pb2.py").write_text("# Generated\n")
    git(checkout, "add", "generated")
    arguments = ["HEAD", "--no-analyze", "--arbitrary-ignored-flag"]
    expected, events, _ = run_original(checkout, arguments, True, (0, 0, 0))
    actual, native_events = run_native_recorded(
        checkout, arguments, True, (0, 0, 0), monkeypatch
    )
    assert actual == expected == 0 and native_events == normalize_original(events)
    assert native.changed_coverage_directories(checkout, "HEAD") == ["lib"]
    (checkout / "lib/a_test.py").unlink()
    expected, events, _ = run_original(checkout, ["HEAD"], True, (7, 7, 7))
    actual, native_events = run_native_recorded(
        checkout, ["HEAD"], True, (7, 7, 7), monkeypatch
    )
    assert actual == expected == 0 and events == native_events == []


def test_merge_base_actual_divergence_and_invalid_revision(
    checkout: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    git(checkout, "restore", "lib/a.py")
    base = git(checkout, "rev-parse", "HEAD")
    git(checkout, "checkout", "-b", "comparison")
    (checkout / "lib/a.py").write_text("x = 3\n")
    git(checkout, "add", "lib/a.py")
    git(checkout, "commit", "-m", "Comparison change")
    git(checkout, "checkout", "main")
    (checkout / "lib/a_test.py").write_text("# Current change\n")
    git(checkout, "add", "lib/a_test.py")
    git(checkout, "commit", "-m", "Current change")
    assert coverage_revision(checkout, "comparison") == base
    for changed in (False, True):
        expected, events, _ = run_original(checkout, ["comparison"], changed, (0, 0, 0))
        actual, native_events = run_native_recorded(
            checkout, ["comparison"], changed, (0, 0, 0), monkeypatch
        )
        assert actual == expected == 0 and native_events == normalize_original(events)
        assert ["analyze", [base]] in native_events
        expected, events, _ = run_original(checkout, ["absent"], changed, (0, 0, 0))
        actual, native_events = run_native_recorded(
            checkout, ["absent"], changed, (0, 0, 0), monkeypatch
        )
        assert actual == expected == 1 and events == native_events == []


def test_missing_default_and_unrelated_history_source_boundaries(
    checkout: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    git(checkout, "restore", "lib/a.py")
    git(checkout, "checkout", "--orphan", "unrelated")
    git(checkout, "commit", "-m", "Unrelated root")
    assert coverage_revision(checkout, "main") == ""
    for changed in (False, True):
        arguments = ["main", "--no-analyze"]
        expected, events, _ = run_original(checkout, arguments, changed, (0, 0, 0))
        actual, native_events = run_native_recorded(
            checkout, arguments, changed, (0, 0, 0), monkeypatch
        )
        assert actual == expected == 0 and native_events == normalize_original(events)
    git(checkout, "branch", "-D", "main")
    for changed in (False, True):
        expected, events, _ = run_original(checkout, [], changed, (0, 0, 0))
        actual, native_events = run_native_recorded(
            checkout, [], changed, (0, 0, 0), monkeypatch
        )
        assert actual == expected == 1 and events == native_events == []


def test_analyzer_timeout_keeps_error_and_cleans_annotations(
    checkout: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = process.run

    def run(args: list[str], **kwargs: Any) -> subprocess.CompletedProcess[Any]:
        if args[:3] == [sys.executable, "-m", "pytest"]:
            (checkout / "lib/a.py,cover").write_text("Generated annotation\n")
            return subprocess.CompletedProcess(args, 0)
        if args[:3] == [sys.executable, "-m", "tools.development.incremental_coverage"]:
            raise subprocess.TimeoutExpired(args, 120)
        return original(args, **kwargs)

    monkeypatch.setattr(process, "run", run)
    with pytest.raises(subprocess.TimeoutExpired):
        native.run_coverage(checkout, ["HEAD"], changed_only=True)
    assert not (checkout / "lib/a.py,cover").exists()
    assert (checkout / "lib/old.py,cover").exists()


def test_real_native_coverage_cli_and_profile(tmp_path: Path) -> None:
    for name, content in {
        "lib/a.py": "def value(x):\n    return x + 1\n",
        "lib/a_test.py": "from lib.a import value\n\ndef test_value():\n    assert value(1) > 0\n",
        "tests/unit/test_a.py": "from lib.a import value\n\ndef test_value():\n    assert value(1) > 0\n",
    }.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    config = tmp_path / "tools/development/coverage.ini"
    config.parent.mkdir(parents=True)
    shutil.copyfile(ROOT / "tools/development/coverage.ini", config)
    git(tmp_path, "init", "-b", "main")
    git(tmp_path, "config", "user.name", "FlagQuantum")
    git(tmp_path, "config", "user.email", "liuwei.chem.phys@gmail.com")
    git(tmp_path, "add", ".")
    git(tmp_path, "commit", "-m", "Real coverage baseline")
    env = dict(
        os.environ,
        PYTHONPATH=os.pathsep.join([str(tmp_path), *sys.path]),
        PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
        PYTEST_PLUGINS="pytest_cov.plugin,xdist.plugin",
    )

    def run(mode: str, *arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                sys.executable,
                "-m",
                "tools.development.coverage_command",
                mode,
                "HEAD",
                *arguments,
            ],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
        )

    (tmp_path / "lib/a.py").write_text("def value(x):\n    return x + 2\n")
    for mode, arguments in (("package", ("-n0", "-q")), ("changed", ())):
        result = run(mode, *arguments)
        assert result.returncode == 0, result.stdout + result.stderr
        assert "All touched lines covered" in result.stdout
        assert not list(tmp_path.rglob("*.py,cover"))
    (tmp_path / "lib/a.py").write_text(
        "def value(x):\n    return x + 2\n\ndef unused():\n    return 99\n"
    )
    result = run("changed")
    assert result.returncode == 1, result.stdout + result.stderr
    assert "uncovered touched lines" in result.stdout
    assert not list(tmp_path.rglob("*.py,cover"))
    result = run("package", "--no-analyze", "-n0", "-q")
    assert result.returncode == 0 and "All touched lines covered" not in result.stdout
    assert (tmp_path / ".coverage").exists()

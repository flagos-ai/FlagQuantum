"""Complete original formatter orchestration and native tool integration."""

import csv
import hashlib
import importlib.metadata
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
from tools.development import format_command, process
from tools.ftqc_source_inputs import FTQC_SOURCE_INVENTORY

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
DEPENDENCIES = REFERENCE / "format-command-deps"


@pytest.fixture
def checkout(tmp_path: Path) -> Path:
    git(tmp_path, "init", "-b", "main")
    git(tmp_path, "config", "user.name", "FlagQuantum")
    git(tmp_path, "config", "user.email", "liuwei.chem.phys@gmail.com")
    checks = tmp_path / "check"
    checks.mkdir()
    shutil.copyfile(
        REFERENCE / "check/format-incremental", checks / "format-incremental"
    )
    (tmp_path / "a.py").write_text("value = 1\n")
    git(tmp_path, "add", ".")
    git(tmp_path, "commit", "-m", "Baseline")
    return tmp_path


def test_frozen_formatter_and_direct_dependency_sources() -> None:
    selected = {"Q-0025", "Q-0096", "Q-0105", "Q-0137"}
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


def test_complete_format_dependency_declaration() -> None:
    from packaging.requirements import Requirement
    from packaging.utils import canonicalize_name

    lines = (
        (REFERENCE / "dev_tools/requirements/deps/format.txt").read_text().splitlines()
    )
    requirements = [
        Requirement(line) for line in lines if line and not line.startswith("#")
    ]
    assert [str(requirement) for requirement in requirements] == [
        "flynt~=0.60",
        "black~=24.8.0",
        "isort~=5.10.1",
    ]
    installed = {
        str(canonicalize_name(distribution.metadata["Name"])): distribution
        for distribution in importlib.metadata.distributions(
            path=[str(REFERENCE / "lint-deps"), str(DEPENDENCIES)]
        )
    }
    for requirement in requirements:
        distribution = installed[canonicalize_name(requirement.name)]
        assert distribution.version in requirement.specifier
        assert any("license" in str(path).lower() for path in distribution.files or [])
    assert installed["flynt"].version == "0.78"
    assert installed["black"].version == "24.8.0"
    assert installed["isort"].version == "5.10.1"
    assert "astor" in installed


def original(
    root: Path,
    arguments: list[str],
    statuses: tuple[int, int, int],
    *,
    version_status: int = 0,
) -> tuple[subprocess.CompletedProcess[str], list[list[str]]]:
    binary = root / "bin"
    binary.mkdir(exist_ok=True)
    log = root / "commands.jsonl"
    log.unlink(missing_ok=True)
    for name, status in zip(("flynt", "isort", "black"), statuses, strict=True):
        executable = binary / name
        executable.write_text(
            f"#!{sys.executable}\nimport json,sys\nfrom pathlib import Path\n"
            f"with Path({str(log)!r}).open('a') as out:\n"
            f" out.write(json.dumps([{name!r}, *sys.argv[1:]]) + '\\n')\n"
            "if '--version' in sys.argv:\n print('test-black-version')\n"
            f" raise SystemExit({version_status})\n"
            f"raise SystemExit({status})\n"
        )
        executable.chmod(0o755)
    result = subprocess.run(
        ["bash", str(root / "check/format-incremental"), *arguments],
        cwd=binary,
        env=dict(os.environ, PATH=str(binary) + ":" + os.environ["PATH"]),
        capture_output=True,
        text=True,
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
    statuses: tuple[int, int, int],
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[int, list[list[str]]]:
    actual = process.run
    calls = []
    status_map = dict(zip(("flynt", "ruff", "black"), statuses, strict=True))

    def run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        if command[0] == "git":
            return actual(command, **kwargs)
        assert command[:2] == [sys.executable, "-m"]
        calls.append(command[2:])
        assert kwargs["cwd"] == root and not kwargs["check"]
        assert kwargs["timeout"] == (15 if "--version" in command else 1200)
        return subprocess.CompletedProcess(
            command, status_map[command[2]], "test-black-version\n", ""
        )

    with monkeypatch.context() as patched:
        patched.setattr(process, "run", run)
        result = format_command.format_files(root, arguments)
    return result, calls


@pytest.mark.parametrize("apply", [False, True])
@pytest.mark.parametrize("statuses", list(itertools.product((0, 7), repeat=3)))
def test_complete_original_command_order_and_failure_matrix(
    checkout: Path,
    monkeypatch: pytest.MonkeyPatch,
    apply: bool,
    statuses: tuple[int, int, int],
) -> None:
    (checkout / "a.py").write_text("value = 2\n")
    arguments = ["HEAD", "--apply"] if apply else ["HEAD"]
    upstream, source_calls = original(checkout, arguments, statuses)
    result, native_calls = native_recorded(checkout, arguments, statuses, monkeypatch)
    assert result == upstream.returncode == int(any(statuses))
    assert source_calls == [
        ["flynt", "a.py", "--quiet", "-f", *([] if apply else ["--dry-run"])],
        ["isort", *([] if apply else ["--check", "--diff"]), "a.py"],
        ["black", "--version"],
        ["black", "--color", *([] if apply else ["--check", "--diff"]), "a.py"],
    ]
    assert native_calls == [
        source_calls[0],
        ["ruff", "check", "--select", "I", "--fix" if apply else "--diff", "a.py"],
        *source_calls[2:],
    ]


def test_version_failure_does_not_change_formatter_status(
    checkout: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, calls = original(checkout, ["--all"], (0, 0, 0), version_status=9)
    actual = process.run

    def run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        if command[0] == "git":
            return actual(command, **kwargs)
        return subprocess.CompletedProcess(
            command, 9 if "--version" in command else 0, "", ""
        )

    monkeypatch.setattr(process, "run", run)
    assert format_command.format_files(checkout, ["--all"]) == source.returncode == 0
    assert [call[0] for call in calls] == ["flynt", "isort", "black", "black"]


def test_repeated_flags_and_empty_revision(
    checkout: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    arguments = ["", "--all", "--apply", "--apply", "--all"]
    source, calls = original(checkout, arguments, (0, 0, 0))
    result, native = native_recorded(checkout, arguments, (0, 0, 0), monkeypatch)
    assert result == source.returncode == 0 and native[0] == calls[0]


@pytest.mark.parametrize("all_files", [False, True])
def test_actual_git_selection_filters_renames_and_untracked(
    checkout: Path, monkeypatch: pytest.MonkeyPatch, all_files: bool
) -> None:
    for name in (
        "old.py",
        "deleted.py",
        "lib/protos/generated.py",
        "lib/generated_pb2.py",
        "has spaces.py",
    ):
        path = checkout / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("value = 1\n")
    git(checkout, "add", ".")
    git(checkout, "commit", "-m", "Tracked files")
    git(checkout, "mv", "old.py", "renamed.py")
    (checkout / "deleted.py").unlink()
    for name in (
        "a.py",
        "has spaces.py",
        "lib/protos/generated.py",
        "lib/generated_pb2.py",
    ):
        (checkout / name).write_text("value = 2\n")
    (checkout / "untracked.py").write_text("value = 2\n")
    arguments = ["--all"] if all_files else ["HEAD"]
    source, calls = original(checkout, arguments, (0, 0, 0))
    result, native = native_recorded(checkout, arguments, (0, 0, 0), monkeypatch)
    assert result == source.returncode == 0
    assert native[0] == calls[0]
    paths = calls[0][1:-3]
    assert "untracked.py" not in paths and "renamed.py" in paths
    if all_files:
        assert "lib/generated_pb2.py" not in paths
        assert "lib/protos/generated.py" in paths and "deleted.py" in paths
    else:
        assert "lib/generated_pb2.py" in paths
        assert "lib/protos/generated.py" not in paths and "deleted.py" not in paths


@pytest.mark.parametrize(
    "arguments", [[], ["missing"], ["HEAD", "HEAD"], ["--all", "missing"]]
)
def test_empty_and_invalid_inputs(
    checkout: Path, monkeypatch: pytest.MonkeyPatch, arguments: list[str]
) -> None:
    source, calls = original(checkout, arguments, (0, 0, 0))
    result, native = native_recorded(checkout, arguments, (0, 0, 0), monkeypatch)
    assert result == source.returncode and not calls and not native


def test_divergent_history_uses_mergebase_and_unrelated_history_source_edge(
    checkout: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    git(checkout, "checkout", "-b", "other")
    (checkout / "other.py").write_text("value = 1\n")
    git(checkout, "add", ".")
    git(checkout, "commit", "-m", "Other branch")
    git(checkout, "checkout", "main")
    (checkout / "a.py").write_text("value = 3\n")
    source, calls = original(checkout, ["other"], (0, 0, 0))
    result, native = native_recorded(checkout, ["other"], (0, 0, 0), monkeypatch)
    assert result == source.returncode == 0 and native[0] == calls[0]
    assert calls[0][1] == "a.py" and "other.py" not in calls[0]
    git(checkout, "checkout", "--orphan", "unrelated")
    git(checkout, "add", ".")
    git(checkout, "commit", "-m", "Unrelated")
    source, calls = original(checkout, ["main"], (0, 0, 0))
    result, native = native_recorded(checkout, ["main"], (0, 0, 0), monkeypatch)
    assert result == source.returncode == 0 and not calls and not native


def test_real_native_check_apply_roundtrip_and_source_semantics(checkout: Path) -> None:
    env = dict(
        os.environ,
        PYTHONPATH=os.pathsep.join(
            [str(DEPENDENCIES), str(Path(__file__).resolve().parents[2])]
        ),
    )
    version = subprocess.run(
        [sys.executable, "-m", "flynt", "--version"],
        env=env,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert version.returncode == 0 and "0.78" in version.stdout
    content = "import sys\nimport os\nname='world'\nvalue='Hello {}'.format(name)\n"
    (checkout / "a.py").write_text(content)

    def invoke(*arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, "-m", "tools.development.format_command", *arguments],
            cwd=checkout,
            env=env,
            text=True,
            capture_output=True,
            timeout=60,
        )

    checked = invoke("HEAD")
    assert checked.returncode == 1, checked.stdout + checked.stderr
    assert (checkout / "a.py").read_text() == content
    applied = invoke("HEAD", "--apply")
    assert applied.returncode == 1, applied.stdout + applied.stderr
    native = (checkout / "a.py").read_text()
    assert ".format" not in native and 'f"Hello {name}"' in native
    assert native.startswith("import os\nimport sys\n")
    assert invoke("HEAD").returncode == 0
    (checkout / "a.py").write_text(content)
    shutil.copyfile(REFERENCE / "pyproject.toml", checkout / "pyproject.toml")
    source = subprocess.run(
        ["bash", str(checkout / "check/format-incremental"), "HEAD", "--apply"],
        cwd=checkout,
        env=dict(env, PATH=str(DEPENDENCIES / "bin") + ":" + os.environ["PATH"]),
        text=True,
        capture_output=True,
        timeout=60,
    )
    assert source.returncode == 1, source.stdout + source.stderr
    original_text = (checkout / "a.py").read_text()
    assert original_text != native
    original_scope: dict[str, Any] = {}
    native_scope: dict[str, Any] = {}
    exec(original_text, original_scope)
    exec(native, native_scope)
    assert original_scope["value"] == native_scope["value"] == "Hello world"

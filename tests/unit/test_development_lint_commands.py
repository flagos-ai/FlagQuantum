"""Complete lint wrapper sources, retained profile and actual native commands."""

import configparser
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
from tools.development import lint_command, process
from tools.ftqc_source_inputs import FTQC_SOURCE_INVENTORY

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def checkout(tmp_path: Path) -> Path:
    git(tmp_path, "init", "-b", "main")
    git(tmp_path, "config", "user.name", "FlagQuantum")
    git(tmp_path, "config", "user.email", "liuwei.chem.phys@gmail.com")
    (tmp_path / "check").mkdir()
    for name in ("pylint", "pylint-changed-files"):
        shutil.copyfile(REFERENCE / "check" / name, tmp_path / "check" / name)
    (tmp_path / "a.py").write_text("value = 1\n")
    git(tmp_path, "add", ".")
    git(tmp_path, "commit", "-m", "Baseline")
    return tmp_path


def test_frozen_complete_lint_sources_and_profile() -> None:
    selected = {"Q-0028", "Q-0029", "Q-0048"}
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
    source = configparser.ConfigParser()
    source.read(REFERENCE / "dev_tools/conf/.pylintrc")
    native = configparser.ConfigParser()
    native.read(Path(lint_command.__file__).with_name("retained_source_lint.ini"))
    expected = {section: dict(source[section]) for section in source.sections()}
    expected["MASTER"]["load-plugins"] = expected["MASTER"]["load-plugins"].replace(
        ",pylint_copyright_checker", ",tools.development.copyright_checker"
    )
    assert {section: dict(native[section]) for section in native.sections()} == expected


def original(
    root: Path, mode: str, arguments: list[str], status: int
) -> tuple[subprocess.CompletedProcess[str], dict[str, Any] | None]:
    binary = root / "bin"
    binary.mkdir(exist_ok=True)
    log = root / "command.json"
    log.unlink(missing_ok=True)
    executable = binary / "pylint"
    executable.write_text(
        f"#!{sys.executable}\nimport json,os,sys\nfrom pathlib import Path\n"
        f"Path({str(log)!r}).write_text(json.dumps(dict(arguments=sys.argv[1:],cwd=os.getcwd(),pythonpath=os.environ.get('PYTHONPATH'))))\n"
        f"raise SystemExit({status})\n"
    )
    executable.chmod(0o755)
    source = "pylint" if mode == "package" else "pylint-changed-files"
    result = subprocess.run(
        ["bash", str(root / "check" / source), *arguments],
        cwd=binary,
        env=dict(os.environ, PATH=str(binary) + ":" + os.environ["PATH"]),
        capture_output=True,
        text=True,
        timeout=30,
    )
    return result, json.loads(log.read_text()) if log.exists() else None


def native_recorded(
    root: Path,
    mode: lint_command.Mode,
    arguments: list[str],
    status: int,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[int, list[list[str]]]:
    actual = process.run
    calls = []

    def run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        if command[0] == "git":
            return actual(command, **kwargs)
        calls.append(command)
        assert kwargs == dict(
            cwd=root, check=False, timeout=1200, log_run_to_stderr=False
        )
        return subprocess.CompletedProcess(command, status)

    with monkeypatch.context() as patched:
        patched.setattr(process, "run", run)
        result = lint_command.run_lint(root, mode, arguments)
    return result, calls


@pytest.mark.parametrize("status", [0, 1, 2, 8, 32, 127])
@pytest.mark.parametrize("mode", ["package", "changed"])
def test_complete_original_status_arguments_and_environment(
    checkout: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: lint_command.Mode,
    status: int,
) -> None:
    (checkout / "a.py").write_text("value = 2\n")
    arguments = (
        ["--output-format=json"] if mode == "package" else ["HEAD", "--fix", "ignored"]
    )
    source, record = original(checkout, mode, arguments, status)
    result, native = native_recorded(checkout, mode, arguments, status, monkeypatch)
    assert result == source.returncode == status
    assert record is not None
    assert record["cwd"] == str(checkout)
    assert record["pythonpath"] == "dev_tools:dev_tools/qualtran_dev_tools"
    source_tail = (
        [*arguments, "qualtran/", "dev_tools/"] if mode == "package" else ["a.py"]
    )
    native_tail = (
        [*arguments, "flagquantum/", "tools/development/"]
        if mode == "package"
        else ["a.py"]
    )
    assert record["arguments"] == [
        "--jobs=0",
        "--rcfile=dev_tools/conf/.pylintrc",
        *source_tail,
    ]
    assert native == [
        [
            sys.executable,
            "-m",
            "ruff",
            "check",
            "--force-exclude",
            "--config=pyproject.toml",
            *native_tail,
        ]
    ]


def test_changed_paths_include_nonpackage_python_and_ignore_deletions(
    checkout: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in ("old.py", "deleted.py", "has spaces.py", "other/file_pb2.py"):
        path = checkout / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("value = 1\n")
    git(checkout, "add", ".")
    git(checkout, "commit", "-m", "Tracked Python sources")
    git(checkout, "mv", "old.py", "renamed.py")
    (checkout / "deleted.py").unlink()
    for name in ("has spaces.py", "other/file_pb2.py"):
        (checkout / name).write_text("value = 2\n")
    (checkout / "untracked.py").write_text("value = 3\n")
    source, record = original(checkout, "changed", ["--ignored", "HEAD"], 0)
    result, calls = native_recorded(
        checkout, "changed", ["--ignored", "HEAD"], 0, monkeypatch
    )
    assert result == source.returncode == 0 and record is not None
    expected = ["has spaces.py", "other/file_pb2.py", "renamed.py"]
    assert record["arguments"][2:] == calls[0][6:] == expected


@pytest.mark.parametrize("arguments", [[], ["missing"], ["HEAD:check/pylint"]])
def test_zero_changes_and_invalid_revision(
    checkout: Path, monkeypatch: pytest.MonkeyPatch, arguments: list[str]
) -> None:
    source, record = original(checkout, "changed", arguments, 0)
    result, calls = native_recorded(checkout, "changed", arguments, 0, monkeypatch)
    assert result == source.returncode and not calls and record is None


def test_divergent_mergebase_and_unrelated_history_source_defect(
    checkout: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    git(checkout, "checkout", "-b", "other")
    (checkout / "other.py").write_text("value = 1\n")
    git(checkout, "add", ".")
    git(checkout, "commit", "-m", "Other branch")
    git(checkout, "checkout", "main")
    (checkout / "a.py").write_text("value = 2\n")
    source, record = original(checkout, "changed", ["other"], 0)
    result, calls = native_recorded(checkout, "changed", ["other"], 0, monkeypatch)
    assert result == source.returncode == 0 and record is not None
    assert record["arguments"][2:] == calls[0][6:] == ["a.py"]
    git(checkout, "checkout", "--orphan", "unrelated")
    git(checkout, "add", ".")
    git(checkout, "commit", "-m", "Unrelated")
    source, record = original(checkout, "changed", ["main"], 0)
    result, calls = native_recorded(checkout, "changed", ["main"], 0, monkeypatch)
    assert result == source.returncode == 0 and not calls and record is None


@pytest.mark.parametrize("mode", ["package", "changed"])
@pytest.mark.parametrize("bad", [False, True])
def test_real_native_ruff_cli(
    checkout: Path, mode: lint_command.Mode, bad: bool
) -> None:
    shutil.copyfile(ROOT / "pyproject.toml", checkout / "pyproject.toml")
    for name in ("flagquantum", "tools/development"):
        (checkout / name).mkdir(parents=True)
        (checkout / name / "values.py").write_text("value = 1\n")
    content = "value = undefined_value\n" if bad else "value = 2\n"
    path = checkout / "flagquantum/values.py"
    path.write_text(content)
    git(checkout, "add", ".")
    arguments = ["HEAD", "--fix"] if mode == "changed" else []
    result = subprocess.run(
        [sys.executable, "-m", "tools.development.lint_command", mode, *arguments],
        cwd=checkout,
        env=dict(os.environ, PYTHONPATH=str(ROOT)),
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == int(bad), result.stdout + result.stderr
    assert ("F821" in result.stdout) == bad
    assert path.read_text() == content


@pytest.mark.parametrize(
    "case", ["valid", "holder", "undefined", "parameters", "truncated"]
)
def test_actual_retained_source_profile_and_original_diagnostics(
    checkout: Path, case: str
) -> None:
    notice = (
        b"\n".join(
            (REFERENCE / "dev_tools/qualtran_dev_tools/pylint_copyright_checker.py")
            .read_bytes()
            .splitlines()[:13]
        )
        + b"\n"
    )
    bodies = {
        "valid": b"value = 1\n",
        "holder": b"value = 1\n",
        "undefined": b"value = missing_name\n",
        "parameters": b'def transform(value, other):\n    """Return a value.\n\n    Args:\n        value (int): The input.\n    """\n    return value + other\n',
        "truncated": b"",
    }
    if case == "holder":
        notice = notice.replace(b"Google LLC", b"Other Corp")
    if case == "truncated":
        notice = b"\n".join(notice.splitlines()[:2]) + b"\n"
    path = checkout / "notice.py"
    path.write_bytes(notice + bodies[case])
    env = dict(
        os.environ,
        PYTHONPATH=os.pathsep.join(
            [
                str(REFERENCE / "lint-deps"),
                str(REFERENCE / "dev_tools/qualtran_dev_tools"),
                str(ROOT),
            ]
        ),
    )
    options = ["--jobs=1", "--output-format=json", str(path)]
    source = subprocess.run(
        [
            sys.executable,
            "-m",
            "pylint",
            "--jobs=0",
            "--rcfile=" + str(REFERENCE / "dev_tools/conf/.pylintrc"),
            *options,
        ],
        cwd=checkout,
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
    )
    native = subprocess.run(
        [
            sys.executable,
            "-m",
            "tools.development.lint_command",
            "retained-source",
            *options,
        ],
        cwd=checkout,
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
    )
    source_messages = json.loads(source.stdout)
    native_messages = json.loads(native.stdout)
    if case == "truncated":
        assert source.returncode == 0 and not source_messages
        assert (
            native.returncode == 8
            and native_messages[0]["symbol"] == "wrong-or-nonexistent-copyright-notice"
        )
    else:
        assert native.returncode == source.returncode, native.stdout + native.stderr
        assert native_messages == source_messages
        assert bool(native_messages) == (case != "valid")


@pytest.mark.parametrize("arguments", [[], ["unknown"]])
def test_invalid_mode(arguments: list[str]) -> None:
    with pytest.raises(
        SystemExit, match="Expected package, changed or retained-source"
    ):
        lint_command.main(arguments)

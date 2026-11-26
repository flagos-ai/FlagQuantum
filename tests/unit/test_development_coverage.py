"""Full commit-context and incremental annotation source closure."""

import subprocess
import sys
import types
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from tests.unit.test_development_tools import REFERENCE, load_source
from tools.development import incremental_coverage as coverage
from tools.development import process
from tools.development.environment import PreparedEnvironment
from tools.ftqc_source_inputs import FTQC_SOURCE_INVENTORY

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.fixture
def source(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[types.ModuleType, types.ModuleType]:
    package = types.ModuleType("_development_reference")
    setattr(package, "shell_tools", load_source("shell_tools"))
    monkeypatch.setitem(sys.modules, package.__name__, package)
    environment = load_source("prepared_env")
    monkeypatch.setitem(sys.modules, package.__name__ + ".prepared_env", environment)
    original = load_source("incremental_coverage")
    return original, environment


@pytest.fixture(autouse=True)
def bounded_processes(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    original = subprocess.run

    def run(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[Any]:
        kwargs.setdefault("timeout", 5)
        return original(*args, **kwargs)

    monkeypatch.setattr(subprocess, "run", run)
    yield


def test_complete_reference_blobs() -> None:
    import csv
    import hashlib

    selected = {"Q-0061", "Q-0062", "Q-0083"}
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
                selected.remove(row["id"])
    assert not selected


def test_complete_original_ignored_line_suite(monkeypatch: pytest.MonkeyPatch) -> None:
    package = types.ModuleType("_development_reference")
    setattr(package, "incremental_coverage", coverage)
    monkeypatch.setitem(sys.modules, package.__name__, package)
    original_tests = load_source("incremental_coverage_test")
    original_tests.test_determine_ignored_lines()


@pytest.mark.parametrize(
    "diff",
    [
        [],
        ["header", "@@ -1 +2 @@", "+a"],
        ["@@ -1,3 +0,0 @@"],
        ["@@ -1,3 +4,2 @@", "@@ -7 +6 @@"],
        ["@@ -1 +2,0 @@", "@@ -1 +2 @@"],
    ],
)
def test_all_diff_ranges(
    source: tuple[types.ModuleType, types.ModuleType], diff: list[str]
) -> None:
    assert coverage.diff_to_new_interesting_lines(diff) == source[
        0
    ].diff_to_new_interesting_lines(diff)


@pytest.mark.parametrize(
    "line",
    [
        "",
        "! x = 2  \n",
        "> x = 1",
        "  # comment",
        "def f():",
        "class X:",
        "pass",
        "import os",
        "from x import y",
        "else:",
        "main(a)",
        "app.run(main)",
        "raise NotImplementedError('x')",
        "assert False",
        "except ImportError:",
        "plt.show()",
        "figure.show()",
        "fig = plt.subplots()",
        "...",
        'value = "#"',
        "x = 2",
    ],
)
def test_annotation_and_manual_line_rules(
    source: tuple[types.ModuleType, types.ModuleType], line: str
) -> None:
    original = source[0]
    assert coverage.fix_line_from_coverage_file(
        line
    ) == original.fix_line_from_coverage_file(line)
    assert coverage.line_content_counts_as_uncovered_manual(
        line
    ) == original.line_content_counts_as_uncovered_manual(line)
    for annotated in (False, True):
        assert coverage.line_counts_as_uncovered(
            line, annotated
        ) == original.line_counts_as_uncovered(line, annotated)


@pytest.mark.parametrize(
    "path",
    [
        "module.py",
        "tools/check.py",
        "dev_tools/check.py",
        "rtd_docs/conf.py",
        "x_pb2.py",
        "x_pb2_grpc.py",
        "setup.py",
        "pkg/_version.py",
        "cirq-google/cirq_google/cloud/api.py",
        "cirq-google/cirq_google/api/v1/api.py",
        "benchmarks/a.py",
        "module.txt",
    ],
)
def test_file_rules(
    source: tuple[types.ModuleType, types.ModuleType], path: str
) -> None:
    assert coverage.is_applicable_python_file(path) == source[
        0
    ].is_applicable_python_file(path)


def test_source_parser_limits(
    source: tuple[types.ModuleType, types.ModuleType],
) -> None:
    for owner in (coverage, source[0]):
        with pytest.raises(IndexError):
            owner.determine_ignored_lines("if TYPE_CHECKING:")
        with pytest.raises(ValueError):
            owner.diff_to_new_interesting_lines(["@@ invalid"])
        assert owner.line_counts_as_uncovered('x = "#"', False)
        lines = ["if True:", "    a = 2", "", "stop = 3"]
        assert owner.naive_find_end_of_scope(lines, 1) == 3


def git(root: Path, *arguments: str) -> str:
    return process.output_of(["git", *arguments], cwd=root, timeout=5)


@pytest.fixture
def checkout(tmp_path: Path) -> tuple[Path, str, str]:
    git(tmp_path, "init", "-q")
    git(tmp_path, "config", "user.name", "FlagQuantum")
    git(tmp_path, "config", "user.email", "liuwei.chem.phys@gmail.com")
    path = tmp_path / "module.py"
    path.write_text("import os\nx = 1\ny = 2\n")
    git(tmp_path, "add", "module.py")
    git(
        tmp_path,
        "-c",
        "core.hooksPath=/dev/null",
        "commit",
        "--no-gpg-sign",
        "-qm",
        "base",
    )
    base = git(tmp_path, "rev-parse", "HEAD")
    path.write_text("import os\nx = 3\ny = 4\n")
    git(tmp_path, "add", "module.py")
    git(
        tmp_path,
        "-c",
        "core.hooksPath=/dev/null",
        "commit",
        "--no-gpg-sign",
        "-qm",
        "change",
    )
    return tmp_path, base, git(tmp_path, "rev-parse", "HEAD")


def test_environment_revisions_and_uncommitted_paths(
    source: tuple[types.ModuleType, types.ModuleType], checkout: tuple[Path, str, str]
) -> None:
    root, base, head = checkout
    for actual, compare in [(head, base), (head, head), (None, base)]:
        native = PreparedEnvironment(actual, compare, str(root))
        original = source[1].PreparedEnv(actual, compare, str(root))
        assert vars(native) == vars(original)
        assert (
            native.get_changed_files() == original.get_changed_files() == ["module.py"]
        )
    (root / "module.py").unlink()
    (root / "untracked.py").write_text("untracked = 1\n")
    for owner in (PreparedEnvironment, source[1].PreparedEnv):
        assert owner(None, head, str(root)).get_changed_files() == ["module.py"]
        with pytest.raises(subprocess.CalledProcessError):
            owner(None, "invalid-revision", str(root)).get_changed_files()


@pytest.mark.parametrize("annotated", [False, True])
def test_complete_incremental_report(
    source: tuple[types.ModuleType, types.ModuleType],
    checkout: tuple[Path, str, str],
    annotated: bool,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root, base, head = checkout
    monkeypatch.chdir(root)
    path = root / "module.py"
    if annotated:
        Path(str(path) + ",cover").write_text("  import os\n> x = 3\n! y = 4\n")
    expected = (
        [(3, " y = 4", "is new or changed")]
        if annotated
        else [(2, "x = 3", "is new or changed"), (3, "y = 4", "is new or changed")]
    )
    assert (
        coverage.get_incremental_uncovered_lines(str(path), base, head)
        == source[0].get_incremental_uncovered_lines(str(path), base, head)
        == expected
    )
    native_env = PreparedEnvironment(head, base, str(root))
    source_env = source[1].PreparedEnv(head, base, str(root))
    capsys.readouterr()
    assert source[0].check_for_uncovered_lines(source_env) == len(expected)
    original_output = capsys.readouterr().out
    assert coverage.check_for_uncovered_lines(native_env) == len(expected)
    assert capsys.readouterr().out == original_output
    assert (
        coverage.get_incremental_uncovered_lines(str(root / "deleted.py"), base, head)
        == []
    )
    clean = PreparedEnvironment(None, head, str(root))
    assert coverage.check_for_uncovered_lines(clean) == 0
    assert "All touched lines covered" in capsys.readouterr().out


def test_native_bounded_git_queries(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[tuple[list[str], dict[str, Any]]] = []

    def output(args: list[str], **kwargs: Any) -> str:
        calls.append((args, kwargs))
        return ""

    monkeypatch.setattr(process, "output_of", output)
    assert PreparedEnvironment(None, "base", None).get_changed_files() == []
    assert calls.pop() == (
        ["git", "diff", "--name-only", "base", "--"],
        dict(cwd=None, timeout=15),
    )
    path = tmp_path / "module.py"
    path.write_text("x = 1\n")
    assert coverage.get_incremental_uncovered_lines(str(path), "base", None) == []
    assert calls.pop() == (
        ["git", "diff", "--unified=0", "base", "--", str(path)],
        dict(timeout=15),
    )


def test_missing_report_destination_preserves_source_failure(
    source: tuple[types.ModuleType, types.ModuleType],
    checkout: tuple[Path, str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, base, head = checkout
    monkeypatch.chdir(root)
    for owner, environment in (
        (coverage, PreparedEnvironment(head, base, None)),
        (source[0], source[1].PreparedEnv(head, base, None)),
    ):
        with pytest.raises(
            TypeError, match="expected str, bytes or os.PathLike object, not NoneType"
        ):
            owner.check_for_uncovered_lines(environment)

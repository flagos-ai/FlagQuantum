# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 check/pytest-and-incremental-coverage and
# check/pytest-changed-files-and-incremental-coverage.

"""Run tests and analyze annotated coverage of changed lines."""

import sys
from collections.abc import Sequence
from pathlib import Path

from tools.development import process
from tools.development.changed_tests import changed_test_paths
from tools.development.environment import PreparedEnvironment
from tools.development.pytest_command import run_tests
from tools.development.repository import get_git_root
from tools.development.repository import merge_base_revision as coverage_revision

CONFIG_ARGUMENT = "--cov-config=tools/development/coverage.ini"


def changed_coverage_directories(root: Path, revision: str) -> list[str]:
    """Select existing parent directories of changed non-protobuf Python files."""
    paths = PreparedEnvironment(None, revision, str(root)).get_changed_files()
    return sorted(
        {
            str(Path(name).parent)
            for name in paths
            if name.endswith(".py")
            and not name.endswith("_pb2.py")
            and (root / Path(name).parent).exists()
        }
    )


def run_coverage(
    root: Path, arguments: Sequence[str], *, changed_only: bool = False
) -> int:
    """Run the complete source coverage workflow and aggregate its exit status.

    Package mode accepts pytest arguments and --no-analyze. Changed-only mode
    retains the source's first-revision-only argument handling. Analysis removes
    untracked *.py,cover files using the source's Git-clean pattern.
    """
    remaining = list(arguments)
    revision = (
        remaining.pop(0)
        if remaining and remaining[0] and not remaining[0].startswith("-")
        else None
    )
    try:
        revision = coverage_revision(root, revision)
    except ValueError as error:
        print(str(error), file=sys.stderr)
        return 1
    if changed_only:
        # The source's failed merge-base/diff pipelines produce no tests.
        if not revision:
            print("No changed files with associated python tests.", file=sys.stderr)
            return 0
        tests = changed_test_paths(root, revision)
        if not tests:
            print("No changed files with associated python tests.", file=sys.stderr)
            return 0
        coverage = [
            "--cov=" + name for name in changed_coverage_directories(root, revision)
        ]
        pytest_result = process.run(
            [
                sys.executable,
                "-m",
                "pytest",
                *tests,
                *coverage,
                "--cov-report=annotate",
                CONFIG_ARGUMENT,
            ],
            cwd=root,
            check=False,
            timeout=1200,
            log_run_to_stderr=False,
        ).returncode
        analyze = True
    else:
        analyze = "--no-analyze" not in remaining
        options = [value for value in remaining if value != "--no-analyze"]
        pytest_result = run_tests(root, "package", ["--cov", CONFIG_ARGUMENT, *options])
        if analyze:
            # The original ignores annotation conversion's status and still
            # invokes its analyzer and cleanup. Preserve that observable order.
            process.run(
                [sys.executable, "-m", "coverage", "annotate"],
                cwd=root,
                check=False,
                timeout=120,
                log_run_to_stderr=False,
            )
    coverage_result = 0
    if analyze:
        try:
            coverage_result = process.run(
                [
                    sys.executable,
                    "-m",
                    "tools.development.incremental_coverage",
                    revision,
                ],
                cwd=root,
                check=False,
                timeout=120,
                log_run_to_stderr=False,
            ).returncode
        finally:
            process.run(
                ["git", "clean", "--force", "--quiet", "-x", "--", "*.py,cover"],
                cwd=root,
                check=False,
                timeout=15,
                log_run_to_stderr=False,
            )
    return int(bool(pytest_result or coverage_result))


def main(argv: Sequence[str] | None = None) -> None:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments or arguments[0] not in {"package", "changed"}:
        raise SystemExit("Expected package or changed followed by coverage arguments")
    changed_only = arguments.pop(0) == "changed"
    raise SystemExit(run_coverage(get_git_root(), arguments, changed_only=changed_only))


if __name__ == "__main__":
    main()

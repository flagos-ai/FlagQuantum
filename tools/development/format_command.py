# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 check/format-incremental.

"""Format changed or tracked Python files with domain-owned development tools."""

import sys
from collections.abc import Sequence
from pathlib import Path

from tools.development import process
from tools.development.repository import (
    comparison_revision,
    get_git_root,
    merge_base_revision,
)


def format_files(root: Path, arguments: Sequence[str]) -> int:
    """Select source-equivalent paths and run all three formatting phases.

    Flynt retains the original f-string conversion. Existing Ruff import sorting
    and Black configuration own native formatting. Every formatter runs even
    when an earlier formatter returns a failure code.
    """
    apply = False
    all_files = False
    revision = ""
    try:
        for argument in arguments:
            if argument == "--apply":
                apply = True
            elif argument == "--all":
                all_files = True
            elif not revision:
                revision = comparison_revision(root, argument) if argument else ""
            else:
                raise ValueError(
                    "Too many arguments. Expected [revision] [--apply] [--all]."
                )
        if all_files:
            print("Formatting all python files.", file=sys.stderr)
            selection = ["git", "ls-files", "*.py", ":(exclude)*_pb2.py"]
        else:
            revision = merge_base_revision(root, revision or None)
            selection = [
                "git",
                "diff",
                "--name-only",
                "--diff-filter=MAR",
                revision,
                "--",
                "*.py",
                ":(exclude)*/protos/*.py",
            ]
    except ValueError as error:
        print(f"\033[31m{error}\033[0m", file=sys.stderr)
        return 1
    selected = process.run(
        selection,
        cwd=root,
        check=False,
        capture_output=True,
        timeout=15,
        log_run_to_stderr=False,
    ).stdout.splitlines()
    if not selected:
        print("\033[32mNo files to format\033[0m.")
        return 0
    flynt_options = ["--quiet", "-f"]
    sort_options = ["--fix"] if apply else ["--diff"]
    black_options = ["--color"]
    if not apply:
        flynt_options.append("--dry-run")
        black_options.extend(["--check", "--diff"])
    commands = [
        [sys.executable, "-m", "flynt", *selected, *flynt_options],
        [
            sys.executable,
            "-m",
            "ruff",
            "check",
            "--select",
            "I",
            *sort_options,
            *selected,
        ],
    ]
    statuses = []
    for command in commands:
        statuses.append(
            process.run(
                command,
                cwd=root,
                check=False,
                timeout=1200,
                log_run_to_stderr=False,
            ).returncode
        )
    version = process.run(
        [sys.executable, "-m", "black", "--version"],
        cwd=root,
        check=False,
        capture_output=True,
        timeout=15,
        log_run_to_stderr=False,
    ).stdout.rstrip("\n")
    print(f"Running the black formatter... (version: {version})", flush=True)
    statuses.append(
        process.run(
            [sys.executable, "-m", "black", *black_options, *selected],
            cwd=root,
            check=False,
            timeout=1200,
            log_run_to_stderr=False,
        ).returncode
    )
    return int(any(status != 0 for status in statuses))


def main(argv: Sequence[str] | None = None) -> None:
    arguments = list(sys.argv[1:] if argv is None else argv)
    raise SystemExit(format_files(get_git_root(), arguments))


if __name__ == "__main__":
    main()

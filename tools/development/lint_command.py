# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 check/pylint and check/pylint-changed-files.

"""Run native lint scopes or an explicit retained-source policy audit."""

import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from tools.development import process
from tools.development.repository import get_git_root, merge_base_revision

Mode = Literal["package", "changed", "retained-source"]


def run_lint(root: Path, mode: Mode, arguments: Sequence[str]) -> int:
    """Run the selected lint policy and retain its exact child status.

    Changed mode retains the source's first-revision-only argument handling.
    Retained-source mode forwards explicit file paths and flags to the complete
    frozen lint profile, including the accepted imported-source notice checker.
    """
    if mode == "retained-source":
        profile = Path(__file__).with_name("retained_source_lint.ini")
        command = [
            sys.executable,
            "-m",
            "pylint",
            "--jobs=0",
            f"--rcfile={profile}",
            *arguments,
        ]
    else:
        options = list(arguments)
        paths = ["flagquantum/", "tools/development/"]
        if mode == "changed":
            revision = (
                options[0]
                if options and options[0] and not options[0].startswith("-")
                else None
            )
            try:
                revision = merge_base_revision(root, revision)
            except ValueError as error:
                print(f"\033[31m{error}\033[0m", file=sys.stderr)
                return 1
            paths = process.run(
                [
                    "git",
                    "diff",
                    "--diff-filter=d",
                    "--name-only",
                    revision,
                    "--",
                    "*.py",
                ],
                cwd=root,
                check=False,
                capture_output=True,
                timeout=15,
                log_run_to_stderr=False,
            ).stdout.splitlines()
            print(
                f"Found {len(paths)} lintable files associated with changes.",
                file=sys.stderr,
            )
            if not paths:
                return 0
            options = []
        command = [
            sys.executable,
            "-m",
            "ruff",
            "check",
            "--force-exclude",
            "--config=pyproject.toml",
            *options,
            *paths,
        ]
    return int(
        process.run(
            command,
            cwd=root,
            check=False,
            timeout=1200,
            log_run_to_stderr=False,
        ).returncode
    )


def main(argv: Sequence[str] | None = None) -> None:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments or arguments[0] not in {"package", "changed", "retained-source"}:
        raise SystemExit(
            "Expected package, changed or retained-source followed by lint arguments"
        )
    mode = arguments.pop(0)
    if mode == "package":
        result = run_lint(get_git_root(), "package", arguments)
    elif mode == "changed":
        result = run_lint(get_git_root(), "changed", arguments)
    else:
        result = run_lint(get_git_root(), "retained-source", arguments)
    raise SystemExit(result)


if __name__ == "__main__":
    main()

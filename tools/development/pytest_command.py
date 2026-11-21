# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 check/pytest, pytest-dev-tools and pytest-changed-files.

"""Run native full, developer or filename-selected pytest commands."""

import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from tools.development import process
from tools.development.changed_tests import changed_test_paths, comparison_revision
from tools.development.repository import get_git_root

Mode = Literal["package", "development", "changed"]


def run_tests(root: Path, mode: Mode, arguments: Sequence[str]) -> int:
    """Run the selected pytest scope and retain its exact exit status."""
    rest = list(arguments)
    if mode == "changed":
        revision = rest.pop(0) if rest and not rest[0].startswith("-") else None
        try:
            revision = comparison_revision(root, revision)
        except ValueError as error:
            print(str(error), file=sys.stderr)
            return 1
        print(f"Comparing against revision {revision!r}.", file=sys.stderr)
        paths = changed_test_paths(root, revision)
        print(
            f"Found {len(paths)} test files associated with changes.", file=sys.stderr
        )
        if not paths:
            return 0
        options = rest
    else:
        paths = (
            ["tests/"] if mode == "package" else ["tools/development/", "tests/unit/"]
        )
        options = ["-n=auto", *rest]
    return int(
        process.run(
            [sys.executable, "-m", "pytest", *options, *paths],
            cwd=root,
            check=False,
            timeout=1200,
            log_run_to_stderr=False,
        ).returncode
    )


def main(argv: Sequence[str] | None = None) -> None:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments or arguments[0] not in {"package", "development", "changed"}:
        raise SystemExit(
            "Expected package, development or changed followed by pytest arguments"
        )
    mode = arguments.pop(0)
    if mode == "package":
        result = run_tests(get_git_root(), "package", arguments)
    elif mode == "development":
        result = run_tests(get_git_root(), "development", arguments)
    else:
        result = run_tests(get_git_root(), "changed", arguments)
    raise SystemExit(result)


if __name__ == "__main__":
    main()

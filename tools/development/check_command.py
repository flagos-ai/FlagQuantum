# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 check/all.

"""Compose native lint, typing, formatting and incremental coverage checks."""

import sys
from collections.abc import Sequence
from pathlib import Path

from tools.development import (
    coverage_command,
    format_command,
    lint_command,
    typecheck_command,
)
from tools.development.repository import comparison_revision, get_git_root


def run_checks(root: Path, arguments: Sequence[str]) -> int:
    """Run every check stage and fail when any stage fails.

    Explicit revisions reach every comparison stage. Empty optional arguments
    are omitted. Both behaviors correct defects in the original shell composer.
    """
    changed_only = False
    apply = False
    revision = None
    try:
        for argument in arguments:
            if argument == "--only-changed-files":
                changed_only = True
            elif argument == "--apply-format-changes":
                apply = True
            elif revision is None:
                if not argument:
                    raise ValueError("No revision ''")
                revision = comparison_revision(root, argument)
            else:
                raise ValueError(
                    "Invalid arguments. Expected [revision] [--only-changed-files] "
                    "[--apply-format-changes]."
                )
    except ValueError as error:
        print(f"\033[31m{error}\033[0m", file=sys.stderr)
        return 1
    comparison = [revision] if revision is not None else []
    statuses = []
    print("Running native lint", flush=True)
    if changed_only:
        statuses.append(lint_command.run_lint(root, "changed", comparison))
    else:
        statuses.append(lint_command.run_lint(root, "package", []))
    print("Running strict package typing", flush=True)
    statuses.append(typecheck_command.run_typecheck(root, "package", []))
    print("Running incremental format", flush=True)
    formatting = [*comparison, "--apply"] if apply else comparison
    statuses.append(format_command.format_files(root, formatting))
    print("Running tests and incremental coverage", flush=True)
    statuses.append(
        coverage_command.run_coverage(root, comparison, changed_only=changed_only)
    )
    return int(any(status != 0 for status in statuses))


def main(argv: Sequence[str] | None = None) -> None:
    arguments = list(sys.argv[1:] if argv is None else argv)
    raise SystemExit(run_checks(get_git_root(), arguments))


if __name__ == "__main__":
    main()
